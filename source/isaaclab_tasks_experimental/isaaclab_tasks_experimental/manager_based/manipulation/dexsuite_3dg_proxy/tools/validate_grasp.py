# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

###########################################################################
# Validate Grasp — Allegro Hand + VBD Deformable Object
#
# Standalone Newton script that validates the same VBD+MuJoCo physics
# coupling used by the dexsuite_3dg_proxy RL training task.  All default
# physics parameters are intentionally kept in sync with
# proxy_newton_cfg.py so that behaviour observed here transfers directly
# to the training environment.
#
# Gravity setup: gravity is disabled during APPROACH/CLOSE/HOLD so the
# object floats at OBJECT_POS.  This isolates contact mechanics from
# gravity dynamics while building the grasp.  At the start of SETTLE
# gravity is enabled; the 30-frame SETTLE phase lets the grasp re-balance
# under load before the LIFT begins.  Gravity remains on during LIFT so
# the test validates the grasp against a realistic downward pull.
#
# Scripted phases:
#   APPROACH  fingers open; wrist descends from WRIST_START_Z to WRIST_GRASP_Z
#   CLOSE     fingers driven to grasp pose; wrist holds at WRIST_GRASP_Z
#   HOLD      everything held; contacts monitored for stability
#   SETTLE    gravity enabled; grasp re-balances for 30 frames
#   LIFT      wrist rises; fingers stay closed — object should follow
#
# Per-frame stdout log (use to verify the animation timeline):
#   phase      — current phase name and step / total
#   wrist      — wrist root position [m]           (verifies hand motion)
#   state      — human-readable phase description   (verifies phase logic)
#   contacts   — active particle–rigid contact count
#   expected   — expected contact range at this phase
#   obj_com    — deformable object CoM [m]          (verifies object is carried)
#
# Contact timeline expectations:
#   APPROACH early  → 0 contacts   (hand above object, no touch)
#   APPROACH late   → 0 contacts   (fingers still open, approaching)
#   CLOSE           → rising count (fingers pressing into object surface)
#   HOLD/LIFT       → stable count (>= HOLD_CONTACT_THRESHOLD)
#
# -------------------------------------------------------------------------
# Physics coupling architecture: same-substep two-way coupling
# -------------------------------------------------------------------------
# Both this script and proxy_newton_manager.py use Newton's two-solver
# pattern with full same-substep two-way coupling:
#   1. SolverMuJoCo  — integrates the Allegro rigid body (joints, fingers)
#   2. SolverVBD     — integrates only the soft-body particles
#                      (integrate_with_external_rigid_solver=True)
#
# Substep order:
#   a. collide(s0, contacts)         — detect contacts at current positions
#   b. apply_soft_body_reactions()   — accumulate reaction forces into
#                                      state.body_f via Newton's third law:
#                                      NORMAL + FRICTION (full coupling)
#   c. MuJoCo.step(s0 → s1)          — reads body_f; fingers feel resistance
#   d. VBD.step(s0 → s1, contacts)   — uses same contacts; pushes particles out
#
# This is operator-splitting (IMEX) with zero time lag: action and reaction
# are computed from the same contact geometry within a single substep.
# The shared kernel and helper live in vbd_coupling.py and are imported
# by both this script and proxy_newton_manager.py.
#
# Full two-way coupling (normal + friction):
#   The reaction fed into body_f mirrors VBD's evaluate_body_particle_contact():
#   - Normal:  F = n * ke * penetration
#   - Friction: F = -mu * ke * penetration * u_t / |u_t|   (IPC-regularised)
#   where mu = sqrt(soft_contact_mu * shape_material_mu) and
#   u_t = tangential relative slip = (particle_Δx - body_surface_Δx)_⊥n.
#   Contact normals are roughly horizontal; friction is vertical — this is
#   the force the actuators need to carry the object against gravity (LIFT).
#
# -------------------------------------------------------------------------
# Command (from repo root /mnt/dev/isaac-newton3):
#   micromamba run -n env_isaaclab3 python \
#       IsaacLab/source/isaaclab_tasks_experimental/\
#       isaaclab_tasks_experimental/manager_based/manipulation/\
#       dexsuite_3dg_proxy/tools/validate_grasp.py \
#       --viewer gl
#
# Default tet mesh is blueHairRagdoll100k_tet.msh — the RL training mesh.
# Override with --tet-mesh path/to/other.msh (requires retuning --particle-radius).
#
# Physics params to tune when the grasp fails:
#   --particle-radius   increase if fingers pass through the object
#   --soft-contact-ke   increase for harder contact surface (default 1e4)
#   --soft-contact-kd   MUST stay near zero (default 1e-5); Newton VBD
#                       implements this as position-level stiffness
#                       kd*ke/dt — a value of 100 gives 5e8 N/m, causing
#                       immediate VBD divergence at first contact
#   --k-damp            MUST stay near zero (default 1e-5); same
#                       kd*ke/dt stiffness mechanism applies to the
#                       material damping term
#   --k-mu              increase for stiffer shear resistance
#   --vbd-iterations    increase (20-30) if particles explode at contact
#   OBJECT_POS          adjust z if palm misses the object vertically
#   WRIST_GRASP_Z       adjust if approach stops too high/low
###########################################################################

from __future__ import annotations

import argparse
import copy
import itertools
import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode
from newton.solvers import SolverNotifyFlags

from isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_3dg_proxy.config.kuka_allegro.physic.newton.vbd_coupling import (
    VBD_MAX_CONTACTS,
    apply_soft_body_reactions,
)

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------

# Hand base orientation (xyzw).
# The Newton allegro example uses (0.21643, 0.706218, -0.648166, 0.185191) which
# results in palm-UP (ceiling).  For a top-down grasp we apply an additional
# 180° rotation around X, which flips the palm to face -Z (floor).
# Computed as: q_x180=(1,0,0,0) * q_example = (0.185191, 0.648166, 0.706218, -0.21643)
HAND_ROTATION = wp.normalize(wp.quat(0.185191, 0.648166, 0.706218, -0.21643))

# Deformable object: placed at a known fixed position in space [m].
# With gravity disabled the object stays here; any drift is a physics bug.
OBJECT_POS = np.array([-0.07, 0.0, 0.55], dtype=np.float32)

# Wrist geometry.
# The wrist root joint is PALM_OFFSET m above the palm centre.
# WRIST_GRASP_Z positions the palm slightly above OBJECT_POS so the fingers
# reach down around the object without the palm crashing into it from above.
# If the palm still goes too low: increase PALM_OFFSET.
# If the fingers can't reach the object: decrease PALM_OFFSET.
# Wrist root XY offset [m].
# The Allegro palm centre is not coincident with the wrist root joint.
# After HAND_ROTATION the palm is displaced in world-space Y.
# HAND_Y shifts the wrist root in -Y to centre the palm over OBJECT_POS.
# Tune HAND_Y until the wrist-xyz log shows the palm over the object.
HAND_X = 0.0
HAND_Y = -0.15  # negative → moves hand left in camera view (camera faces -X, right=+Y)

# WRIST_GRASP_Z: wrist root z during CLOSE/HOLD.
# Empirically: contacts first appear (palm hits object top) when wrist ≈ 0.60.
# At 0.58 the palm is ~2 cm closer to the object, giving the fingers better
# reach around the sides without the palm digging in.
# Open fingers very wide (FINGER_OPEN_FRAC=0.05) so they spread below/around
# the object at this height; CLOSE sweeps them inward from the sides.
# If palm still pushes object: raise WRIST_GRASP_Z toward 0.62.
# If contacts never form during CLOSE: lower toward 0.56.
WRIST_GRASP_Z = 0.635
WRIST_START_Z = WRIST_GRASP_Z + 0.35  # = 0.93

# Fraction of [lower, upper] range for the open/close finger targets.
# 0.05 = nearly fully open → fingers spread wide below the palm → fingers
#        reach object level and sweep in from below during CLOSE.
# Increase toward 0.30 if fingers collide with the object when opening.
FINGER_OPEN_FRAC  = 0.05
FINGER_CLOSE_FRAC = 0.80

# Ragdoll initial orientation.
# The mesh is modelled standing upright (long axis along +Z).
# Two successive rotations are applied:
#   1. 90° around world X  →  figure lies flat (long axis along -Y)
#   2. 90° around world Z  →  figure rotates in the horizontal plane (long axis along +X)
# Combined quaternion (xyzw) = q_Z(90°) * q_X(90°) = (0.5, 0.5, 0.5, 0.5), ‖q‖=1.
# Adjust here if the ragdoll needs a different alignment relative to the fingers.
RAGDOLL_ROT = wp.quat(0.5, 0.5, 0.5, 0.5)

# Scripted phase durations [frames].
#   APPROACH  — wrist descends from WRIST_START_Z to WRIST_GRASP_Z; fingers open
#   CLOSE     — fingers driven closed; wrist holds at WRIST_GRASP_Z
#   HOLD      — everything held; contacts monitored for stability
PHASE_NAMES  = ["APPROACH", "CLOSE", "HOLD", "SETTLE", "LIFT"]
PHASE_STEPS  = {"APPROACH": 100, "CLOSE": 120, "HOLD": 50, "SETTLE": 30, "LIFT": 80}
TOTAL_FRAMES = sum(PHASE_STEPS.values())  # = 380

# Expected contact ranges per phase for the diagnostic log.
# Format: (min_expected, max_expected, description).
EXPECTED_CONTACTS: dict[str, tuple[int, int, str]] = {
    "APPROACH": (0,    0,  "0 — fingers open, hand above/around object"),
    "CLOSE":    (0, 9999,  "rising as fingers press into object surface"),
    "HOLD":     (1, 9999,  ">= HOLD_CONTACT_THRESHOLD — stable grasp"),
    "SETTLE":   (1, 9999,  "non-zero — grasp holds while gravity loads the object"),
    "LIFT":     (0, 9999,  "non-zero if object is carried against gravity; 0 if dropped"),
}

# Hold success: minimum contact count that must be reached during HOLD.
HOLD_CONTACT_THRESHOLD = 10

# Default asset paths — relative to the task's own assets/ directory.
# Override with ASSETS_ROOT env var for containerised deployments.
_ASSETS_ROOT = os.environ.get(
    "ASSETS_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "assets")),
)
# blueHairRagdoll100k_tet.msh is the RL training mesh (~255 particles, 673 tets).
# blueHairRagdollLR.msh is 5.4x denser (1386 particles) — harder to stabilise.
# Always test with the training mesh so physics behaviour transfers to the task.
DEFAULT_TET_MESH = os.path.join(_ASSETS_ROOT, "blueHairRagdoll100k_tet.msh")
# Dense mesh: 5.4× more particles (~1386 vs ~255), ~10mm inter-particle spacing.
# Use with --dense to test whether asset resolution is the root cause of grasp failure.
# Requires retuning --particle-radius (0.010 vs 0.015 for the sparse mesh).
DEFAULT_DENSE_TET_MESH = os.path.join(_ASSETS_ROOT, "blueHairRagdollLR.msh")
# Rigid mode: full-resolution surface mesh USD used as a non-deformable rigid body.
# Rigid mode: surface mesh OBJ used as a non-deformable rigid body.
# The USD (blueHairRagdoll16k.usd) has a broken instanceable_meshes.usd reference.
DEFAULT_RIGID_OBJ = os.path.join(_ASSETS_ROOT, "blueHairRagdoll16k.obj")


# ---------------------------------------------------------------------------
# Warp helpers
# ---------------------------------------------------------------------------

@wp.kernel
def _set_transform(
    arr: wp.array(dtype=wp.transform),
    idx: int,
    xform: wp.transform,
):
    """Set arr[idx] = xform without a CPU↔GPU round-trip for the full array."""
    if wp.tid() == 0:
        arr[idx] = xform


@wp.kernel
def _kernel_sum_contact_force_magnitudes(
    contact_count:    wp.array(dtype=wp.int32),
    contact_particle: wp.array(dtype=wp.int32),
    contact_shape:    wp.array(dtype=wp.int32),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_normal:   wp.array(dtype=wp.vec3),
    particle_q:       wp.array(dtype=wp.vec3),
    particle_radius:  wp.array(dtype=wp.float32),
    body_q:           wp.array(dtype=wp.transform),
    shape_body:       wp.array(dtype=wp.int32),
    soft_contact_ke:  float,
    force_sum_out:    wp.array(dtype=wp.float32),
):
    """Accumulate the sum of individual contact force magnitudes [N].

    Unlike summing body_f (which cancels opposing forces across bodies),
    this gives the true total force energy injected into the system per step.
    One thread per contact slot; threads beyond actual count early-exit.
    """
    tid = wp.tid()
    if tid >= contact_count[0]:
        return

    s_idx    = contact_shape[tid]
    body_idx = shape_body[s_idx]
    if body_idx < 0:
        return  # static shape — no rigid body to associate force with

    X_wb = body_q[body_idx]
    bx   = wp.transform_point(X_wb, contact_body_pos[tid])
    n    = contact_normal[tid]

    penetration = -(wp.dot(n, particle_q[contact_particle[tid]] - bx) - particle_radius[contact_particle[tid]])
    if penetration <= 0.0:
        return

    f_mag = soft_contact_ke * penetration  # |F| = ke * penetration [N]
    wp.atomic_add(force_sum_out, 0, f_mag)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load_tet_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a Gmsh .msh file → (nodes [V,3] float32, tets [T,4] int32)."""
    try:
        import meshio
    except ImportError:
        raise ImportError("meshio is required: pip install meshio")
    mesh = meshio.read(path)
    nodes = mesh.points.astype(np.float32)
    tets = mesh.cells_dict.get("tetra")
    if tets is None:
        raise ValueError(f"No tetrahedral cells in {path}. Run tools/mesh_to_tet.py first.")
    return nodes, tets.astype(np.int32)


def _load_surface_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a surface mesh file → (vertices [V,3] float32, triangles [F,3] int32).

    Supports any format meshio can read (.obj, .stl, .ply, …).
    """
    try:
        import meshio
    except ImportError:
        raise ImportError("meshio is required: pip install meshio")
    mesh = meshio.read(path)
    verts = mesh.points.astype(np.float32)
    tris = mesh.cells_dict.get("triangle")
    if tris is None:
        raise ValueError(f"No triangle cells in {path}.")
    return verts, tris.astype(np.int32)


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + float(np.clip(t, 0.0, 1.0)) * (b - a)


def _extract_surface_triangles(tets: np.ndarray) -> np.ndarray:
    """Extract surface triangles from a tetrahedral mesh.

    A surface face is a tet face shared by exactly one tetrahedron.
    Each tet has four triangular faces (combinations of its 4 vertices).

    Returns:
        Surface triangles, shape (F, 3), int32.
    """
    face_combos = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    # Map sorted-vertex-key → (count, actual triangle preserving original winding)
    count: dict[tuple, int] = {}
    actual: dict[tuple, tuple] = {}
    for tet in tets:
        for i, j, k in face_combos:
            tri = (int(tet[i]), int(tet[j]), int(tet[k]))
            key = tuple(sorted(tri))
            count[key] = count.get(key, 0) + 1
            actual[key] = tri
    surface = [actual[k] for k, c in count.items() if c == 1]
    return np.array(surface, dtype=np.int32)


# ---------------------------------------------------------------------------
# Headless viewer stub — used for parameter sweep runs
# ---------------------------------------------------------------------------

class _NullViewer:
    """Minimal viewer stub for headless sweep runs (no rendering, no OpenGL)."""

    def set_model(self, m): pass
    def set_camera(self, *a, **kw): pass
    def begin_frame(self, t): pass
    def log_state(self, s): pass
    def log_contacts(self, c, s): pass
    def end_frame(self): pass


# ---------------------------------------------------------------------------
# Example class — follows the Newton example pattern
# ---------------------------------------------------------------------------

class Example:
    """Scripted grasp validation using Newton's viewer infrastructure."""

    def __init__(self, viewer, args, verbose: bool = True):
        self.viewer   = viewer
        self.fps      = 120  # matches RL task sim_dt = 1/120 s
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt   = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self._finger_stiffness = args.finger_stiffness
        self.two_way_coupling  = args.two_way_coupling
        self._verbose = verbose
        self._args = args  # stored for _log_phase_exit diagnostics
        # --rigid: replace VBD soft body with a free-floating rigid sphere at OBJECT_POS.
        # Tests whether the grasp geometry / kinematics work independently of VBD stability.
        self.rigid_mode = getattr(args, "rigid", False)

        # ------------------------------------------------------------------ #
        # Resolve assets
        # ------------------------------------------------------------------ #
        robot_usd = getattr(args, "robot_usd", None)
        if not robot_usd:
            asset_path = newton.utils.download_asset("wonik_allegro")
            robot_usd  = str(asset_path / "usd" / "allegro_left_hand_with_cube.usda")
        if not os.path.isfile(robot_usd):
            raise FileNotFoundError(f"Robot USD not found: {robot_usd}")

        # --dense overrides tet mesh to the denser LR variant (~1386 particles, ~10mm spacing).
        if getattr(args, "dense", False):
            tet_mesh_path = DEFAULT_DENSE_TET_MESH
            # Adjust particle radius if still at the sparse-mesh default.
            if args.particle_radius == 0.015:
                args.particle_radius = 0.010
                if verbose:
                    print(f"[validate_grasp] --dense: particle_radius auto-adjusted to 0.010 m")
        else:
            tet_mesh_path = getattr(args, "tet_mesh", None) or DEFAULT_TET_MESH
        if not os.path.isabs(tet_mesh_path) and not os.path.isfile(tet_mesh_path):
            tet_mesh_path = os.path.join(_ASSETS_ROOT, os.path.basename(tet_mesh_path))
        if not os.path.isfile(tet_mesh_path):
            raise FileNotFoundError(
                f"Tet mesh not found: {tet_mesh_path}\n"
                "Set --tet-mesh or ASSETS_ROOT, or run tools/mesh_to_tet.py first."
            )

        # Load tet mesh — needed for VBD mode and for rigid mesh collision shape.
        nodes, tets = _load_tet_mesh(tet_mesh_path)

        if verbose:
            print(f"[validate_grasp] Robot USD   : {robot_usd}")
            if self.rigid_mode:
                print(f"[validate_grasp] Object      : RIGID mesh (blueHairRagdoll16k.obj) at {OBJECT_POS.tolist()}")
            else:
                print(f"[validate_grasp] Tet mesh    : {tet_mesh_path}")
            print(f"[validate_grasp] Object pos  : {OBJECT_POS.tolist()} [m]  (gravity off during APPROACH/CLOSE/HOLD)")
            print(
                f"[validate_grasp] Wrist z     : start={WRIST_START_Z:.3f}  grasp={WRIST_GRASP_Z:.3f}  "
                f"object z={OBJECT_POS[2]:.3f}  offset={WRIST_GRASP_Z - float(OBJECT_POS[2]):+.3f} m"
            )
            print(f"[validate_grasp] Ragdoll rot : 90°X then 90°Z  →  long axis along +X, lying flat")

        # ------------------------------------------------------------------ #
        # Build Newton model
        # ------------------------------------------------------------------ #
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        # Allegro hand — placed at approach start height.
        # ".*object" excludes the DexCube free-floating body that ships inside
        # allegro_left_hand_with_cube.usda — we use the deformable object instead.
        builder.add_usd(
            robot_usd,
            xform=wp.transform(
                wp.vec3(HAND_X, HAND_Y, WRIST_START_Z),
                HAND_ROTATION,
            ),
            enable_self_collisions=False,
            ignore_paths=[".*Dummy", ".*CollisionPlane", ".*object"],
            hide_collision_shapes=True,
        )

        # Configure finger drive (all DOFs — cube joint excluded via ignore_paths).
        # joint_target_ke [N·m/rad]: position stiffness (kp).  Lower = more compliant.
        # joint_target_kd [N·m·s/rad]: velocity damping (kv).
        # In real hardware, current-limited motors cap the torque each joint can exert.
        # --finger-stiffness simulates lower-authority actuators that give way under contact.
        self.finger_dofs = builder.joint_dof_count
        for i in range(self.finger_dofs):
            builder.joint_target_ke[i]   = args.finger_stiffness
            builder.joint_target_kd[i]   = args.finger_damping
            builder.joint_target_mode[i] = int(JointTargetMode.POSITION)
            if builder.joint_armature[i] == 0.0:
                builder.joint_armature[i] = 1e-2

        # Open / close targets — fractions of the [lower, upper] joint range.
        limit_lo = np.array(builder.joint_limit_lower[: self.finger_dofs], dtype=np.float32)
        limit_hi = np.array(builder.joint_limit_upper[: self.finger_dofs], dtype=np.float32)
        span = limit_hi - limit_lo
        self.open_targets  = (limit_lo + getattr(args, "finger_open_frac",  FINGER_OPEN_FRAC)  * span).astype(np.float32)
        self.close_targets = (limit_lo + getattr(args, "finger_close_frac", FINGER_CLOSE_FRAC) * span).astype(np.float32)

        # Object — either a free-floating rigid sphere (--rigid) or VBD soft mesh.
        # No gravity: the object stays at OBJECT_POS; the hand descends to it.
        if self.rigid_mode:
            # Load the ragdoll surface mesh as a rigid (non-deformable) body.
            # Uses the 16k OBJ surface mesh — the USD has a broken instanceable reference.
            # MuJoCo handles all finger↔mesh contacts natively — no VBD needed.
            # Tests whether the hand kinematics can form a physical grasp independently
            # of VBD stability. If rigid grasping also fails → placement/geometry issue.
            if not os.path.isfile(DEFAULT_RIGID_OBJ):
                raise FileNotFoundError(
                    f"Rigid ragdoll OBJ not found: {DEFAULT_RIGID_OBJ}\n"
                    "Expected blueHairRagdoll16k.obj in $ASSETS_ROOT."
                )
            rigid_verts, rigid_tris = _load_surface_mesh(DEFAULT_RIGID_OBJ)
            rigid_mesh = newton.Mesh(rigid_verts, rigid_tris.flatten())
            rigid_body_id = builder.add_body(
                xform=wp.transform(
                    wp.vec3(float(OBJECT_POS[0]), float(OBJECT_POS[1]), float(OBJECT_POS[2])),
                    wp.quat_identity(),
                ),
                label="rigid_ragdoll",
            )
            # Apply RAGDOLL_ROT as the shape's local transform — identical orientation
            # to the VBD soft body (long axis along +X, lying flat).
            import copy as _copy
            _rigid_cfg = _copy.copy(builder.default_shape_cfg)
            _rigid_cfg.mu = args.soft_contact_mu  # same friction param as VBD mode
            _rigid_cfg.density = args.density      # controls rigid body mass
            builder.add_shape_mesh(
                rigid_body_id,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), RAGDOLL_ROT),
                mesh=rigid_mesh,
                cfg=_rigid_cfg,
            )
            self.rigid_body_idx = rigid_body_id
            self.particle_count = 0
            self.particle_start = 0
        else:
            particle_offset = len(builder.particle_q)
            builder.add_soft_mesh(
                pos=wp.vec3(float(OBJECT_POS[0]), float(OBJECT_POS[1]), float(OBJECT_POS[2])),
                rot=RAGDOLL_ROT,  # 90° around X: upright figure laid horizontal along Y
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in nodes],
                indices=[int(i) for i in tets.flatten()],
                density=args.density,
                k_mu=args.k_mu,
                k_lambda=args.k_lambda,
                k_damp=args.k_damp,
                add_surface_mesh_edges=True,
                particle_radius=args.particle_radius,
            )
            self.particle_count = len(builder.particle_q) - particle_offset
            self.particle_start = particle_offset

        # No ground plane — object floats in space.

        # Disable gravity before finalize so the Model carries a properly-typed
        # gravity array.  Setting model.gravity after finalize replaces the warp
        # array with a raw vec3, which breaks SolverMuJoCo's gravity.numpy()[0].
        builder.gravity = 0.0  # scalar magnitude; builder computes vec via up_vector

        # Graph-colour the tet mesh for the VBD Gauss-Seidel solve (skip in rigid mode).
        if not self.rigid_mode:
            builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = args.soft_contact_ke
        self.model.soft_contact_kd = args.soft_contact_kd
        self.model.soft_contact_mu = args.soft_contact_mu

        # ------------------------------------------------------------------ #
        # Solvers
        # ------------------------------------------------------------------ #
        # Root joint (connects world → hand base body) is the first joint.
        # joint_X_p[0] is the world-space parent transform of the hand root.
        self.root_joint_id = 0

        # MuJoCo-Warp solver — handles all rigid body dynamics.
        self.mujoco_solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=200,
            nconmax=300,
            impratio=10.0,
            cone="elliptic",
            iterations=100,
            ls_iterations=50,
            use_mujoco_contacts=True,
        )

        # VBD solver — handles only the soft-body particles.
        # integrate_with_external_rigid_solver=True is required: VBD does not
        # support revolute joints, so it cannot integrate the Allegro hand.
        # With this flag VBD reads state_1.body_q (written by MuJoCo) for contacts.
        # Skipped in --rigid mode (no particles, MuJoCo handles all contacts).
        self.vbd_solver = None if self.rigid_mode else newton.solvers.SolverVBD(
            self.model,
            iterations=args.vbd_iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=False,
            particle_max_velocity=args.particle_max_velocity,
        )

        # ------------------------------------------------------------------ #
        # State + collision
        # ------------------------------------------------------------------ #
        self.state_0  = self.model.state()
        self.state_1  = self.model.state()
        self.control  = self.model.control()

        # VBD contact buffer — None in rigid mode (no particles, no VBD contacts).
        self.soft_contacts = None if self.rigid_mode else self.model.contacts()
        # Scalar accumulator for sum of contact force magnitudes — reset each frame.
        self._force_sum_buf = wp.zeros(1, dtype=wp.float32, device=self.model.device)

        # Forward kinematics to initialise body_q from joint_q.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        # ------------------------------------------------------------------ #
        # Control buffer (CPU-side copy updated each phase)
        # ------------------------------------------------------------------ #
        self._ctrl_buf = self.control.joint_target_pos.numpy().copy().astype(np.float32)
        self._ctrl_buf[: self.finger_dofs] = self.open_targets
        wp.copy(
            self.control.joint_target_pos,
            wp.array(self._ctrl_buf, dtype=wp.float32, device=self.model.device),
        )

        # ------------------------------------------------------------------ #
        # Phase state machine
        # ------------------------------------------------------------------ #
        self.phase_idx    = 0   # index into PHASE_NAMES
        self.phase_step   = 0   # steps elapsed in the current phase
        self.total_step   = 0

        self.max_hold_contacts: int = 0
        self._prev_nan: bool = False   # NaN transition detection
        self._prev_contacts: int = 0   # contact count one frame before NaN
        self._done: bool = False        # set True when LIFT phase ends
        # ||net body_f|| — norm of the vector-sum of all body reaction forces [N].
        # Forces from opposite sides of the hand partially cancel.
        self._last_reaction_N: float = 0.0
        # Sum of individual |ke*penetration| magnitudes across all active contacts [N].
        # This is the true total force injected into the VBD solve — does not cancel.
        self._last_force_sum_N: float = 0.0

        # Set initial wrist to WRIST_START_Z with XY offset.
        self._set_wrist_pos(np.array([HAND_X, HAND_Y, WRIST_START_Z], dtype=np.float32))

        self.viewer.set_model(self.model)

        # Camera: side view centred between start height and object.
        cam_z = (WRIST_START_Z + float(OBJECT_POS[2])) * 0.5
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(
                wp.vec3(1.5, 0.0, cam_z),
                -5.0,
                180.0,
            )

        if self._verbose:
            print(
                f"[validate_grasp] Model       : {self.model.body_count} bodies, "
                f"{self.particle_count} object particles "
                f"(total {self.model.particle_count})"
                + (f"  [RIGID mesh blueHairRagdoll16k.obj]" if self.rigid_mode else "")
            )
            print(f"[validate_grasp] Phases      : {list(PHASE_STEPS.items())}")
            print(f"[validate_grasp] Gravity     : disabled during APPROACH/CLOSE/HOLD; "
                  f"enabled at SETTLE (0,0,-9.81) [m/s²]")
            print(f"[validate_grasp] Two-way     : {'enabled' if self.two_way_coupling else 'DISABLED (one-way)'}")
            print(f"[validate_grasp] Sim freq    : {self.fps} Hz  "
                  f"substeps={self.sim_substeps}  dt_sub={self.sim_dt*1000:.3f} ms  "
                  f"m/dt²≈{7e-3/self.sim_dt**2:.0f} N/m  ke/m_dt²≈{args.soft_contact_ke/(7e-3/self.sim_dt**2):.2f}")
            print(f"[validate_grasp] Finger kp   : {args.finger_stiffness} N·m/rad  "
                  f"kd={args.finger_damping} N·m·s/rad  "
                  f"(RL task: kp=9.0 kd=0.1 effort_limit=0.5 N·m)")
            print()
            # Log header.
            # |net F|  = norm of vector-sum of body_f (opposing forces cancel).
            # Σ|F_i|   = sum of individual |ke*penetration| (true VBD force budget).
            # F/c      = Σ|F_i| / contacts — mean force per active contact.
            # max_v    = max ||particle_qd|| over all object particles [m/s].
            print(
                f"[validate_grasp]  {'phase':8s} | {'step':>8s} | "
                f"{'wrist (x,y,z)':>26s} | {'state':<20s} | {'contacts':>8s} | "
                f"{'max_v':>7s} | {'|net F|':>9s} | {'Σ|F_i|':>9s} | {'F/c':>8s} | "
                f"{'expected':<40s} | {'obj_com (x,y,z)':>24s}"
            )
            print("[validate_grasp] " + "-" * 200)

        # One-time diagnostics: asset spacing, coverage, placement, material.
        if not self.rigid_mode and self._verbose:
            self._log_init_diagnostics(args, nodes)

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _set_wrist_pos(self, pos: np.ndarray) -> None:
        """Update the hand root joint parent transform to move the wrist."""
        xform = wp.transform(
            wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
            HAND_ROTATION,
        )
        wp.launch(
            _set_transform,
            dim=1,
            inputs=[self.model.joint_X_p, self.root_joint_id, xform],
        )
        self.mujoco_solver.notify_model_changed(SolverNotifyFlags.JOINT_PROPERTIES)

    def _get_wrist_pos(self) -> np.ndarray:
        """Read back the wrist root position from the joint parent transform [m]."""
        xform_np = self.model.joint_X_p.numpy()
        # wp.transform layout: [tx, ty, tz, qx, qy, qz, qw]
        return xform_np[self.root_joint_id, :3].copy()

    def _set_finger_targets(self, alpha: float) -> None:
        """Drive finger targets: alpha=0 → open (30%), alpha=1 → close (80%)."""
        targets = _lerp(self.open_targets, self.close_targets, alpha)
        self._ctrl_buf[: self.finger_dofs] = targets
        wp.copy(
            self.control.joint_target_pos,
            wp.array(self._ctrl_buf, dtype=wp.float32, device=self.model.device),
        )

    def _particle_com(self) -> np.ndarray:
        """Return object CoM [m].

        In VBD mode: mean position of all object particles.
        In rigid mode: position of the rigid sphere body from state_0.body_q.
        """
        if self.rigid_mode:
            try:
                bq = self.state_0.body_q.numpy()
                return bq[self.rigid_body_idx, :3].copy()
            except Exception:
                return OBJECT_POS.copy()
        pq = self.state_0.particle_q
        if pq is None or pq.ptr is None:
            return OBJECT_POS.copy()
        pq_np = pq.numpy()
        return pq_np[self.particle_start : self.particle_start + self.particle_count].mean(axis=0)

    def _set_gravity(self, gx: float, gy: float, gz: float) -> None:
        """Set simulation gravity [m/s²] by mutating model.gravity in-place.

        model.gravity is a wp.array(dtype=wp.vec3) after finalize — it must be
        updated via copy rather than attribute assignment, which would replace the
        array with a raw vec3 and break SolverMuJoCo's gravity.numpy()[0] call.
        """
        wp.copy(
            self.model.gravity,
            wp.array([wp.vec3(gx, gy, gz)], dtype=wp.vec3, device=self.model.device),
        )

    def _soft_contact_count(self) -> int:
        """Return active contact count.

        In VBD mode: Newton soft-contact buffer count.
        In rigid mode: MuJoCo ``mj_data.ncon`` (finger↔ragdoll contacts).
        """
        if self.rigid_mode:
            try:
                return int(self.mujoco_solver.mj_data.ncon)
            except Exception:
                return 0
        cnt = self.soft_contacts.soft_contact_count
        return 0 if cnt is None else int(wp.to_torch(cnt).item())

    def _particle_max_speed(self) -> float:
        """Return max ||particle_qd|| [m/s] over object particles (0 in rigid mode)."""
        if self.rigid_mode or self.particle_count == 0:
            return 0.0
        pqd = self.state_0.particle_qd
        if pqd is None or pqd.ptr is None:
            return 0.0
        pqd_np = pqd.numpy()
        obj_vel = pqd_np[self.particle_start : self.particle_start + self.particle_count]
        if len(obj_vel) == 0:
            return 0.0
        return float(np.linalg.norm(obj_vel, axis=1).max())

    def _contacts_by_body(self) -> dict[int, int]:
        """Return active VBD contact count per rigid body index."""
        n = self._soft_contact_count()
        if n == 0:
            return {}
        shape_arr = self.soft_contacts.soft_contact_shape.numpy()[:n]
        sb = self.model.shape_body.numpy()
        result: dict[int, int] = {}
        for s in shape_arr:
            b = int(sb[int(s)])
            if b >= 0:
                result[b] = result.get(b, 0) + 1
        return result

    def _log_init_diagnostics(self, args, nodes: np.ndarray) -> None:
        """One-time diagnostics printed at init: asset spacing, coverage, placement, material."""
        # --- Asset (mesh-space stats) ---
        bb_lo = nodes.min(axis=0)
        bb_hi = nodes.max(axis=0)
        bb_size = bb_hi - bb_lo
        # Nearest-neighbour spacing via subsampled brute-force (no scipy dependency).
        n_sample = min(300, len(nodes))
        pts = nodes[np.random.default_rng(42).choice(len(nodes), n_sample, replace=False)]
        diff = pts[:, None, :] - pts[None, :, :]      # (N, N, 3)
        dist = np.sqrt((diff ** 2).sum(-1))            # (N, N)
        np.fill_diagonal(dist, np.inf)
        nn = dist.min(axis=1)
        spacing_mean = float(nn.mean())
        spacing_min  = float(nn.min())
        spacing_max  = float(nn.max())
        half_spacing = spacing_mean / 2.0
        coverage = args.particle_radius / half_spacing if half_spacing > 0 else 0.0

        print("[diag] ===== INIT DIAGNOSTICS =====")
        print("[diag] --- Asset ---")
        print(f"[diag]   Particles (vertices)  : {self.particle_count}")
        print(f"[diag]   Mesh bbox (mesh-space) : {bb_lo.tolist()} → {bb_hi.tolist()}")
        print(f"[diag]   Mesh size [m]          : {bb_size.tolist()}")
        print(f"[diag]   NN spacing [m]         : mean={spacing_mean:.4f}  min={spacing_min:.4f}  max={spacing_max:.4f}")
        print(f"[diag]   particle_radius        : {args.particle_radius:.4f} m")
        cov_note = "OK — spheres overlap, surface sealed" if coverage >= 1.0 else "GAP — fingers may pass through!"
        print(f"[diag]   Coverage ratio r/(s/2) : {coverage:.2f}  [{cov_note}]")
        print(f"[diag]   (coverage < 1 means gaps between particle spheres larger than the spheres themselves)")

        # --- Placement ---
        print("[diag] --- Placement ---")
        print(f"[diag]   OBJECT_POS [m]        : {OBJECT_POS.tolist()}")
        print(f"[diag]   WRIST_GRASP_Z [m]     : {WRIST_GRASP_Z:.3f}  WRIST_START_Z={WRIST_START_Z:.3f}")
        print(f"[diag]   Wrist-to-object z     : {WRIST_GRASP_Z - float(OBJECT_POS[2]):+.3f} m "
              f"(positive = wrist above object CoM)")
        # Rough palm/fingertip estimate — depends on hand geometry, but gives order of magnitude.
        # In the Allegro wonik_allegro USD, wrist root → palm centre is approx 10-12cm in the
        # palm-normal direction; fingertip extends another 10-12cm.
        APPROX_WRIST_TO_PALM = 0.10   # m (palm centre below wrist root after HAND_ROTATION)
        APPROX_PALM_TO_TIP   = 0.10   # m (fingertip below palm)
        approx_palm_z = WRIST_GRASP_Z - APPROX_WRIST_TO_PALM
        approx_tip_z  = approx_palm_z - APPROX_PALM_TO_TIP
        print(f"[diag]   ≈ palm z (est)         : {approx_palm_z:.3f}")
        print(f"[diag]   ≈ fingertip z (est)    : {approx_tip_z:.3f}  "
              f"vs object top z≈{float(OBJECT_POS[2]) + float(bb_size.max()) * 0.5:.3f}")
        tip_gap = approx_tip_z - float(OBJECT_POS[2])
        print(f"[diag]   ≈ fingertip-object gap : {tip_gap:+.3f} m "
              f"({'fingertips reach below object CoM ✓' if tip_gap < 0 else 'fingertips above object CoM — may miss!'})")

        # --- Material & solver ---
        print("[diag] --- Material & Solver ---")
        print(f"[diag]   k_mu={args.k_mu:.1e} Pa  k_lambda={args.k_lambda:.1e} Pa  k_damp={args.k_damp:.1e}")
        print(f"[diag]   soft_contact_ke={args.soft_contact_ke:.1e}  kd={args.soft_contact_kd:.1e}  "
              f"mu={args.soft_contact_mu:.2f}")
        eff_kd = args.soft_contact_kd * args.soft_contact_ke / self.sim_dt
        eff_md = args.k_damp * args.k_mu / self.sim_dt
        print(f"[diag]   Effective contact damping stiffness  kd·ke/dt  : {eff_kd:.2e} N/m "
              f"({'stable < 1e6' if eff_kd < 1e6 else 'DANGER — may cause VBD divergence!'})")
        print(f"[diag]   Effective material damping stiffness kd·mu/dt  : {eff_md:.2e} N/m")
        print(f"[diag]   density={args.density:.0f} kg/m³  max_particle_velocity={args.particle_max_velocity:.1f} m/s")
        print(f"[diag]   substeps={self.sim_substeps}  vbd_iters={args.vbd_iterations}  "
              f"dt_sub={self.sim_dt*1000:.3f} ms")
        print(f"[diag]   finger kp={args.finger_stiffness:.1f} N·m/rad  kd={args.finger_damping:.2f}")

        # --- Body layout ---
        print("[diag] --- Bodies ---")
        print(f"[diag]   bodies={self.model.body_count}  shapes={self.model.shape_count}  "
              f"joints={self.model.joint_count}  finger_dofs={self.finger_dofs}")
        try:
            bn = list(self.model.body_name)
            tips = [(i, n) for i, n in enumerate(bn)
                    if any(k in n.lower() for k in ("tip", "fingertip", "distal", "link_3"))]
            if tips:
                print(f"[diag]   Likely fingertip bodies : {tips}")
            else:
                print(f"[diag]   Bodies (first 20) : {list(enumerate(bn))[:20]}")
        except (AttributeError, TypeError):
            print("[diag]   Body names: not available in model")
        print("[diag] ===================================")
        print()

    def _log_phase_exit(self, phase_name: str) -> None:
        """Diagnostics printed once when leaving a phase."""
        contacts = self._soft_contact_count()
        com = self._particle_com()
        max_v = self._particle_max_speed()

        print(f"\n[diag] --- End of {phase_name} ---")
        print(f"[diag]   contacts={contacts}  obj_com={[f'{v:.4f}' for v in com.tolist()]}  "
              f"max_v={max_v:.3f} m/s")

        if phase_name == "APPROACH":
            # Closest rigid body to object CoM → tells us how close the hand got.
            try:
                bq = self.state_0.body_q.numpy()           # (B, 7): [tx,ty,tz,…]
                body_pos = bq[:self.model.body_count, :3]  # (B, 3)
                dists = np.linalg.norm(body_pos - com[None, :], axis=1)
                closest8 = np.argsort(dists)[:8]
                try:
                    bn = list(self.model.body_name)
                    near = [(i, f"{dists[i]:.3f}m", bn[i]) for i in closest8]
                except (AttributeError, TypeError):
                    near = [(i, f"{dists[i]:.3f}m") for i in closest8]
                print(f"[diag]   8 bodies nearest to obj CoM: {near}")
            except Exception as exc:
                print(f"[diag]   (body read failed: {exc})")
            if contacts == 0:
                print(f"[diag]   WARN: 0 contacts — try lowering WRIST_GRASP_Z "
                      f"(current {WRIST_GRASP_Z:.3f}) or increasing --particle-radius")

        elif phase_name == "CLOSE":
            by_body = self._contacts_by_body()
            if by_body:
                print(f"[diag]   Contacts by body idx : {dict(sorted(by_body.items()))}")
                try:
                    bn = list(self.model.body_name)
                    named = {bn[b]: c for b, c in by_body.items() if b < len(bn)}
                    print(f"[diag]   Contacts by body name: {named}")
                except (AttributeError, TypeError):
                    pass
            else:
                print("[diag]   WARN: 0 contacts at end of CLOSE — fingers never touched the object!")
                print("[diag]         Likely causes (check in order):")
                print(f"[diag]           1. Placement: WRIST_GRASP_Z={WRIST_GRASP_Z:.3f} too high "
                      f"→ try {WRIST_GRASP_Z - 0.05:.3f}")
                print(f"[diag]           2. Coverage: particle_radius={self._args.particle_radius:.3f} m may be too "
                      f"small relative to inter-particle spacing → increase --particle-radius or use --dense")
                print(f"[diag]           3. Finger reach: FINGER_OPEN_FRAC={FINGER_OPEN_FRAC:.2f} "
                      f"spreads too far → try 0.2")
            # Joint tracking error at end of CLOSE.
            try:
                jq = self.state_0.joint_q
                if jq is not None and jq.ptr is not None:
                    jq_np = jq.numpy()[:self.finger_dofs]
                    err = np.abs(jq_np - self.close_targets)
                    print(f"[diag]   Joint tracking error: max={err.max():.4f}  mean={err.mean():.4f} rad")
            except Exception as exc:
                print(f"[diag]   (joint error read failed: {exc})")

        elif phase_name in ("HOLD", "SETTLE"):
            drift = float(np.linalg.norm(com - OBJECT_POS))
            print(f"[diag]   Object drift from init pos : {drift:.4f} m "
                  f"({'stable' if drift < 0.05 else 'DRIFTING — grasp is weak'})")

        elif phase_name == "LIFT":
            lift_height = float(com[2]) - float(OBJECT_POS[2])
            lifted = not np.isnan(com).any() and lift_height > 0.05
            print(f"[diag]   Object lift height : {lift_height:+.4f} m  "
                  f"[{'LIFTED ✓' if lifted else 'DROPPED / not lifted'}]")
            if not lifted:
                print("[diag]   Grip failure diagnosis:")
                print(f"[diag]     finger_close_frac={self._args.finger_close_frac:.2f}  "
                      f"finger_stiffness={self._args.finger_stiffness:.1f} N·m/rad  "
                      f"mu={self._args.soft_contact_mu:.2f}")
                print("[diag]     → try --soft-contact-mu 2.0 or 3.0 for higher friction (most likely fix)")
                print("[diag]     → try --density 200 for a 5× lighter object (reduces required friction force)")
                print("[diag]     → try --finger-close-frac 0.90 for deeper finger wrap")
                print("[diag]     → try --finger-stiffness 20 to test if geometry allows grip with stronger actuators")
            # Joint tracking error at end of LIFT — fingers being pushed open shows as large error.
            try:
                jq = self.state_0.joint_q
                if jq is not None and jq.ptr is not None:
                    jq_np = jq.numpy()[:self.finger_dofs]
                    err = np.abs(jq_np - self.close_targets)
                    print(f"[diag]   Finger joint error vs close target: max={err.max():.4f}  "
                          f"mean={err.mean():.4f} rad")
                    print(f"[diag]   (large error = fingers were pushed open by the object weight)")
            except Exception:
                pass
        print()

    def _diagnose_nan(self, contacts_before: int) -> None:
        """Dump detailed state at the frame where NaN first appears."""
        pq = self.state_0.particle_q
        pqd = self.state_0.particle_qd
        print(f"\n[validate_grasp] ===== NaN DIAGNOSIS (step {self.total_step}) =====")
        print(f"[validate_grasp]   Phase         : {PHASE_NAMES[self.phase_idx]} "
              f"step {self.phase_step}/{PHASE_STEPS[PHASE_NAMES[self.phase_idx]]}")
        print(f"[validate_grasp]   Contacts prev : {contacts_before}  (frame before NaN)")
        print(f"[validate_grasp]   Forces prev   : net={self._last_reaction_N:.1f} N  "
              f"sum_magnitudes={self._last_force_sum_N:.1f} N  "
              f"per_contact={self._last_force_sum_N/max(contacts_before,1):.3f} N/c")
        print(f"[validate_grasp]   soft_contact_ke={self.model.soft_contact_ke}  "
              f"kd={self.model.soft_contact_kd}  "
              f"mu={self.model.soft_contact_mu}")
        if pq is not None and pq.ptr is not None:
            pq_np  = pq.numpy()
            pqd_np = pqd.numpy() if (pqd is not None and pqd.ptr is not None) else None
            nan_mask = np.isnan(pq_np).any(axis=1)
            n_nan = int(nan_mask.sum())
            print(f"[validate_grasp]   NaN particles : {n_nan} / {len(pq_np)}")
            if n_nan > 0:
                # Report bounding box of the NaN particles in the PREVIOUS frame
                # (values are NaN now, so show the non-NaN subset).
                ok_pos = pq_np[~nan_mask]
                if len(ok_pos):
                    print(f"[validate_grasp]   Non-NaN bbox  : "
                          f"x=[{ok_pos[:,0].min():.4f},{ok_pos[:,0].max():.4f}]  "
                          f"y=[{ok_pos[:,1].min():.4f},{ok_pos[:,1].max():.4f}]  "
                          f"z=[{ok_pos[:,2].min():.4f},{ok_pos[:,2].max():.4f}]")
            if pqd_np is not None:
                speed = np.linalg.norm(pqd_np[~np.isnan(pqd_np).any(axis=1)], axis=1)
                if len(speed):
                    print(f"[validate_grasp]   Particle speed: max={speed.max():.4f}  "
                          f"mean={speed.mean():.4f}  p99={np.percentile(speed, 99):.4f} [m/s]")
        fpc = self._last_force_sum_N / max(contacts_before, 1)
        print(f"[validate_grasp]   Hint: VBD diverges from tet element inversion under simultaneous")
        print(f"[validate_grasp]         loading at {contacts_before} contacts "
              f"(Σ|F_i|={self._last_force_sum_N:.0f} N, {fpc:.2f} N/c).")
        print(f"[validate_grasp]         Root cause: too many particles displaced simultaneously")
        print(f"[validate_grasp]         before GS iterations converge → some tets invert → NaN.")
        print(f"[validate_grasp]         NOTE: more substeps/iterations make it WORSE (not better):")
        print(f"[validate_grasp]           more substeps → same position correction / smaller dt")
        print(f"[validate_grasp]                         → v = Δx/dt doubles → amplified instability")
        print(f"[validate_grasp]           ke=500 (softer) → fingers sink deeper → more contacts")
        print(f"[validate_grasp]                         → more simultaneous deformation → more inversions")
        print(f"[validate_grasp]         Fix: reduce simultaneous contact count by making fingers yield")
        print(f"[validate_grasp]         → try: --finger-stiffness 2   (fingers yield at ~2 N·m; "
              f"current={self._finger_stiffness:.0f})")
        print(f"[validate_grasp]         → try: reduce FINGER_CLOSE_FRAC (currently {FINGER_CLOSE_FRAC:.2f})"
              f" so fingers don't push as far")
        print(f"[validate_grasp] ======================================================\n")

    # ---------------------------------------------------------------------- #
    # Simulation — two-phase: rigid MuJoCo then VBD
    # ---------------------------------------------------------------------- #

    def simulate(self) -> None:
        """One frame of simulation.

        In VBD mode (default): two-phase MuJoCo + VBD coupling that matches
        proxy_newton_manager._simulate_two_phase.

        In rigid mode (--rigid): MuJoCo-only — no particles, no VBD, no collide().
        MuJoCo handles finger↔sphere contact natively.
        """
        if self.rigid_mode:
            for _s in range(self.sim_substeps):
                self.state_0.clear_forces()
                wp.copy(self.state_1.body_q,  self.state_0.body_q)
                wp.copy(self.state_1.body_qd, self.state_0.body_qd)
                self.mujoco_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
            return

        for s in range(self.sim_substeps):
            # Zero external forces from the previous substep.
            self.state_0.clear_forces()

            # Detect particle-rigid contacts at current positions.
            # This MUST happen before the rigid step so the reaction forces are ready.
            self.model.collide(self.state_0, self.soft_contacts)

            # Two-way coupling: inject equal-and-opposite contact reactions into body_f.
            # MuJoCo.step() will read body_f and apply them to the finger joints.
            # Disabled with --no-two-way-coupling for one-way (object reacts, hand does not).
            if self.two_way_coupling:
                apply_soft_body_reactions(
                    self.soft_contacts, self.state_0, self.model,
                    particle_q_prev=self.state_1.particle_q,
                    friction_epsilon=self.vbd_solver.friction_epsilon,
                    dt=self.sim_dt,
                )

            # On the last substep, snapshot force metrics for the frame log.
            if s == self.sim_substeps - 1:
                # Net vector sum of body reaction forces (opposing forces cancel).
                if self.state_0.body_f is not None:
                    bf = self.state_0.body_f.numpy()
                    self._last_reaction_N = float(np.linalg.norm(bf[:, :3].sum(axis=0)))
                # Sum of individual |F_i| = ke * penetration_i (does NOT cancel).
                # This is the true force budget the VBD solver must absorb.
                self._force_sum_buf.zero_()
                wp.launch(
                    _kernel_sum_contact_force_magnitudes,
                    dim=VBD_MAX_CONTACTS,
                    inputs=[
                        self.soft_contacts.soft_contact_count,
                        self.soft_contacts.soft_contact_particle,
                        self.soft_contacts.soft_contact_shape,
                        self.soft_contacts.soft_contact_body_pos,
                        self.soft_contacts.soft_contact_normal,
                        self.state_0.particle_q,
                        self.model.particle_radius,
                        self.state_0.body_q,
                        self.model.shape_body,
                        float(self.model.soft_contact_ke),
                        self._force_sum_buf,
                    ],
                )
                self._last_force_sum_N = float(self._force_sum_buf.numpy()[0])

            # Pre-seed state_1 body transforms from state_0.
            # MuJoCo only writes dynamic bodies; static bodies keep the value we set.
            wp.copy(self.state_1.body_q,  self.state_0.body_q)
            wp.copy(self.state_1.body_qd, self.state_0.body_qd)

            # Rigid step: reads state_0.body_f for soft-contact reactions → state_1.
            self.mujoco_solver.step(
                self.state_0, self.state_1, self.control, None, self.sim_dt
            )

            # VBD step: uses the same contacts detected in step 2 (same-substep coupling).
            # Reads state_1.body_q (updated rigid positions) for contact geometry.
            self.vbd_solver.step(
                self.state_0, self.state_1, self.control, self.soft_contacts, self.sim_dt
            )

            self.state_0, self.state_1 = self.state_1, self.state_0

    # ---------------------------------------------------------------------- #
    # Newton example interface
    # ---------------------------------------------------------------------- #

    def step(self) -> None:
        if self._done:
            return

        phase_name  = PHASE_NAMES[self.phase_idx]
        phase_total = PHASE_STEPS[phase_name]
        alpha       = self.phase_step / max(phase_total - 1, 1)  # 0.0 → 1.0

        wrist_start = np.array([HAND_X, HAND_Y, WRIST_START_Z], dtype=np.float32)
        wrist_grasp = np.array([HAND_X, HAND_Y, WRIST_GRASP_Z], dtype=np.float32)

        # ---- Phase actions ------------------------------------------------
        if phase_name == "APPROACH":
            # Wrist descends from start to grasp height; fingers stay open.
            self._set_wrist_pos(_lerp(wrist_start, wrist_grasp, alpha))

        elif phase_name == "CLOSE":
            # Wrist holds at grasp height; fingers drive closed.
            self._set_wrist_pos(wrist_grasp)
            self._set_finger_targets(alpha)

        elif phase_name == "HOLD":
            # Everything held; monitor contact stability.
            self._set_wrist_pos(wrist_grasp)
            self._set_finger_targets(1.0)

        elif phase_name == "SETTLE":
            # Enable gravity on the first SETTLE frame, then hold the grasp steady
            # for 30 frames so the object re-balances under load before LIFT begins.
            if self.phase_step == 0:
                self._set_gravity(0.0, 0.0, -9.81)
            self._set_wrist_pos(wrist_grasp)
            self._set_finger_targets(1.0)

        elif phase_name == "LIFT":
            # Wrist rises to start height; fingers stay closed.
            # Gravity is still on from SETTLE — object must be carried against it.
            self._set_wrist_pos(_lerp(wrist_grasp, wrist_start, alpha))
            self._set_finger_targets(1.0)

        # ---- Simulate -----------------------------------------------------
        self.simulate()
        self.sim_time += self.frame_dt

        # ---- Track hold contacts ------------------------------------------
        if phase_name == "HOLD":
            self.max_hold_contacts = max(self.max_hold_contacts, self._soft_contact_count())

        # ---- Advance phase ------------------------------------------------
        self.phase_step += 1
        self.total_step += 1
        if self.phase_step >= phase_total:
            if self._verbose:
                self._log_phase_exit(phase_name)
            if self.phase_idx < len(PHASE_NAMES) - 1:
                self.phase_step = 0
                self.phase_idx += 1
            else:
                # LIFT phase complete — stop simulating (prevents step counter overflow
                # and VBD instability from continued high-contact simulation).
                self._done = True

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if not self.rigid_mode:
            self.viewer.log_contacts(self.soft_contacts, self.state_0)
        self.viewer.end_frame()

        # Per-frame diagnostic log — every frame so the full animation timeline
        # is visible in stdout.  Redirect to a file for offline analysis.
        # Suppressed when not verbose (e.g. during parameter sweep).
        if not self._verbose:
            return

        phase_name  = PHASE_NAMES[self.phase_idx]
        phase_total = PHASE_STEPS[phase_name]
        contacts    = self._soft_contact_count()
        wrist       = self._get_wrist_pos()
        obj_com     = self._particle_com()
        max_v       = self._particle_max_speed()  # max ||particle_qd|| [m/s] — NaN warning
        net_f_N     = self._last_reaction_N       # ||Σ F_i|| — vector sum (cancels)
        sum_f_N     = self._last_force_sum_N      # Σ||F_i|| — scalar sum (true budget)
        f_per_c     = sum_f_N / max(contacts, 1)  # mean |F| per contact

        # NaN transition: print detailed diagnostics on the first frame NaN appears.
        is_nan = np.isnan(obj_com).any()
        if is_nan and not self._prev_nan:
            self._diagnose_nan(contacts_before=self._prev_contacts)
        self._prev_nan = bool(is_nan)
        self._prev_contacts = contacts

        exp_lo, exp_hi, exp_desc = EXPECTED_CONTACTS[phase_name]
        ok = exp_lo <= contacts <= exp_hi
        ok_marker = "  " if ok else "!!"

        # Derive a human-readable state label from the phase and alpha.
        alpha = self.phase_step / max(phase_total - 1, 1)
        if phase_name == "APPROACH":
            state_label = f"approaching ({alpha*100:.0f}%)"
        elif phase_name == "CLOSE":
            state_label = f"closing ({alpha*100:.0f}%)"
        elif phase_name == "LIFT":
            state_label = f"lifting ({alpha*100:.0f}%)"
        else:
            state_label = "holding"

        print(
            f"[validate_grasp] {ok_marker}"
            f" {phase_name:8s} | {self.phase_step:4d}/{phase_total:<4d} | "
            f"wrist=({wrist[0]:+.3f},{wrist[1]:+.3f},{wrist[2]:+.3f}) | "
            f"{state_label:<20s} | {contacts:8d} | "
            f"{max_v:>5.2f}m/s | "
            f"{net_f_N:>6.1f} N | {sum_f_N:>6.1f} N | {f_per_c:>5.2f} N/c | "
            f"{exp_desc:<40s} | "
            f"obj=({obj_com[0]:+.3f},{obj_com[1]:+.3f},{obj_com[2]:+.3f})",
            flush=True,
        )

    def test_final(self) -> None:
        """Called by newton.examples.run() when --test is passed."""
        success = self.max_hold_contacts >= HOLD_CONTACT_THRESHOLD
        print()
        print("[validate_grasp] === RESULTS ===")
        print(f"[validate_grasp] Max contacts during HOLD  : {self.max_hold_contacts}")
        print(f"[validate_grasp] Hold contact threshold    : {HOLD_CONTACT_THRESHOLD}")
        print(f"[validate_grasp] Grasp SUCCESS             : {success}")
        if not success:
            raise ValueError(
                f"Grasp validation FAILED: max contacts during HOLD = {self.max_hold_contacts} "
                f"(need >= {HOLD_CONTACT_THRESHOLD}).\n"
                "Tuning hints:\n"
                "  --particle-radius   increase if fingers pass through the object\n"
                "  --soft-contact-ke   increase for harder contact surface\n"
                "  --k-mu              increase for stiffer material\n"
                "  --vbd-iterations    increase (20-30) if particles explode\n"
                "  OBJECT_POS          adjust z to match palm height\n"
                "  PALM_OFFSET         adjust if WRIST_GRASP_Z needs tuning"
            )


# ---------------------------------------------------------------------------
# Headless sweep infrastructure
# ---------------------------------------------------------------------------

def _run_headless(args) -> dict[str, float | bool]:
    """Run a complete grasp episode headless and return scalar metrics.

    Returns:
        hold_contacts: max contacts during HOLD phase (>= HOLD_CONTACT_THRESHOLD = success).
        lift_z:        object CoM z at LIFT end minus OBJECT_POS[2] [m].
                       Positive = object lifted against gravity.
        final_contacts: contacts at last LIFT frame (non-zero = still holding).
        nan:           True if VBD diverged (particles → NaN) at any point.
    """
    viewer = _NullViewer()
    ex = Example(viewer, args, verbose=False)
    for _ in range(TOTAL_FRAMES):
        if ex._done:
            break
        ex.step()
    com = ex._particle_com()
    return {
        "hold_contacts":   float(ex.max_hold_contacts),
        "lift_z":          float(com[2]) - float(OBJECT_POS[2]),
        "final_contacts":  float(ex._soft_contact_count()),
        "nan":             bool(np.isnan(com).any()),
    }


def _run_sweep(base_args) -> None:
    """Grid search over key physics parameters, reporting grasp metrics.

    Grid axes (27 runs):
      k_mu            : [1e3, 1e4, 1e5]   Lamé shear stiffness
      soft_contact_ke : [5e3, 1e4, 5e4]   particle-rigid contact stiffness
      particle_radius : [0.010, 0.015, 0.020]  particle collision radius

    For each combination, one full episode (APPROACH → LIFT) is run
    headless.  The key metric is lift_z > 0.05 m (object lifted ≥ 5 cm).
    """
    grid = {
        "k_mu":            [1e3, 1e4, 1e5],
        "soft_contact_ke": [5e3, 1e4, 5e4],
        "particle_radius": [0.010, 0.015, 0.020],
    }
    total = 1
    for v in grid.values():
        total *= len(v)
    print(f"[sweep] Parameter sweep: {' × '.join(f'{k}({len(v)})' for k, v in grid.items())} = {total} runs")
    print(f"[sweep] Success criterion: not NaN  AND  hold_contacts >= {HOLD_CONTACT_THRESHOLD}  "
          f"AND  lift_z > 0.05 m")
    print()

    results = []
    run = 0
    for k_mu, ke, pr in itertools.product(grid["k_mu"], grid["soft_contact_ke"], grid["particle_radius"]):
        run += 1
        a = copy.copy(base_args)
        a.k_mu = k_mu
        a.k_lambda = k_mu   # keep lambda == mu (equal stiffness)
        a.soft_contact_ke = ke
        a.particle_radius = pr

        try:
            m = _run_headless(a)
        except Exception as exc:
            m = {"hold_contacts": 0.0, "lift_z": float("nan"), "final_contacts": 0.0, "nan": True}
            print(f"[sweep] run {run:2d}/{total} EXCEPTION: {exc}", flush=True)
            results.append((k_mu, ke, pr, m))
            continue

        success = not m["nan"] and m["hold_contacts"] >= HOLD_CONTACT_THRESHOLD and m["lift_z"] > 0.05
        tag = "SUCCESS" if success else ("NaN" if m["nan"] else "fail")
        print(
            f"[sweep] run {run:2d}/{total}  "
            f"k_mu={k_mu:.0e}  ke={ke:.0e}  pr={pr:.3f}  →  "
            f"hold_c={m['hold_contacts']:.0f}  lift={m['lift_z']:+.3f}m  [{tag}]",
            flush=True,
        )
        results.append((k_mu, ke, pr, m))

    # Summary table of successful runs.
    successes = [r for r in results if not r[3]["nan"]
                 and r[3]["hold_contacts"] >= HOLD_CONTACT_THRESHOLD
                 and r[3]["lift_z"] > 0.05]
    print()
    print(f"[sweep] ===== SWEEP RESULTS: {len(successes)}/{total} successful runs =====")
    if successes:
        successes.sort(key=lambda r: -r[3]["lift_z"])
        print("[sweep] Top runs by lift height:")
        for k_mu, ke, pr, m in successes[:10]:
            print(f"[sweep]   k_mu={k_mu:.0e}  ke={ke:.0e}  pr={pr:.3f}  "
                  f"hold_c={m['hold_contacts']:.0f}  lift={m['lift_z']:+.3f}m")
    else:
        print("[sweep] No successful runs. Suggested next steps:")
        # Find best partial result (highest lift_z without NaN).
        valid_full = [(k_mu, ke, pr, m) for k_mu, ke, pr, m in results if not m["nan"]]
        if valid_full:
            best = max(valid_full, key=lambda r: r[3]["lift_z"])
            print(f"[sweep]   Best non-NaN: k_mu={best[0]:.0e} ke={best[1]:.0e} pr={best[2]:.3f} "
                  f"lift={best[3]['lift_z']:+.3f}m  hold={best[3]['hold_contacts']:.0f}")
        print("[sweep]   All NaN → reduce k_mu, ke, or increase --vbd-iterations")
        print("[sweep]   All 0 contacts → placement issue; try --rigid to debug geometry")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    # Override the default 100-frame limit so all phases complete.
    parser.set_defaults(num_frames=TOTAL_FRAMES)

    # Scene / mesh
    parser.add_argument(
        "--tet-mesh",
        type=str,
        default=DEFAULT_TET_MESH,
        help="Path to the ragdoll tet mesh (.msh Gmsh format). "
             f"Default: $ASSETS_ROOT/blueHairRagdoll100k_tet.msh ({DEFAULT_TET_MESH}). "
             "This is the mesh used by the RL training task. Switching meshes requires "
             "retuning particle_radius (match inter-particle spacing) and "
             "vbd_max_contacts_per_env.",
    )
    parser.add_argument(
        "--robot-usd",
        type=str,
        default=None,
        help="Robot USD file. Defaults to Newton's wonik_allegro download. "
             "Pass $ISAACLAB_NUCLEUS_DIR/Robots/KukaAllegro/kuka.usd for the full arm.",
    )

    # Simulation
    parser.add_argument("--substeps",       type=int,   default=10,   help="Substeps per frame.")
    parser.add_argument("--vbd-iterations", type=int,   default=20,
                        help="VBD solver iterations per substep. Increase to 25-30 if NaN appears during HOLD/LIFT.")

    # VBD material / contact
    # IMPORTANT — these defaults are intentionally kept in sync with
    # proxy_newton_cfg.py (Dexsuite3dgProxyNewtonCfg) so that behaviour
    # observed here transfers directly to the RL training environment.
    # If you change a default here, update proxy_newton_cfg.py as well.
    #
    # Critical constraints (Newton VBD internal formulation):
    #   soft_contact_kd and k_damp are position-level stiffness multipliers,
    #   NOT velocity-level damping coefficients.  The effective stiffness is:
    #     stiffness = kd * ke / dt
    #   With ke=1e4 and dt=1/(120Hz*10substeps)=0.000833s:
    #     kd=1e-5 → stiffness = 120 N/m       ← stable
    #     kd=1e-2 → stiffness = 120 000 N/m   ← unstable
    #     kd=100  → stiffness = 1.2×10⁹ N/m   ← immediate VBD divergence
    #   Keep both at 1e-5.
    parser.add_argument("--density",          type=float, default=1e3,  help="Object density [kg/m³].")
    parser.add_argument("--k-mu",             type=float, default=1e4,  help="Lamé μ [Pa] — shear stiffness.")
    parser.add_argument("--k-lambda",         type=float, default=1e4,  help="Lamé λ [Pa] — bulk stiffness.")
    parser.add_argument("--k-damp",           type=float, default=1e-5,
                        help="VBD material damping. MUST stay near zero — see comment above.")
    parser.add_argument("--particle-radius",  type=float, default=0.015,
                        help="Particle collision radius [m]. Should be ~half the inter-particle spacing. "
                             "Default 0.015 m (100k_tet training mesh, ~255 particles, ~20mm spacing). "
                             "Use 0.010 m for the denser blueHairRagdollLR.msh (1386 particles, ~10mm spacing).")
    parser.add_argument("--soft-contact-ke",  type=float, default=1e4,
                        help="Soft-contact stiffness [N/m]. 1e4 is the sweet spot for multi-finger grasps: "
                             "1e5 causes VBD NaN runaway at 700+ contacts; 5e3 (softer) is actually worse "
                             "because particles over-penetrate and then overcorrect with large velocities.")
    parser.add_argument("--soft-contact-kd",  type=float, default=1e-5,
                        help="Soft-contact damping. MUST stay near zero — see comment above.")
    parser.add_argument("--soft-contact-mu",  type=float, default=1.5,
                        help="Soft-contact friction coefficient. Default 1.5: validated with "
                             "--dense and --finger-stiffness 9; the full two-way coupling "
                             "(normal + friction reaction) requires mu >= ~1.5 for a reliable lift.")
    parser.add_argument("--finger-stiffness", type=float, default=9.0,
                        help="Joint position stiffness kp [N·m/rad] for all finger DOFs. "
                             "Default 9.0 matches the ImplicitActuatorCfg kp in kuka_allegro.py "
                             "(the actual RL training asset). Lower = more compliant fingers that "
                             "give way under contact; higher = stiffer, more likely to cause VBD NaN.")
    parser.add_argument("--finger-damping",   type=float, default=0.1,
                        help="Joint velocity damping kv [N·m·s/rad] for all finger DOFs. "
                             "Default 0.1 matches kuka_allegro.py ImplicitActuatorCfg kd.")
    parser.add_argument("--finger-close-frac", type=float, default=FINGER_CLOSE_FRAC,
                        help=f"Fraction of joint range to close to (default {FINGER_CLOSE_FRAC}). "
                             "Higher = fingers wrap tighter, more contact depth, harder to hold. "
                             "Try 0.90 or 0.95 if the grip opens during lift.")
    parser.add_argument("--finger-open-frac", type=float, default=FINGER_OPEN_FRAC,
                        help=f"Fraction of joint range for fully-open pose (default {FINGER_OPEN_FRAC}). "
                             "Lower = fingers spread wider during approach.")
    parser.add_argument("--particle-max-velocity", type=float, default=10.0,
                        help="Maximum particle speed [m/s] per substep (default: 10.0). "
                             "Displacements exceeding max_vel * dt are scaled back proportionally. "
                             "Prevents velocity runaway when a moving rigid body contacts many particles "
                             "simultaneously (e.g. during LIFT). Set to 'inf' to disable clamping.")
    parser.add_argument("--two-way-coupling", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable two-way coupling (default: on). "
                             "With --no-two-way-coupling the object reacts to the fingers but the "
                             "fingers do not feel the object — useful for ablation / debugging.")
    parser.add_argument("--stiff", action="store_true",
                        help="Override k_mu and k_lambda to 1e7 Pa for rigidity debug. "
                             "A stiff object should show large F_reaction values and no particle drift, "
                             "which confirms two-way coupling is active.")
    parser.add_argument(
        "--rigid",
        action="store_true",
        help="Replace the VBD soft body with a free-floating rigid mesh body loaded from "
             "blueHairRagdoll16k.obj (same orientation as VBD mode). "
             "Tests whether the hand kinematics can form a physical grasp independently of VBD stability. "
             "If rigid grasping also fails, the issue is placement/geometry, not physics solver.",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Use blueHairRagdollLR.msh (~1386 particles, ~10mm spacing) instead of the default sparse "
             "training mesh (~255 particles, ~20mm spacing). Tests whether particle resolution is the "
             "root cause of poor contact coverage. particle_radius is auto-adjusted to 0.010 m.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a 27-point headless parameter sweep over k_mu × soft_contact_ke × particle_radius "
             "and print a ranked results table. No viewer required. "
             "Use to find stable parameter combinations when the default settings fail.",
    )

    viewer, args = newton.examples.init(parser)
    if args.stiff:
        args.k_mu     = 1e7
        args.k_lambda = 1e7

    if getattr(args, "sweep", False):
        # Headless sweep — runs all 27 combinations and exits.
        _run_sweep(args)
    else:
        example = Example(viewer=viewer, args=args)
        newton.examples.run(example, args)
