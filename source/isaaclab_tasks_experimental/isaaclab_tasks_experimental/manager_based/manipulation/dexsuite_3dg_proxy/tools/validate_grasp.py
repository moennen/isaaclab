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
#                                      state.body_f via Newton's third law
#   c. MuJoCo.step(s0 → s1)          — reads body_f; fingers feel resistance
#   d. VBD.step(s0 → s1, contacts)   — uses same contacts; pushes particles out
#
# This is operator-splitting (IMEX) with zero time lag: action and reaction
# are computed from the same contact geometry within a single substep.
# The shared kernel and helper live in vbd_coupling.py and are imported
# by both this script and proxy_newton_manager.py.
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
OBJECT_POS = np.array([0.0, 0.0, 0.50], dtype=np.float32)

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
WRIST_GRASP_Z = 0.595
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

# Default asset paths — resolved relative to ASSETS_ROOT env var if set,
# otherwise relative to this file's location (8 levels up = repo root / assets/).
_ASSETS_ROOT = os.environ.get(
    "ASSETS_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../../../../../../assets")),
)
# blueHairRagdoll100k_tet.msh is the RL training mesh (~255 particles, 673 tets).
# blueHairRagdollLR.msh is 5.4x denser (1386 particles) — harder to stabilise.
# Always test with the training mesh so physics behaviour transfers to the task.
DEFAULT_TET_MESH = os.path.join(_ASSETS_ROOT, "blueHairRagdoll100k_tet.msh")


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


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + float(np.clip(t, 0.0, 1.0)) * (b - a)


# ---------------------------------------------------------------------------
# Example class — follows the Newton example pattern
# ---------------------------------------------------------------------------

class Example:
    """Scripted grasp validation using Newton's viewer infrastructure."""

    def __init__(self, viewer, args):
        self.viewer   = viewer
        self.fps      = 120  # matches RL task sim_dt = 1/120 s
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt   = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self._finger_stiffness = args.finger_stiffness
        self.two_way_coupling  = args.two_way_coupling

        # ------------------------------------------------------------------ #
        # Resolve assets
        # ------------------------------------------------------------------ #
        robot_usd = getattr(args, "robot_usd", None)
        if not robot_usd:
            asset_path = newton.utils.download_asset("wonik_allegro")
            robot_usd  = str(asset_path / "usd" / "allegro_left_hand_with_cube.usda")
        if not os.path.isfile(robot_usd):
            raise FileNotFoundError(f"Robot USD not found: {robot_usd}")

        tet_mesh_path = getattr(args, "tet_mesh", None) or DEFAULT_TET_MESH
        if not os.path.isabs(tet_mesh_path) and not os.path.isfile(tet_mesh_path):
            tet_mesh_path = os.path.join(_ASSETS_ROOT, os.path.basename(tet_mesh_path))
        if not os.path.isfile(tet_mesh_path):
            raise FileNotFoundError(
                f"Tet mesh not found: {tet_mesh_path}\n"
                "Set --tet-mesh or ASSETS_ROOT, or run tools/mesh_to_tet.py first."
            )

        print(f"[validate_grasp] Robot USD   : {robot_usd}")
        print(f"[validate_grasp] Tet mesh    : {tet_mesh_path}")
        print(f"[validate_grasp] Object pos  : {OBJECT_POS.tolist()} [m]  (gravity off during APPROACH/CLOSE/HOLD)")
        print(
            f"[validate_grasp] Wrist z     : start={WRIST_START_Z:.3f}  grasp={WRIST_GRASP_Z:.3f}  "
            f"object z={OBJECT_POS[2]:.3f}  offset={WRIST_GRASP_Z - float(OBJECT_POS[2]):+.3f} m"
        )
        print(f"[validate_grasp] Ragdoll rot : 90°X then 90°Z  →  long axis along +X, lying flat")

        # Load tet mesh.
        nodes, tets = _load_tet_mesh(tet_mesh_path)

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
        self.open_targets  = (limit_lo + FINGER_OPEN_FRAC  * span).astype(np.float32)
        self.close_targets = (limit_lo + FINGER_CLOSE_FRAC * span).astype(np.float32)

        # Deformable object — placed at OBJECT_POS.
        # No gravity: the object stays here; the hand descends to it.
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

        # Graph-colour the tet mesh for the VBD Gauss-Seidel solve.
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
        self.vbd_solver = newton.solvers.SolverVBD(
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

        self.soft_contacts = self.model.contacts()
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

        print(
            f"[validate_grasp] Model       : {self.model.body_count} bodies, "
            f"{self.particle_count} object particles "
            f"(total {self.model.particle_count})"
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
              f"(RL task: kp=3.0 kd=0.1 effort_limit=0.5 N·m)")
        print()
        # Log header.
        # |net F|  = norm of vector-sum of body_f (opposing forces cancel).
        # Σ|F_i|   = sum of individual |ke*penetration| (true VBD force budget).
        # F/c      = Σ|F_i| / contacts — mean force per active contact.
        print(
            f"[validate_grasp]  {'phase':8s} | {'step':>8s} | "
            f"{'wrist (x,y,z)':>26s} | {'state':<20s} | {'contacts':>8s} | "
            f"{'|net F|':>9s} | {'Σ|F_i|':>9s} | {'F/c':>8s} | "
            f"{'expected':<40s} | {'obj_com (x,y,z)':>24s}"
        )
        print("[validate_grasp] " + "-" * 190)

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
        """Return object particle CoM [m] (CPU read, fine for 1-4 worlds)."""
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
        """Return active soft-contact count."""
        cnt = self.soft_contacts.soft_contact_count
        return 0 if cnt is None else int(wp.to_torch(cnt).item())

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
        """One frame of two-phase simulation (matches proxy_newton_manager._simulate_two_phase).

        Substep order:
          1. clear_forces()               — zero state_0.body_f
          2. collide()                    — detect particle-rigid contacts at current positions
          3. apply_soft_body_reactions()  — write reaction forces into state_0.body_f
                                           (skipped when --no-two-way-coupling)
          4. mujoco_solver.step()         — rigid step; reads body_f for soft-contact reactions
          5. vbd_solver.step()            — soft step; uses same contacts as step 2
          6. swap states
        """
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
                apply_soft_body_reactions(self.soft_contacts, self.state_0, self.model)

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
        self.viewer.log_contacts(self.soft_contacts, self.state_0)
        self.viewer.end_frame()

        # Per-frame diagnostic log — every frame so the full animation timeline
        # is visible in stdout.  Redirect to a file for offline analysis.
        phase_name  = PHASE_NAMES[self.phase_idx]
        phase_total = PHASE_STEPS[phase_name]
        contacts    = self._soft_contact_count()
        wrist       = self._get_wrist_pos()
        obj_com     = self._particle_com()
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
    parser.add_argument("--soft-contact-mu",  type=float, default=0.8,
                        help="Soft-contact friction coefficient.")
    parser.add_argument("--finger-stiffness", type=float, default=3.0,
                        help="Joint position stiffness kp [N·m/rad] for all finger DOFs. "
                             "Default 3.0 matches the ImplicitActuatorCfg kp in kuka_allegro.py "
                             "(the actual RL training asset). Lower = more compliant fingers that "
                             "give way under contact; higher = stiffer, more likely to cause VBD NaN.")
    parser.add_argument("--finger-damping",   type=float, default=0.1,
                        help="Joint velocity damping kv [N·m·s/rad] for all finger DOFs. "
                             "Default 0.1 matches kuka_allegro.py ImplicitActuatorCfg kd.")
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

    viewer, args = newton.examples.init(parser)
    if args.stiff:
        args.k_mu     = 1e7
        args.k_lambda = 1e7
    example = Example(viewer=viewer, args=args)
    newton.examples.run(example, args)
