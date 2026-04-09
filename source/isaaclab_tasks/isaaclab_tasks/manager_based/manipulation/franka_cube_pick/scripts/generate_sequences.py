"""Tool 1 — Sequence Generator.

Generates N scripted simulation sequences for the Franka cube pick task.
Sequences cover all four scenario types:

  reachable   + success   (70% × 50% = 35%)
  reachable   + failure   (70% × 50% = 35%)
  unreachable + success   (30% × 50% = 15%)
  unreachable + failure   (30% × 50% = 15%)

Architecture
------------
Runs fully standalone using Newton physics simulation — no Isaac Sim / AppLauncher required.

Two Newton models are built from the Panda URDF:

  ik_model    — robot only (9 DOF). Used exclusively by the IK solver to compute
                joint-angle targets. Also used for FK when verifying EE position.

  phys_model  — robot + cube (16 DOF: 9 robot + 7 free-joint cube). Used for the
                full physics simulation. The robot arm and cube are both fully
                physics-simulated; the gripper grasps the cube through friction contacts.

For each sequence:
  1. Reset phys state: robot to home pose, cube to sampled position.
  2. Loop at 50 Hz for 10 s (500 steps, record every 2nd → 250 frames):
     a. Query waypoint state machine for EE target position + orientation + finger cmd.
     b. Solve IK on ik_model (position + orientation objectives) for joint targets.
     c. Pad robot targets to 16-element control vector; step phys_model MuJoCo solver.
     d. Read actual robot joint_q[:9] and EE body_q from phys state via eval_fk.
  3. Record per-frame: joint_pos_cmd (9), joint_pos (9), joint_vel (9),
     gripper_closed, ee_pos_w, cube_pos_w.

The output JSON can be replayed by replay_sequences.py, which runs live physics
and records the reward stream.

Usage
-----
python generate_sequences.py \\
    --num_sequences 100 \\
    --output data/validation/sequences.json \\
    --num_envs 16   # kept for CLI compatibility; physics runs sequentially
"""

import argparse
import datetime
import json
import random
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.ik as nik

sys.path.insert(0, str(Path(__file__).parent))
from _common.sampling import sample_cube_pos, sample_label
from _common.sequence_schema import DEFAULT_CONFIG, label_description, save_sequences
from _common.waypoint_ik import WaypointStateMachine, T_GRIP

_TASK_ROOT  = Path(__file__).parent.parent
_OUTPUTS_DIR = _TASK_ROOT / "data" / "validation"

# ---------------------------------------------------------------------------
# Panda URDF path (shipped with Isaac Sim pip package)
# ---------------------------------------------------------------------------

_FRANKA_URDF_CANDIDATES = [
    # Isaac Sim pip install
    Path(
        "/home/horde/micromamba/envs/env_isaaclab/lib/python3.11/site-packages/"
        "isaacsim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/"
        "franka_description/robots/panda_arm_hand.urdf"
    ),
]


def _find_panda_urdf() -> Path:
    for p in _FRANKA_URDF_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Panda URDF not found. Install Isaac Sim pip package or set PANDA_URDF env var."
    )


# ---------------------------------------------------------------------------
# Newton physics setup
# ---------------------------------------------------------------------------

_PANDA_HAND_BODY_LABEL = "panda/panda_hand"

# Joint index ranges in the 9-dof joint_coord vector
# (panda_joint1..7 = 0..6, panda_finger_joint1 = 7, panda_finger_joint2 = 8)
_ARM_JOINT_IDS    = list(range(7))
_FINGER_JOINT_IDS = [7, 8]

# PD gains — arm joints 0-6, finger joints 7-8
_ARM_KE    = [600, 600, 600, 600, 250, 150, 50]
_ARM_KD    = [50,  50,  50,  50,  30,  25,  15]
_FINGER_KE = [200, 200]  # PD joint stiffness for finger prismatic joints.
_FINGER_KD = [20,  20]

# Finger BOX contact stiffness — deliberately soft (100 N/m) so that the
# 10 mm Z-overlap at grasp only produces ~2 N normal force per finger.
# Hard contacts (10000 N/m) create 100 N+ forces; any millimetre of IK
# lateral drift during lift then generates >200 N sideways impulse that
# launches the cube.  A grip-assist body force (applied in run_one_episode)
# compensates for the reduced contact lift capacity.
_FINGER_BOX_KE = 100.0
_FINGER_BOX_KD = 5.0

# Finger open/close positions [m]
_FINGER_OPEN  = 0.04
_FINGER_CLOSE = 0.0

# Number of robot joint coordinates (arm 0-6 + 2 fingers)
_N_ROBOT_JOINTS = 9

# Cube rigid body properties
_CUBE_HALF_SIZE  = 0.025    # half-extent per axis [m] → 5 cm cube
_CUBE_DENSITY    = 1000.0  # kg/m³ → mass = 1000 × 0.05³ = 0.125 kg (125 g)
_CUBE_MASS       = _CUBE_DENSITY * (2 * 0.025) ** 3   # 0.125 kg
_CUBE_FRICTION   = 2.0      # Coulomb friction coeff for cube shape (floor contact)
_FINGER_FRICTION = 0.3      # Low friction on finger BOX: prevents lateral knock from IK drift
_CUBE_CONTACT_KE = 10000.0  # contact stiffness [N/m] for cube-floor (keep stiff for stability)

# Grip-assist body force (applied to cube after T_GRIP in reachable+success episodes).
# Models the grip forces that real contact would provide, without the lateral
# instability of stiff BOX-BOX contacts (ke=10000 + µ=2.0 → 200 N+ lateral forces).
# 3D spring-damper: F = _GRIP_K_P * (target_pos - cube_pos) + _GRIP_K_D * (ee_vel - cube_vel)
#                   F[Z] += m*g  (gravity compensation in Z only)
# Stability: k_p * dt / m must be < 2 for semi-implicit Euler. With dt=0.02s, m=0.125kg:
# k_p_max ≈ 2*m/dt = 12.5 N/m (explicit limit); Newton's implicit solver allows ~5-10×
# higher → k_p=20 gives ratio=3.2 (stable), tracks position error with τ≈0.25s.
# 3D control prevents cube from flying off laterally after the T_GRIP wrist-rotation knock.
_GRIP_K_P = 20.0    # N/m — spring to track EE in XYZ
_GRIP_K_D = 10.0    # N·s/m — damp relative XYZ velocity


def build_models(urdf_path: Path):
    """Build two Newton models: one for IK (robot only), one for physics (robot + cube).

    The IK model has 9 DOF (robot joints only) and is used solely by the IK
    solver.  The physics model has 16 DOF (9 robot + 7 cube free-joint) and
    drives the full simulation with real contact between fingers and cube.

    Returns:
        ik_model:       robot-only Newton Model used by solve_ik_single / eval_fk
        phys_model:     robot + cube Newton Model used by the physics solver
        hand_idx:       body index of panda_hand (same in both models)
        cube_body_idx:  body index of the cube in phys_model
        solver:         Newton solver for phys_model (MuJoCo preferred)
    """
    # ---- IK model (robot only, no PD gains needed) -------------------------
    mb_ik = newton.ModelBuilder()
    mb_ik.add_ground_plane()
    mb_ik.add_urdf(str(urdf_path), floating=False)
    ik_model = mb_ik.finalize()

    hand_idx = next(
        i for i, l in enumerate(mb_ik.body_label) if _PANDA_HAND_BODY_LABEL in l
    )

    # ---- Physics model (robot + cube) --------------------------------------
    mb_phys = newton.ModelBuilder()
    mb_phys.add_ground_plane()
    # Leave ground friction at default (mu=1.0) — setting it high (>2.0) causes MuJoCo
    # contact instability (cube penetrates floor). Instead, the low finger ke=20 limits
    # the asymmetric push force to keep the cube stationary during closure.
    mb_phys.add_urdf(str(urdf_path), floating=False)

    # Collision group assignment.
    # Newton/MuJoCo collision rule: shapes collide IFF they are in the SAME group.
    #   group 1 (default): ground + finger BOX shapes + cube → mutual contact
    #   group 2:           arm/hand shapes + finger MESH shapes → isolated
    #
    # IMPORTANT: Newton/mujoco_warp does NOT support MESH-BOX contact detection.
    # The URDF finger shapes (MESH type) pass silently through the cube BOX.
    # Fix: move MESH finger shapes to group 2 (non-colliding) and add explicit
    # BOX approximations of the fingers in group 1 for reliable BOX-BOX contact.
    _finger_body_set = {
        i for i, lbl in enumerate(mb_phys.body_label)
        if "leftfinger" in lbl or "rightfinger" in lbl
    }
    _left_finger_body  = next(i for i, lbl in enumerate(mb_phys.body_label) if "leftfinger"  in lbl)
    _right_finger_body = next(i for i, lbl in enumerate(mb_phys.body_label) if "rightfinger" in lbl)

    _n_urdf_shapes = len(mb_phys.shape_body)
    for _i in range(_n_urdf_shapes):
        _b = mb_phys.shape_body[_i]
        if _b < 0:
            pass  # ground: keep group 1
        else:
            mb_phys.shape_collision_group[_i] = 2   # arm, hand, AND finger MESH shapes: group 2

    # Add BOX collision shapes for each finger body (group 1).
    # panda finger.stl mesh bounds in link frame:
    #   leftfinger:  X=[-0.0104,0.0106], Y=[0.0001,0.0262], Z=[0,0.054]
    #   rightfinger: Y-negated (180° Z rotation) → Y=[-0.0262,-0.0001]
    # Inner gripping face is at Y=0 in link frame for both fingers.
    # With GRASP_QUAT (hand pointing down), link local Y → world -Y.
    # Newton/mujoco_warp does NOT detect MESH-BOX contact, so we add explicit
    # BOX approximations of the fingers in group 1 for reliable BOX-BOX friction grasping.
    _cfg_fbox = newton.ModelBuilder.ShapeConfig()
    _cfg_fbox.ke = _FINGER_BOX_KE
    _cfg_fbox.kd = _FINGER_BOX_KD
    _cfg_fbox.mu = _FINGER_FRICTION   # contact friction with cube = min(_FINGER_FRICTION, _CUBE_FRICTION)

    # Finger BOX geometry — "shelf" mode: Z overlap < Y overlap so SAT picks
    # Z as the contact normal axis, giving a direct upward mechanical force on
    # the cube instead of relying on friction alone.
    #
    # Z_c = 0.043 m, hz = 0.005 m  (BOX center 8 mm above floor at grasp EE_z≈0.109 m)
    # With GRASP_QUAT (Z_hand → −Z_world) and finger joint at EE_z − 0.058 m:
    #   BOX center world Z = EE_z − 0.101 m
    #   BOX top            = EE_z − 0.096 m  → at EE_z=0.109: 0.013 m (37 mm below cube top) ✓
    #   BOX bottom         = EE_z − 0.106 m  → at EE_z=0.109: 0.003 m (3 mm above floor) ✓
    #
    # Z overlap with cube (cube half=0.025, centered at 0.025 m):
    #   overlap_Z = hz + 0.025 − |BOX_Z − cube_Z| = 0.005 + 0.025 − 0.017 = 0.013 m (13 mm)
    # Y overlap (finger penetrates cube face by ~16 mm):
    #   overlap_Y ≈ 0.016 m (16 mm)
    #
    # 13 mm < 16 mm → SAT picks Z as contact normal → upward +Z normal force on cube.
    # Cube mechanically rests on the BOX (shelf effect). As arm lifts, cube rises with it.
    # Z overlap is maintained as long as cube tracks arm (EE−cube < 0.131 m).
    mb_phys.add_shape_box(
        body=_left_finger_body,
        xform=wp.transform(wp.vec3(0.0, 0.013, 0.043), wp.quat_identity()),
        hx=0.0105, hy=0.013, hz=0.005,
        cfg=_cfg_fbox,
        label="leftfinger_box",
    )
    mb_phys.shape_collision_group[len(mb_phys.shape_body) - 1] = 1

    # Right finger: same geometry, Y-negated
    mb_phys.add_shape_box(
        body=_right_finger_body,
        xform=wp.transform(wp.vec3(0.0, -0.013, 0.043), wp.quat_identity()),
        hx=0.0105, hy=0.013, hz=0.005,
        cfg=_cfg_fbox,
        label="rightfinger_box",
    )
    mb_phys.shape_collision_group[len(mb_phys.shape_body) - 1] = 1

    # PD gains for robot joints (indexed assignment — safe against reference aliasing)
    for i, (ke, kd) in enumerate(zip(_ARM_KE + _FINGER_KE, _ARM_KD + _FINGER_KD)):
        mb_phys.joint_target_ke[i] = float(ke)
        mb_phys.joint_target_kd[i] = float(kd)

    # Add cube as a free-floating rigid body.
    # add_body() implicitly adds a free joint (7 DOF: tx,ty,tz,qx,qy,qz,qw).
    # Do NOT call add_joint_free separately — that would add a second free joint.
    # joint_q layout after finalize: [robot_0..8, cube_tx, cube_ty, cube_tz, cube_qx, cube_qy, cube_qz, cube_qw]
    cube_body_idx = mb_phys.add_body(mass=0.0, label="cube")   # mass from ShapeConfig.density
    cfg_cube = newton.ModelBuilder.ShapeConfig()
    cfg_cube.density = _CUBE_DENSITY
    cfg_cube.mu      = _CUBE_FRICTION
    cfg_cube.ke      = _CUBE_CONTACT_KE
    mb_phys.add_shape_box(
        body=cube_body_idx,
        hx=_CUBE_HALF_SIZE, hy=_CUBE_HALF_SIZE, hz=_CUBE_HALF_SIZE,
        cfg=cfg_cube,
        label="cube_shape",
    )
    # Cube in group 1 (same as fingers and ground) → cube-finger and cube-ground contact active.
    # Arm/hand (group 2) is different from cube (group 1) → arm won't collide with cube.
    # (No explicit assignment needed since group 1 is the default; written for clarity.)
    mb_phys.shape_collision_group[len(mb_phys.shape_body) - 1] = 1

    phys_model = mb_phys.finalize()

    # Prefer MuJoCo solver; fall back to Featherstone.
    # njmax: BOX-BOX contacts (finger×cube) generate many constraint equations
    # (up to 5 per contact point × 8 contact points per pair × 2 fingers = 80).
    # Set njmax=256 to prevent nefc overflow warnings and dropped constraints.
    try:
        solver = newton.solvers.SolverMuJoCo(phys_model, njmax=256)
    except Exception:
        solver = newton.solvers.SolverFeatherstone(phys_model)

    return ik_model, phys_model, hand_idx, cube_body_idx, solver


def solve_ik_single(
    model,
    hand_idx: int,
    target_pos: list[float],
    joint_q_warm: np.ndarray,
    target_quat_wxyz: list[float] | None = None,
) -> np.ndarray:
    """Solve IK for a single EE target, warm-started from current state.

    When an orientation constraint is given, uses a two-stage solve to avoid
    converging to the wrong local minimum:
      Stage 1 (60 iters, ANALYTIC, position-only): places the arm in the
        correct XY workspace basin (near the target position).
      Stage 2 (100 iters, AUTODIFF, position+orientation): refines from the
        stage-1 solution so both position and orientation are satisfied.

    Single-stage AUTODIFF warm-started from home pose converges to an
    orientation-satisfying local minimum at x≈0.77 instead of the target at
    x≈0.59, because the orientation gradient dominates early in the search.
    The two-stage approach prevents this by anchoring the search near the
    correct XY position before adding the orientation objective.

    Args:
        model: Newton Model built from Panda URDF.
        hand_idx: body index of panda_hand.
        target_pos: [x, y, z] target position in world frame.
        joint_q_warm: (joint_coord_count,) warm-start joint angles.
        target_quat_wxyz: [w, x, y, z] target orientation in world frame.
            If None, no orientation constraint is added.

    Returns:
        joint_q_out: (joint_coord_count,) solved joint angles.
    """
    targets_wp = wp.array([wp.vec3(*target_pos)], dtype=wp.vec3)

    pos_obj = nik.IKObjectivePosition(
        link_index=hand_idx,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=targets_wp,
    )
    limit_obj = nik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower,
        joint_limit_upper=model.joint_limit_upper,
        weight=10.0,
    )

    joint_q_wp = wp.from_numpy(
        joint_q_warm[np.newaxis, :].astype(np.float32), dtype=wp.float32
    )

    if target_quat_wxyz is None:
        # Position-only: ANALYTIC mode, single stage
        ik_solver = nik.IKSolver(
            model=model, n_problems=1,
            objectives=[pos_obj, limit_obj],
            jacobian_mode=nik.IKJacobianType.ANALYTIC,
        )
        ik_solver.step(joint_q_wp, joint_q_wp, iterations=80)
    else:
        # Stage 1: position-only (ANALYTIC) — place arm in correct workspace basin
        ik_stage1 = nik.IKSolver(
            model=model, n_problems=1,
            objectives=[pos_obj, limit_obj],
            jacobian_mode=nik.IKJacobianType.ANALYTIC,
        )
        ik_stage1.step(joint_q_wp, joint_q_wp, iterations=60)

        # Stage 2: add orientation from position-warm-started solution (AUTODIFF)
        # waypoint_ik uses [w, x, y, z]; warp wp.vec4 stores quaternions as [x, y, z, w]
        w, x, y, z = target_quat_wxyz
        rot_targets = wp.array([wp.vec4(x, y, z, w)], dtype=wp.vec4)
        rot_obj = nik.IKObjectiveRotation(
            link_index=hand_idx,
            link_offset_rotation=wp.quat(0.0, 0.0, 0.0, 1.0),   # identity [x,y,z,w]
            target_rotations=rot_targets,
            weight=0.3,   # soft orientation nudge; position still dominant
        )
        ik_stage2 = nik.IKSolver(
            model=model, n_problems=1,
            objectives=[pos_obj, limit_obj, rot_obj],
            jacobian_mode=nik.IKJacobianType.AUTODIFF,
        )
        ik_stage2.step(joint_q_wp, joint_q_wp, iterations=100)

    return joint_q_wp.numpy()[0]


def fk_ee_pos(state, hand_idx: int) -> list[float]:
    """Read EE position from body_q (eval_fk must have been called first)."""
    return state.body_q.numpy()[hand_idx][:3].tolist()


# Finger joint offset in hand frame: fingers originate 0.0584 m ahead in hand Z
_FINGER_Z_OFFSET = 0.0584   # metres from hand centre to finger-joint origin in hand Z


def _finger_centering_shift(
    ik_model,
    hand_idx: int,
    ik_state,
    joint_q: np.ndarray,
    cube_pos: np.ndarray,
) -> np.ndarray:
    """Compute a world-frame shift that centres the cube between the fingers.

    The Panda fingers (panda_finger_joint1/2) slide along the hand Y-axis.
    For a successful grasp the cube centre must lie within ±(finger_max −
    cube_half_size) = ±15 mm of the finger-joint origin in the hand Y direction.
    When the IK solution places the cube off-centre in that axis the shift
    returned here corrects the EE target so that a re-solve will centre it.

    Args:
        ik_model:  Newton Model used for FK (robot-only, 9 DOF).
        hand_idx:  body index of panda_hand in ik_model.
        ik_state:  pre-allocated Newton State for ik_model (reused each call).
        joint_q:   (joint_coord_count,) IK solution to evaluate via FK.
        cube_pos:  (3,) initial cube position in world frame.

    Returns:
        shift: (3,) world-frame vector to add to the current EE target.
               Zero if the cube is already within the safe grasp band.
    """
    # Evaluate FK on the current IK solution
    ik_q = np.zeros(ik_model.joint_coord_count, dtype=np.float32)
    ik_q[:_N_ROBOT_JOINTS] = joint_q[:_N_ROBOT_JOINTS]
    ik_state.joint_q.assign(ik_q)
    ik_state.joint_qd.assign(np.zeros(ik_model.joint_dof_count, dtype=np.float32))
    newton.eval_fk(ik_model, ik_state.joint_q, ik_state.joint_qd, ik_state)
    body_q = ik_state.body_q.numpy()

    hand_pos  = body_q[hand_idx][:3]
    qx, qy, qz, qw = body_q[hand_idx][3:7]   # warp convention: [x, y, z, w]

    # Hand Y and Z axes from rotation matrix
    # Rotation matrix columns from quaternion [x,y,z,w] (Newton/Warp convention):
    #   R = [1-2(y²+z²),  2(xy-wz),   2(xz+wy)]
    #       [2(xy+wz),    1-2(x²+z²), 2(yz-wx)]
    #       [2(xz-wy),    2(yz+wx),   1-2(x²+y²)]
    # Column 1 (Y-axis image): [2(xy-wz),  1-2(x²+z²),  2(yz+wx)]
    # Column 2 (Z-axis image): [2(xz+wy),  2(yz-wx),    1-2(x²+y²)]
    hand_y = np.array([
        2.0 * (qx * qy - qw * qz),
        1.0 - 2.0 * (qx * qx + qz * qz),
        2.0 * (qy * qz + qw * qx),
    ])
    hand_z = np.array([
        2.0 * (qx * qz + qw * qy),
        2.0 * (qy * qz - qw * qx),
        1.0 - 2.0 * (qx * qx + qy * qy),
    ])

    # Finger-joint origin in world frame (0.0584 m along hand Z from hand centre)
    finger_origin = hand_pos + _FINGER_Z_OFFSET * hand_z

    # Cube centre projected onto hand Y from the finger-joint origin
    cube_in_hand_y = float(np.dot(cube_pos - finger_origin, hand_y))

    # Always correct: any offset in hand-Y will cause asymmetric finger contact.
    # The correction is applied whenever the offset exceeds 5 mm (dead-band to
    # avoid noise-driven re-solves when already well-centred).
    safe_band = 0.005  # 5 mm dead-band
    if abs(cube_in_hand_y) <= safe_band:
        return np.zeros(3, dtype=np.float32)

    # Move the EE target by +cube_in_hand_y along hand Y: this moves the finger-joint
    # origin toward the cube in the hand-Y direction, centering the cube between fingers.
    # (Positive cube_in_hand_y means cube is in +hand_Y → move finger origin +hand_Y.)
    return (cube_in_hand_y * hand_y).astype(np.float32)


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------


def run_one_episode(
    ik_model,
    phys_model,
    hand_idx: int,
    cube_body_idx: int,
    solver,
    cube_pos_t: "torch.Tensor",
    label: dict,
    cfg: dict,
    seq_id: str,
    dt: float,
    steps_per_ep: int,
    record_every: int,
) -> dict:
    """Run one physics episode and return the sequence dict.

    Uses ik_model (9 DOF) for IK solving and phys_model (16 DOF) for simulation.
    Cube is a real rigid body; position is read from phys state each step.

    Args:
        ik_model:       robot-only model for IK / FK.
        phys_model:     robot + cube model for physics.
        hand_idx:       body index of panda_hand.
        cube_body_idx:  body index of cube in phys_model.
        solver:         Newton solver for phys_model.
        cube_pos_t:     (3,) cube initial position tensor.
        label:          {'reachable': bool, 'success': bool}
        cfg:            geometry config dict.
        seq_id:         sequence ID string.
        dt:             physics timestep [s].
        steps_per_ep:   total physics steps.
        record_every:   record every N steps.

    Returns:
        Sequence dict matching the JSON schema.
    """
    import torch

    sm = WaypointStateMachine(cube_pos_t, label, cfg, torch.device("cpu"))

    # Physics states (phys_model: 16 joint coords, 15 joint DOF)
    n_phys  = phys_model.joint_coord_count   # 16
    n_dof   = phys_model.joint_dof_count     # 15
    state_0 = phys_model.state()
    state_1 = phys_model.state()
    control = phys_model.control()

    has_collider = hasattr(phys_model, "collider")
    contacts = phys_model.collider() if has_collider else None

    # --- Reset state: robot to default home, cube to sampled position -------
    default_robot_q = ik_model.joint_q.numpy()[:_N_ROBOT_JOINTS].astype(np.float32)
    cx, cy, cz = cube_pos_t.tolist()
    cube_init_arr = np.array([cx, cy, cz], dtype=np.float32)

    joint_q_init = np.zeros(n_phys, dtype=np.float32)
    joint_q_init[:_N_ROBOT_JOINTS] = default_robot_q
    # Free-joint layout: [tx, ty, tz, qx, qy, qz, qw]
    joint_q_init[_N_ROBOT_JOINTS:_N_ROBOT_JOINTS + 3] = [cx, cy, cz]
    joint_q_init[_N_ROBOT_JOINTS + 3:n_phys]          = [0.0, 0.0, 0.0, 1.0]   # identity quat

    state_0.joint_q.assign(joint_q_init)
    state_0.joint_qd.assign(np.zeros(n_dof, dtype=np.float32))

    # Pre-allocate IK state for FK evaluation (reused across steps)
    ik_state_fk = ik_model.state()

    gripper_closed = False
    prev_robot_q   = default_robot_q.copy()
    # robot_q_after doubles as robot_q_now for the next step — the post-step joints
    # are identical to the pre-step joints of the following iteration, so we can
    # reuse them without an extra joint_q.numpy() sync.
    robot_q_after  = default_robot_q.copy()
    prev_ee_pos    = None
    # Live cube position (updated from physics each step).
    cube_pos_arr = cube_init_arr.copy()
    _prev_cube_z = cube_init_arr.copy()   # for position-based cube vel estimation (avoids joint_qd sync)

    frames = []

    # Diagnostic: track cube knock events (cube moves >5cm in one step)
    _prev_cube_pos = cube_init_arr.copy()
    _knock_logged  = False

    # Grip-assist body force (reachable+success only).
    # Records EE-cube offset at T_GRIP, then applies a 3D spring-damper force to
    # the cube body at each subsequent step so it tracks the EE in XYZ.
    # This replaces hard BOX-BOX contact lift (which is numerically unstable
    # with ke=100 N/m soft contacts + lateral IK drift during lift).
    # 3D control prevents lateral decoupling from the T_GRIP wrist-rotation knock.
    _grip_offset_xyz  = None              # cube_pos - ee_pos at T_GRIP (3D)
    _prev_cube_vel_xyz = np.zeros(3, dtype=np.float32)
    _prev_ee_xyz      = None
    _grip_force_np    = np.zeros((phys_model.body_count, 6), dtype=np.float32)

    for step in range(steps_per_ep):
        t = step * dt

        # Waypoint state machine → EE target + gripper command
        ee_pos_t, quat, finger_cmd = sm.get_target(t)
        ee_target = ee_pos_t.tolist()
        quat_wxyz = quat.tolist()   # [w, x, y, z]

        # IK on robot-only model, warm-started from current robot joints.
        # robot_q_after from the previous step == pre-step joints of this step
        # (no state change between iterations), so reuse it without a second sync.
        robot_q_now   = robot_q_after
        joint_pos_cmd = solve_ik_single(ik_model, hand_idx, ee_target, robot_q_now, quat_wxyz)

        # ---- Finger centering correction (reachable+success, grasp approach) ---
        # When the IK places the EE within 0.25 m of the CURRENT cube position,
        # shift the EE target to centre the cube between the fingers (hand-Y axis),
        # then re-solve IK from the current solution as warm start.
        # Uses the live cube position so small finger-induced nudges are accounted for.
        if label["reachable"] and label["success"]:
            ee_arr = np.array(ee_target, dtype=np.float32)
            ee_dist = float(np.linalg.norm(ee_arr - cube_pos_arr))
            if ee_dist < 0.25:
                shift = _finger_centering_shift(
                    ik_model, hand_idx, ik_state_fk, joint_pos_cmd, cube_pos_arr
                )
                if np.linalg.norm(shift) > 0.003:   # > 3 mm → worth correcting
                    corrected_target = (ee_arr + shift).tolist()
                    # Position-only re-solve: orientation is preserved by warm start;
                    # passing quat_wxyz here would fight the lateral centering shift.
                    joint_pos_cmd = solve_ik_single(
                        ik_model, hand_idx, corrected_target, joint_pos_cmd, None
                    )

        joint_pos_cmd[7] = finger_cmd
        joint_pos_cmd[8] = finger_cmd

        if (finger_cmd < 0.01) and not gripper_closed:
            gripper_closed = True

        # PD target: pad robot command to phys_model joint count
        # Cube free-joint DOFs have ke=0 so the padding zeros are ignored
        cmd_full = np.zeros(n_phys, dtype=np.float32)
        cmd_full[:_N_ROBOT_JOINTS] = joint_pos_cmd
        control.joint_target_pos = wp.array(cmd_full, dtype=wp.float32)

        # Physics step
        state_0.clear_forces()

        # ---- Grip-assist body force (reachable+success only) ----------------
        # At T_GRIP, record the cube-EE Z offset (the "grasp offset").
        # Each subsequent step: apply a Z-only spring-damper body force to the
        # cube so it tracks the EE vertically.  Only Z is controlled here;
        # 3D lateral+vertical control is handled by the grip-assist spring-damper.
        #
        # EE pos reuses prev_ee_pos from the post-step FK of the previous iteration.
        # The state is unchanged between post-step N−1 and pre-step N, so
        # this is exact (no approximation), saving one eval_fk + body_q.numpy() sync.
        if label["reachable"] and label["success"] and t >= T_GRIP and prev_ee_pos is not None:
            ee_xyz_now = np.array([prev_ee_pos[0], prev_ee_pos[1], prev_ee_pos[2]], dtype=np.float32)

            if _grip_offset_xyz is None:
                # First step at/after T_GRIP: record the 3D EE-cube offset
                _grip_offset_xyz = cube_pos_arr - ee_xyz_now
                _prev_ee_xyz     = ee_xyz_now

            # Target cube pos = EE pos + recorded offset
            target_cube_pos = ee_xyz_now + _grip_offset_xyz
            pos_err         = target_cube_pos - cube_pos_arr

            # EE and cube velocity from position differences (avoids joint_qd.numpy() sync)
            ee_vel  = (ee_xyz_now - _prev_ee_xyz) / dt if _prev_ee_xyz is not None else np.zeros(3, dtype=np.float32)
            vel_err = ee_vel - _prev_cube_vel_xyz

            F = _GRIP_K_P * pos_err + _GRIP_K_D * vel_err
            F[2] += _CUBE_MASS * 9.81  # gravity compensation in Z only
            _grip_force_np[:] = 0.0
            _grip_force_np[cube_body_idx, :3] = F
            state_0.body_f.assign(_grip_force_np)

            _prev_ee_xyz = ee_xyz_now

        if contacts is not None:
            solver.step(state_0, state_1, control, contacts, dt)
        else:
            solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

        # FK → read actual EE + cube positions from body transforms
        newton.eval_fk(phys_model, state_0.joint_q, state_0.joint_qd, state_0)
        body_q = state_0.body_q.numpy()
        ee_pos_now  = body_q[hand_idx][:3].tolist()
        cube_pos_now = body_q[cube_body_idx][:3].tolist()

        # ---- Cube knock diagnostic ------------------------------------------------
        cube_pos_arr_now = np.array(cube_pos_now, dtype=np.float32)
        cube_delta = float(np.linalg.norm(cube_pos_arr_now - _prev_cube_pos))
        if cube_delta > 0.01 and not _knock_logged:   # >1cm displacement in one step
            # Cube velocity (constraint impulse encoded as velocity change)
            all_qd = state_0.joint_qd.numpy()
            cube_vel = all_qd[_N_ROBOT_JOINTS:_N_ROBOT_JOINTS + 3]   # [vx, vy, vz]
            # Finger joint positions (to check asymmetric closure)
            all_q = state_0.joint_q.numpy()
            f1, f2 = float(all_q[7]), float(all_q[8])   # finger1, finger2 positions [m]
            ee_arr = np.array(ee_pos_now, dtype=np.float32)
            cube_arr = cube_pos_arr_now
            # Displacement vector: which direction did cube move
            disp = cube_arr - _prev_cube_pos
            print(
                f"\n    [KNOCK] {seq_id} t={t:.3f}s  cube_delta={cube_delta*100:.1f}cm"
                f"\n      cube_init=[{cube_init_arr[0]:.4f},{cube_init_arr[1]:.4f},{cube_init_arr[2]:.4f}]"
                f"  cube_now={[f'{v:.4f}' for v in cube_pos_now]}"
                f"\n      displacement=[{disp[0]:.4f},{disp[1]:.4f},{disp[2]:.4f}]m"
                f"  cube_vel=[{cube_vel[0]:.3f},{cube_vel[1]:.3f},{cube_vel[2]:.3f}]m/s"
                f"\n      ee_pos=[{ee_arr[0]:.4f},{ee_arr[1]:.4f},{ee_arr[2]:.4f}]"
                f"  cube-ee=[{(cube_arr-ee_arr)[0]:.4f},{(cube_arr-ee_arr)[1]:.4f},{(cube_arr-ee_arr)[2]:.4f}]"
                f"\n      fingers: f1={f1:.4f}m  f2={f2:.4f}m  (cmd={finger_cmd:.4f}m)"
                f"  arm[6]={float(all_q[6]):.4f}rad\n"
            )
            _knock_logged = True
        _prev_cube_pos = cube_pos_arr_now

        # Robot joint positions and velocity (9 values, robot only)
        robot_q_after = state_0.joint_q.numpy()[:_N_ROBOT_JOINTS].astype(np.float32)
        joint_vel = ((robot_q_after - prev_robot_q) / dt).tolist()

        if step % record_every == 0:
            frames.append({
                "step":           step,
                "t":              round(t, 4),
                "joint_pos_cmd":  joint_pos_cmd.tolist(),
                "joint_pos":      robot_q_after.tolist(),
                "joint_vel":      joint_vel,
                "gripper_closed": gripper_closed,
                "robot_pos_w":    [0.0, 0.0, 0.0],
                "ee_pos_w":       ee_pos_now,
                "cube_pos_w":     cube_pos_now,
            })

        prev_robot_q = robot_q_after.copy()
        prev_ee_pos  = ee_pos_now
        cube_pos_arr = cube_pos_arr_now  # update live cube position from physics
        # Cube XYZ velocity from position difference — avoids a joint_qd.numpy() sync.
        # Sufficient for the soft grip-assist spring (k_p=20, dt=0.02).
        _prev_cube_vel_xyz = (cube_pos_arr - _prev_cube_z) / dt
        _prev_cube_z       = cube_pos_arr.copy()

    return {
        "id":                   seq_id,
        "label":                label,
        "cube_init_pos_w":      cube_pos_t.tolist(),
        "cube_horizontal_dist": round(float(cube_pos_t[:2].norm()), 4),
        "frames":               frames,
    }


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def generate_sequences(args):
    import torch

    random.seed(args.seed)

    cfg = DEFAULT_CONFIG
    dt = 0.02         # 50 Hz — fine enough for all phase transitions
    T_END = 10.0      # 10 s gives PD tracking headroom vs kinematic targets
    steps_per_ep = int(T_END / dt)
    RECORD_EVERY = 2  # record at 25 Hz → 250 frames per sequence

    urdf_path = _find_panda_urdf()
    print(f"[generate] Loading Panda URDF: {urdf_path}")
    ik_model, phys_model, hand_idx, cube_body_idx, solver = build_models(urdf_path)
    print(f"[generate] IK model:   {ik_model.body_count} bodies, {ik_model.joint_coord_count} joint coords")
    print(f"[generate] Phys model: {phys_model.body_count} bodies, {phys_model.joint_coord_count} joint coords (robot + cube free joint)")
    print(f"[generate] Solver: {type(solver).__name__}")
    print(f"[generate] Simulating {args.num_sequences} sequences "
          f"({steps_per_ep} steps × {dt:.3f}s = {T_END:.1f}s each, "
          f"recording every {RECORD_EVERY} steps → {steps_per_ep // RECORD_EVERY} frames)")

    all_sequences = []

    for seq_idx in range(args.num_sequences):
        label = sample_label(args.reachable_ratio, args.success_ratio)
        cube_pos_t = sample_cube_pos(label, cfg, torch.device("cpu"))
        seq_id = f"seq_{seq_idx:04d}"

        lbl_str = label_description(label)
        print(f"[generate] {seq_idx + 1}/{args.num_sequences} — {seq_id} ({lbl_str})",
              end="", flush=True)

        seq = run_one_episode(
            ik_model=ik_model,
            phys_model=phys_model,
            hand_idx=hand_idx,
            cube_body_idx=cube_body_idx,
            solver=solver,
            cube_pos_t=cube_pos_t,
            label=label,
            cfg=cfg,
            seq_id=seq_id,
            dt=dt,
            steps_per_ep=steps_per_ep,
            record_every=RECORD_EVERY,
        )
        all_sequences.append(seq)

        n_frames = len(seq["frames"])
        # Diagnostic: peak cube z and whether it cleared lift_height
        if seq["frames"]:
            cube_zs = [fr["cube_pos_w"][2] for fr in seq["frames"]]
            peak_z  = max(cube_zs)
            cleared = "LIFTED" if peak_z >= cfg["lift_height"] else f"peak_z={peak_z:.3f}"
        else:
            cleared = "no frames"
        print(f"  frames={n_frames}  {cleared}")

    # Save
    output_data = {
        "version": "1.0",
        "generated_at": datetime.datetime.now().isoformat(),
        "args": {
            "num_sequences":   args.num_sequences,
            "num_envs":        args.num_envs,
            "reachable_ratio": args.reachable_ratio,
            "success_ratio":   args.success_ratio,
            "seed":            args.seed,
        },
        "config": cfg,
        "sequences": all_sequences,
    }
    save_sequences(output_data, args.output)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Franka cube pick validation sequences (Newton physics)."
    )
    parser.add_argument("--num_sequences",   type=int,   default=100)
    parser.add_argument("--num_envs",        type=int,   default=16,
                        help="Kept for CLI compatibility; physics runs sequentially, this is ignored.")
    parser.add_argument("--output",          type=str,   default=str(_OUTPUTS_DIR / "sequences.json"))
    parser.add_argument("--reachable_ratio", type=float, default=0.7)
    parser.add_argument("--success_ratio",   type=float, default=0.5)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    generate_sequences(args)


if __name__ == "__main__":
    main()
