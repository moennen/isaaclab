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
from _common.waypoint_ik import WaypointStateMachine, T_GRASP, T_GRIP

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

# PD gains — matching example_ik_cube_stacking.py (the Newton reference for this task).
# High stiffness + gravity compensation = arm tracks IK target within ~1 mm.
_ARM_KE          = [4500, 4500, 3500, 3500, 2000, 2000, 2000]
_ARM_KD          = [450,  450,  350,  350,  200,  200,  200]
_FINGER_KE       = [100,  100]
_FINGER_KD       = [10,   10]
_ARM_ARMATURE    = [0.30, 0.30, 0.30, 0.30, 0.11, 0.11, 0.11]
_FINGER_ARMATURE = [0.15, 0.15]
_ARM_EFFORT      = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
_FINGER_EFFORT   = [100.0, 100.0]

# Contact parameters — matching example_ik_cube_stacking.py.
# ke=5e4: strong normal spring → enough normal force at small penetration.
# kf=1e3: tangential spring → resists sliding within the friction cone.
# mu=0.75: sufficient with impratio=1000 (friction near-rigid constraint).
_CONTACT_KE  = 5.0e4   # N/m   normal spring stiffness
_CONTACT_KD  = 5.0e2   # N·s/m normal damping
_CONTACT_KF  = 1.0e3   # N/m   tangential (friction) spring stiffness
_CONTACT_MU  = 0.75    # friction coefficient

# Number of physics substeps per 20ms frame.
# ke=5e4, m=0.1kg → ω_n=707 rad/s → contact period T=8.9ms.
# dt=20ms > T causes large impulses from small overlaps.
# 10 substeps → sub_dt=2ms < T → numerically stable.
# Matches example_ik_cube_stacking.py (sim_substeps=10, sim_dt=1.67ms).
_N_SUBSTEPS = 10

# Finger open/close positions [m]
_FINGER_OPEN  = 0.04
_FINGER_CLOSE = 0.0

# Number of robot joint coordinates (arm 0-6 + 2 fingers)
_N_ROBOT_JOINTS = 9

# Cube rigid body properties
_CUBE_HALF_SIZE = 0.025    # half-extent per axis [m] → 5 cm cube
_CUBE_DENSITY   = 400.0    # kg/m³ — midpoint of example range (300-500)
_CUBE_MASS      = _CUBE_DENSITY * (2 * _CUBE_HALF_SIZE) ** 3   # 0.100 kg



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
        solver:         Newton solver for phys_model
    """
    # ---- IK model (robot only) -----------------------------------------------
    mb_ik = newton.ModelBuilder()
    mb_ik.add_ground_plane()
    mb_ik.add_urdf(str(urdf_path), floating=False, enable_self_collisions=False,
                   parse_visuals_as_colliders=False)
    ik_model = mb_ik.finalize()

    hand_idx = next(
        i for i, l in enumerate(mb_ik.body_label) if _PANDA_HAND_BODY_LABEL in l
    )

    # ---- Physics model (robot + cube) ----------------------------------------
    mb_phys = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb_phys)
    mb_phys.add_ground_plane()
    mb_phys.add_urdf(str(urdf_path), floating=False, enable_self_collisions=False,
                     parse_visuals_as_colliders=False)

    # ---- Gravity compensation ------------------------------------------------
    # Without gravcomp the PD fights link weight → tracking errors during grasp/lift.
    gravcomp_dof = mb_phys.custom_attributes["mujoco:jnt_actgravcomp"]
    if gravcomp_dof.values is None:
        gravcomp_dof.values = {}
    for dof_idx in range(7):   # arm DOFs 0-6 only
        gravcomp_dof.values[dof_idx] = True

    gravcomp_body = mb_phys.custom_attributes["mujoco:gravcomp"]
    if gravcomp_body.values is None:
        gravcomp_body.values = {}
    for body_idx in range(2, len(mb_phys.body_label)):   # skip world (0) and fixed base (1)
        gravcomp_body.values[body_idx] = 1.0

    # ---- Collision group assignment ------------------------------------------
    # Move all URDF shapes to group 2 (mesh-based, not BOX-detectable by Newton).
    # Explicit BOX finger shapes added below in group 1 for BOX-BOX contact.
    _left_finger_body  = next(i for i, lbl in enumerate(mb_phys.body_label) if "leftfinger"  in lbl)
    _right_finger_body = next(i for i, lbl in enumerate(mb_phys.body_label) if "rightfinger" in lbl)
    for _i in range(len(mb_phys.shape_body)):
        if mb_phys.shape_body[_i] >= 0:   # skip ground (body=-1)
            mb_phys.shape_collision_group[_i] = 2

    # ---- Finger BOX shapes ---------------------------------------------------
    # Squeeze-mode geometry: SAT Z_pen=41mm >> Y_pen=15mm → Y-normal contact.
    # Squeeze normal × mu friction in Z lifts cube. Wide hx tolerates XY drift.
    _cfg_fbox = newton.ModelBuilder.ShapeConfig()
    _cfg_fbox.ke = _CONTACT_KE
    _cfg_fbox.kd = _CONTACT_KD
    _cfg_fbox.kf = _CONTACT_KF
    _cfg_fbox.mu = _CONTACT_MU

    mb_phys.add_shape_box(
        body=_left_finger_body,
        xform=wp.transform(wp.vec3(0.0, 0.013, 0.027), wp.quat_identity()),
        hx=0.025, hy=0.013, hz=0.018,
        cfg=_cfg_fbox,
        label="leftfinger_box",
    )
    mb_phys.shape_collision_group[len(mb_phys.shape_body) - 1] = 1

    mb_phys.add_shape_box(
        body=_right_finger_body,
        xform=wp.transform(wp.vec3(0.0, -0.013, 0.027), wp.quat_identity()),
        hx=0.025, hy=0.013, hz=0.018,
        cfg=_cfg_fbox,
        label="rightfinger_box",
    )
    mb_phys.shape_collision_group[len(mb_phys.shape_body) - 1] = 1

    # ---- PD gains + armature + effort limits ---------------------------------
    all_ke     = _ARM_KE      + _FINGER_KE
    all_kd     = _ARM_KD      + _FINGER_KD
    all_arm    = _ARM_ARMATURE + _FINGER_ARMATURE
    all_effort = _ARM_EFFORT  + _FINGER_EFFORT
    for i in range(_N_ROBOT_JOINTS):
        mb_phys.joint_target_ke[i]    = float(all_ke[i])
        mb_phys.joint_target_kd[i]    = float(all_kd[i])
        mb_phys.joint_armature[i]     = float(all_arm[i])
        mb_phys.joint_effort_limit[i] = float(all_effort[i])

    # ---- Cube rigid body -----------------------------------------------------
    cfg_cube         = newton.ModelBuilder.ShapeConfig()
    cfg_cube.density = _CUBE_DENSITY
    cfg_cube.ke      = _CONTACT_KE
    cfg_cube.kd      = _CONTACT_KD
    cfg_cube.kf      = _CONTACT_KF
    cfg_cube.mu      = _CONTACT_MU

    cube_body_idx = mb_phys.add_body(mass=0.0, label="cube")
    mb_phys.add_shape_box(
        body=cube_body_idx,
        hx=_CUBE_HALF_SIZE, hy=_CUBE_HALF_SIZE, hz=_CUBE_HALF_SIZE,
        cfg=cfg_cube,
        label="cube_shape",
    )
    mb_phys.shape_collision_group[len(mb_phys.shape_body) - 1] = 1

    phys_model = mb_phys.finalize()

    # ---- Solver — parameters mirror example_ik_cube_stacking.py -------------
    # impratio=1000: friction near-rigid → cube stays gripped during arm motion.
    # cone=elliptic: accurate 3D friction cone.
    solver = newton.solvers.SolverMuJoCo(
        phys_model,
        solver="newton",
        integrator="implicitfast",
        iterations=20,
        ls_iterations=100,
        nconmax=512,
        njmax=1000,
        cone="elliptic",
        impratio=1000.0,
    )

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

    frames = []

    # Diagnostic: track cube knock events (cube moves >1cm in one step)
    _prev_cube_pos = cube_init_arr.copy()
    _knock_logged  = False

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

        # ---- Finger centering correction (reachable+success, approach only) ---
        if label["reachable"] and label["success"] and t < T_GRASP:
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

        # Physics substep loop — sub_dt=2ms keeps contact numerically stable.
        state_0.clear_forces()
        sub_dt = dt / _N_SUBSTEPS
        for _sub in range(_N_SUBSTEPS):
            if contacts is not None:
                solver.step(state_0, state_1, control, contacts, sub_dt)
            else:
                solver.step(state_0, state_1, control, None, sub_dt)
            state_0, state_1 = state_1, state_0

        # FK → read actual EE + cube positions from body transforms
        newton.eval_fk(phys_model, state_0.joint_q, state_0.joint_qd, state_0)
        body_q = state_0.body_q.numpy()
        ee_pos_now   = body_q[hand_idx][:3].tolist()
        cube_pos_now = body_q[cube_body_idx][:3].tolist()

        # ---- Cube knock diagnostic ------------------------------------------------
        cube_pos_arr_now = np.array(cube_pos_now, dtype=np.float32)
        cube_delta = float(np.linalg.norm(cube_pos_arr_now - _prev_cube_pos))
        if cube_delta > 0.01 and not _knock_logged:   # >1cm displacement in one step
            all_qd   = state_0.joint_qd.numpy()
            cube_vel = all_qd[_N_ROBOT_JOINTS:_N_ROBOT_JOINTS + 3]
            all_q    = state_0.joint_q.numpy()
            f1, f2   = float(all_q[7]), float(all_q[8])
            ee_arr   = np.array(ee_pos_now, dtype=np.float32)
            disp     = cube_pos_arr_now - _prev_cube_pos
            print(
                f"\n    [KNOCK] {seq_id} t={t:.3f}s  cube_delta={cube_delta*100:.1f}cm"
                f"\n      cube_init=[{cube_init_arr[0]:.4f},{cube_init_arr[1]:.4f},{cube_init_arr[2]:.4f}]"
                f"  cube_now={[f'{v:.4f}' for v in cube_pos_now]}"
                f"\n      displacement=[{disp[0]:.4f},{disp[1]:.4f},{disp[2]:.4f}]m"
                f"  cube_vel=[{cube_vel[0]:.3f},{cube_vel[1]:.3f},{cube_vel[2]:.3f}]m/s"
                f"\n      ee_pos=[{ee_arr[0]:.4f},{ee_arr[1]:.4f},{ee_arr[2]:.4f}]"
                f"  cube-ee=[{(cube_pos_arr_now-ee_arr)[0]:.4f},{(cube_pos_arr_now-ee_arr)[1]:.4f},{(cube_pos_arr_now-ee_arr)[2]:.4f}]"
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
        cube_pos_arr = cube_pos_arr_now

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

    # When running parallel shards, each shard uses seed = base_seed + seq_id_start
    # so that shards produce non-overlapping random sequences while remaining reproducible.
    effective_seed = args.seed + args.seq_id_start
    random.seed(effective_seed)

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
    id_range = f"seq_{args.seq_id_start:04d}..seq_{args.seq_id_start + args.num_sequences - 1:04d}"
    print(f"[generate] Simulating {args.num_sequences} sequences ({id_range}) "
          f"({steps_per_ep} steps × {dt:.3f}s = {T_END:.1f}s each, "
          f"recording every {RECORD_EVERY} steps → {steps_per_ep // RECORD_EVERY} frames)")

    all_sequences = []

    for seq_idx in range(args.num_sequences):
        label = sample_label(args.reachable_ratio, args.success_ratio)
        cube_pos_t = sample_cube_pos(label, cfg, torch.device("cpu"))
        global_seq_idx = args.seq_id_start + seq_idx
        seq_id = f"seq_{global_seq_idx:04d}"

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
    parser.add_argument("--seq_id_start",    type=int,   default=0,
                        help="Start seq IDs at this offset (for parallel shards). "
                             "Effective seed = seed + seq_id_start.")
    args = parser.parse_args()

    generate_sequences(args)


if __name__ == "__main__":
    main()
