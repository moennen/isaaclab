"""Tool 1 — Sequence Generator (batched multi-world).

Generates N scripted simulation sequences for the Franka cube pick task.
Sequences cover all four scenario types:

  reachable   + success   (70% × 50% = 35%)
  reachable   + failure   (70% × 50% = 35%)
  unreachable + success   (30% × 50% = 15%)
  unreachable + failure   (30% × 50% = 15%)

Architecture
------------
Runs fully standalone using Newton physics simulation — no Isaac Sim / AppLauncher required.

Two Newton models are used:

  single_model  — robot only (9 DOF, finalized from robot_builder). Used exclusively by
                  the batch IK solver (IKSolver(model=single_model, n_problems=num_worlds)).

  batched_model — N × (robot + cube). Tiled via begin_world/add_builder/end_world.
                  Cube added per-world. Ground plane at scene level.
                  Full physics with PD control and contact friction.

Batch size (--num-worlds, default 16): all num_worlds episodes run simultaneously.
Each outer step:
  1. Python: compute N EE targets from N state machines (O(N) arithmetic, negligible).
  2. GPU: batch IK — IKSolver(n_problems=N) solves all N worlds simultaneously.
  3. GPU: physics — one solver.step() advances all N worlds together.
     Inner substep loop (10 × 2ms) captured as a CUDA graph (first batch only).
  4. One GPU→CPU sync reads body_q for all N worlds.

Cube positions set via state_0.joint_q at each batch reset — model built once.

IK API: IKObjectivePosition, IKObjectiveRotation, IKObjectiveJointLimit, IKJacobianType.
(Old names IKPositionObjective / IKJacobianMode do not exist in this Newton version.)

Usage
-----
python generate_sequences.py \\
    --num_sequences 100 \\
    --num-worlds 16 \\
    --output data/validation/sequences.json
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

_TASK_ROOT   = Path(__file__).parent.parent
_OUTPUTS_DIR = _TASK_ROOT / "data" / "validation"

# ---------------------------------------------------------------------------
# Panda URDF path (shipped with Isaac Sim pip package)
# ---------------------------------------------------------------------------

_FRANKA_URDF_CANDIDATES = [
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
# Newton physics constants (matching example_ik_cube_stacking.py)
# ---------------------------------------------------------------------------

_PANDA_HAND_BODY_LABEL = "panda_hand"  # substring of ModelBuilder.body_label entry

_ARM_KE          = [4500, 4500, 3500, 3500, 2000, 2000, 2000]
_ARM_KD          = [450,  450,  350,  350,  200,  200,  200]
_FINGER_KE       = [100,  100]
_FINGER_KD       = [10,   10]
_ARM_ARMATURE    = [0.30, 0.30, 0.30, 0.30, 0.11, 0.11, 0.11]
_FINGER_ARMATURE = [0.15, 0.15]
_ARM_EFFORT      = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
_FINGER_EFFORT   = [100.0, 100.0]

# Contact parameters.
_CONTACT_KE  = 5.0e4
_CONTACT_KD  = 5.0e2
_CONTACT_KF  = 1.0e3
_CONTACT_MU  = 0.75

# Physics substeps per 20ms frame (must be EVEN for CUDA graph correctness —
# even substeps return state_0 as the current buffer after the loop).
_N_SUBSTEPS = 10

_N_ROBOT_JOINTS = 9   # 7 arm + 2 finger joints

# Initial "ready" joint configuration (arm in workspace, elbow bent down).
# Matching example_ik_cube_stacking.py so the arm starts near the cube workspace.
_HOME_JOINT_Q = [
    -3.6802115e-03,  # joint1
     2.3901723e-02,  # joint2
     3.6804110e-03,  # joint3
    -2.3683236e+00,  # joint4  (elbow bent ~136°)
    -1.2918962e-04,  # joint5
     2.3922248e+00,  # joint6  (~137°, wrist-up)
     7.8549200e-01,  # joint7  (45°)
     0.04,           # finger1 open (4 cm per finger)
     0.04,           # finger2 open
]

_CUBE_HALF_SIZE = 0.025   # 5 cm cube
_CUBE_DENSITY   = 400.0   # kg/m³
_CUBE_MASS      = _CUBE_DENSITY * (2 * _CUBE_HALF_SIZE) ** 3   # 0.100 kg


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_robot_builder(urdf_path: Path) -> tuple:
    """Build unfinalized robot-only ModelBuilder for N-world tiling.

    Includes: URDF, SolverMuJoCo custom attributes, gravity compensation,
    finger BOX shapes (group 1 for BOX-BOX contact with cube), PD gains,
    armature, effort limits.

    Does NOT include: cube, ground plane (added per-world / at scene level).

    Returns:
        robot_mb:         unfinalized ModelBuilder.
        hand_local_idx:   body index of panda_hand within this builder.
        robot_body_count: number of bodies (= cube's local body offset after tiling).
    """
    mb = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.add_urdf(str(urdf_path), floating=False, enable_self_collisions=False,
                parse_visuals_as_colliders=False)

    hand_local_idx = next(
        i for i, k in enumerate(mb.body_label) if _PANDA_HAND_BODY_LABEL in k
    )
    left_finger_body  = next(i for i, k in enumerate(mb.body_label) if "leftfinger"  in k)
    right_finger_body = next(i for i, k in enumerate(mb.body_label) if "rightfinger" in k)

    # ---- Gravity compensation ------------------------------------------------
    gravcomp_dof = mb.custom_attributes["mujoco:jnt_actgravcomp"]
    if gravcomp_dof.values is None:
        gravcomp_dof.values = {}
    for dof_idx in range(7):
        gravcomp_dof.values[dof_idx] = True

    gravcomp_body = mb.custom_attributes["mujoco:gravcomp"]
    if gravcomp_body.values is None:
        gravcomp_body.values = {}
    for body_idx in range(2, mb.body_count):
        gravcomp_body.values[body_idx] = 1.0

    # ---- Collision groups: URDF mesh shapes → group 2 ----------------------
    for i in range(len(mb.shape_body)):
        if mb.shape_body[i] >= 0:
            mb.shape_collision_group[i] = 2

    # ---- Finger BOX shapes — group 1 for BOX-BOX contact with cube ----------
    # Squeeze-mode: SAT Y_pen=15mm << Z_pen=41mm → Y-normal horizontal contact.
    # Horizontal normal × mu friction in Z lifts the cube.
    _cfg_fbox = newton.ModelBuilder.ShapeConfig()
    _cfg_fbox.ke = _CONTACT_KE
    _cfg_fbox.kd = _CONTACT_KD
    _cfg_fbox.kf = _CONTACT_KF
    _cfg_fbox.mu = _CONTACT_MU

    mb.add_shape_box(
        body=left_finger_body,
        xform=wp.transform(wp.vec3(0.0, 0.013, 0.027), wp.quat_identity()),
        hx=0.025, hy=0.013, hz=0.018,
        cfg=_cfg_fbox, label="leftfinger_box",
    )
    mb.shape_collision_group[len(mb.shape_body) - 1] = 1

    mb.add_shape_box(
        body=right_finger_body,
        xform=wp.transform(wp.vec3(0.0, -0.013, 0.027), wp.quat_identity()),
        hx=0.025, hy=0.013, hz=0.018,
        cfg=_cfg_fbox, label="rightfinger_box",
    )
    mb.shape_collision_group[len(mb.shape_body) - 1] = 1

    # ---- PD gains + armature + effort limits --------------------------------
    all_ke     = _ARM_KE      + _FINGER_KE
    all_kd     = _ARM_KD      + _FINGER_KD
    all_arm    = _ARM_ARMATURE + _FINGER_ARMATURE
    all_effort = _ARM_EFFORT  + _FINGER_EFFORT
    for i in range(_N_ROBOT_JOINTS):
        mb.joint_target_ke[i]    = float(all_ke[i])
        mb.joint_target_kd[i]    = float(all_kd[i])
        mb.joint_armature[i]     = float(all_arm[i])
        mb.joint_effort_limit[i] = float(all_effort[i])

    # ---- Initial joint configuration: arm in "ready" workspace pose ---------
    # Without this the arm starts at all-zero joints (fully extended upward,
    # EE at [0.088, 0, 0.926]), far from the cube workspace, and PD tracking
    # never converges to the trajectory within the episode.
    for i in range(_N_ROBOT_JOINTS):
        mb.joint_q[i]          = float(_HOME_JOINT_Q[i])
        mb.joint_target_pos[i] = float(_HOME_JOINT_Q[i])

    robot_body_count = mb.body_count
    return mb, hand_local_idx, robot_body_count


def build_batched_model(robot_builder: newton.ModelBuilder, num_worlds: int) -> tuple:
    """Tile num_worlds robot+cube environments into one batched Newton model.

    Each world: robot (from robot_builder via add_builder) + 1 cube (added per-world).
    Ground plane: added once at scene level (shared).

    Cube initial positions are set via state_0.joint_q at reset time; the model is
    built once and reused across all batches.

    Returns:
        batched_model: finalized Newton Model (N × (robot + cube)).
        solver:        SolverMuJoCo for batched_model (ls_parallel=True).
    """
    cfg_cube = newton.ModelBuilder.ShapeConfig()
    cfg_cube.density = _CUBE_DENSITY
    cfg_cube.ke      = _CONTACT_KE
    cfg_cube.kd      = _CONTACT_KD
    cfg_cube.kf      = _CONTACT_KF
    cfg_cube.mu      = _CONTACT_MU

    scene = newton.ModelBuilder()
    for _world_id in range(num_worlds):
        scene.begin_world()
        scene.add_builder(robot_builder)
        # Cube at ground level — actual positions set via state_0.joint_q at reset
        cube_body = scene.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, _CUBE_HALF_SIZE), wp.quat_identity()),
            mass=0.0, label="cube",
        )
        scene.add_shape_box(
            body=cube_body,
            hx=_CUBE_HALF_SIZE, hy=_CUBE_HALF_SIZE, hz=_CUBE_HALF_SIZE,
            cfg=cfg_cube, label="cube_shape",
        )
        scene.shape_collision_group[len(scene.shape_body) - 1] = 1
        scene.end_world()

    scene.add_ground_plane()
    batched_model = scene.finalize()

    solver = newton.solvers.SolverMuJoCo(
        batched_model,
        solver="newton",
        integrator="implicitfast",
        iterations=20,
        ls_parallel=True,
        ls_iterations=100,
        nconmax=512 * num_worlds,
        njmax=1000 * num_worlds,
        cone="elliptic",
        impratio=1000.0,
    )
    return batched_model, solver


# ---------------------------------------------------------------------------
# Batch IK setup
# ---------------------------------------------------------------------------

def build_batch_ik(single_model, hand_local_idx: int, num_worlds: int) -> tuple:
    """Create persistent batch IK solvers for two-stage solving.

    Stage 1 (ANALYTIC, position-only, 60 iters): anchors arm in correct workspace
      basin before orientation gradient is added (avoids bad local minima).
    Stage 2 (AUTODIFF, position+rotation, 100 iters): refines with orientation.

    Targets are updated each step by reassigning the .target_positions /
    .target_rotations attributes on objective objects before calling step().
    This works because IKSolver reads self.target_positions at call time.

    Returns:
        joint_q_ik:    wp.array(shape=(num_worlds, ik_dofs), dtype=float32).
        pos_obj_1:     IKObjectivePosition for stage 1.
        stage1_solver: IKSolver for position-only stage.
        pos_obj_2:     IKObjectivePosition for stage 2.
        rot_obj_2:     IKObjectiveRotation for stage 2.
        stage2_solver: IKSolver for position+rotation stage.
    """
    ik_dofs = single_model.joint_coord_count

    dummy_pos = wp.zeros(num_worlds, dtype=wp.vec3)
    dummy_rot = wp.array([wp.vec4(0.0, 0.0, 0.0, 1.0)] * num_worlds, dtype=wp.vec4)

    joint_limit_lower = single_model.joint_limit_lower
    joint_limit_upper = single_model.joint_limit_upper

    # Stage 1: position-only (ANALYTIC)
    pos_obj_1   = nik.IKObjectivePosition(
        link_index=hand_local_idx,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=dummy_pos,
    )
    limit_obj_1 = nik.IKObjectiveJointLimit(
        joint_limit_lower=joint_limit_lower,
        joint_limit_upper=joint_limit_upper,
        weight=10.0,
    )
    stage1_solver = nik.IKSolver(
        model=single_model, n_problems=num_worlds,
        objectives=[pos_obj_1, limit_obj_1],
        jacobian_mode=nik.IKJacobianType.ANALYTIC,
    )

    # Stage 2: position + rotation (AUTODIFF)
    pos_obj_2   = nik.IKObjectivePosition(
        link_index=hand_local_idx,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=dummy_pos,
    )
    limit_obj_2 = nik.IKObjectiveJointLimit(
        joint_limit_lower=joint_limit_lower,
        joint_limit_upper=joint_limit_upper,
        weight=10.0,
    )
    rot_obj_2   = nik.IKObjectiveRotation(
        link_index=hand_local_idx,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=dummy_rot,
        weight=0.3,
    )
    stage2_solver = nik.IKSolver(
        model=single_model, n_problems=num_worlds,
        objectives=[pos_obj_2, limit_obj_2, rot_obj_2],
        jacobian_mode=nik.IKJacobianType.AUTODIFF,
    )

    home_q = single_model.joint_q.numpy()[:_N_ROBOT_JOINTS].astype(np.float32)
    joint_q_ik = wp.array(
        np.tile(home_q, (num_worlds, 1)).astype(np.float32),
        dtype=wp.float32,
    )
    return joint_q_ik, pos_obj_1, stage1_solver, pos_obj_2, rot_obj_2, stage2_solver


# ---------------------------------------------------------------------------
# State reset
# ---------------------------------------------------------------------------

def reset_batch_state(
    batched_model,
    state_0,
    state_1,
    cube_positions_np: np.ndarray,
    default_robot_q: np.ndarray,
    joint_q_ik,
    num_worlds: int,
) -> None:
    """Reset all worlds: robot → home pose, cubes → sampled positions.

    Also resets joint_q_ik warm-start to home pose.
    """
    n_coord = batched_model.joint_coord_count // num_worlds
    n_dof   = batched_model.joint_dof_count   // num_worlds

    joint_q  = np.zeros(batched_model.joint_coord_count, dtype=np.float32)
    joint_qd = np.zeros(batched_model.joint_dof_count,   dtype=np.float32)

    for w in range(num_worlds):
        base = w * n_coord
        joint_q[base : base + _N_ROBOT_JOINTS] = default_robot_q
        cx, cy, cz = cube_positions_np[w]
        joint_q[base + _N_ROBOT_JOINTS     : base + _N_ROBOT_JOINTS + 3] = [cx, cy, cz]
        joint_q[base + _N_ROBOT_JOINTS + 3 : base + _N_ROBOT_JOINTS + 7] = [0.0, 0.0, 0.0, 1.0]

    state_0.joint_q.assign(joint_q)
    state_0.joint_qd.assign(joint_qd)
    state_1.joint_q.assign(joint_q)
    state_1.joint_qd.assign(joint_qd)
    joint_q_ik.assign(np.tile(default_robot_q, (num_worlds, 1)).astype(np.float32))


# ---------------------------------------------------------------------------
# Physics inner loop (CUDA graph target)
# ---------------------------------------------------------------------------

def _simulate(solver, state_0, state_1, control, contacts, sub_dt, n_substeps):
    """Run n_substeps physics substeps.

    n_substeps MUST be even — with an even count, state_0 holds the current
    result after the loop (both for Python callers and for CUDA graph replay).

    This function is called directly during CUDA graph capture (recording the
    GPU kernel sequence), then wp.capture_launch(graph) replays it each step.
    During capture, the Python-level state swaps determine which buffer each
    kernel call writes to; the graph records the resulting kernel sequence.
    """
    for _ in range(n_substeps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, sub_dt)
        state_0, state_1 = state_1, state_0


# ---------------------------------------------------------------------------
# Batch episode runner
# ---------------------------------------------------------------------------

def run_batch_episode(
    batched_model,
    solver,
    single_model,
    hand_local_idx: int,
    robot_body_count: int,
    num_worlds: int,
    state_0,
    state_1,
    control,
    joint_q_ik,
    pos_obj_1,
    stage1_solver,
    pos_obj_2,
    rot_obj_2,
    stage2_solver,
    state_machines: list,
    cube_init_positions_np: np.ndarray,
    seq_ids: list,
    dt: float,
    steps_per_ep: int,
    record_every: int,
    graph_sim,  # mutable list [graph_or_None] — captured on first call
) -> list:
    """Run one batch of num_worlds episodes simultaneously.

    graph_sim is a list of length 1 that holds the CUDA graph (or None before
    first capture). It is populated on the first call and reused thereafter.

    Returns:
        frames_per_world: list of num_worlds frame lists.
    """
    n_coord_per_world    = batched_model.joint_coord_count // num_worlds
    # joint_target_pos uses DOF count (free joint = 6 DOFs vs 7 coords); derive from actual size
    n_ctrl_per_world     = len(control.joint_target_pos) // num_worlds
    num_bodies_per_world = batched_model.body_count // num_worlds
    cube_local_idx       = robot_body_count   # cube added right after robot

    sub_dt = dt / _N_SUBSTEPS

    # ---- Reset ---------------------------------------------------------------
    default_robot_q = single_model.joint_q.numpy()[:_N_ROBOT_JOINTS].astype(np.float32)
    reset_batch_state(
        batched_model, state_0, state_1,
        cube_init_positions_np, default_robot_q, joint_q_ik, num_worlds,
    )
    newton.eval_fk(batched_model, state_0.joint_q, state_0.joint_qd, state_0)

    # ---- Capture CUDA graph (first batch only) --------------------------------
    # Note: SolverMuJoCo internally switches CUDA streams, which is not allowed
    # during ScopedCapture. If capture fails, we fall back to eager mode — the
    # N-world parallelism already provides the main speedup.
    # graph_sim[0]: None = not yet tried, False = disabled, graph = active
    use_graph = graph_sim[0] is not None and graph_sim[0] is not False
    if graph_sim[0] is None and wp.get_device().is_cuda:
        try:
            with wp.ScopedCapture() as capture:
                contacts_captured = batched_model.collide(state_0)
                _simulate(solver, state_0, state_1, control, contacts_captured,
                          sub_dt, _N_SUBSTEPS)
            graph_sim[0] = capture.graph
            use_graph = True
            print(f"  [CUDA graph captured]", flush=True)
        except Exception:
            # Mark as permanently disabled so we don't retry each batch.
            graph_sim[0] = False
            # State may have partially advanced — reset to the clean initial state.
            reset_batch_state(
                batched_model, state_0, state_1,
                cube_init_positions_np, default_robot_q, joint_q_ik, num_worlds,
            )
            newton.eval_fk(batched_model, state_0.joint_q, state_0.joint_qd, state_0)

    # Fallback contacts object (CPU path / pre-graph)
    contacts_fallback = None
    if not use_graph:
        try:
            contacts_fallback = batched_model.collide(state_0)
        except AttributeError:
            contacts_fallback = (batched_model.collider()
                                 if hasattr(batched_model, "collider") else None)

    # ---- Per-world bookkeeping -----------------------------------------------
    gripper_closed   = [False] * num_worlds
    prev_robot_q     = np.tile(default_robot_q, (num_worlds, 1))
    frames_per_world = [[] for _ in range(num_worlds)]
    prev_cube_pos    = cube_init_positions_np.copy()
    knock_logged     = [False] * num_worlds

    # ---- Episode loop --------------------------------------------------------
    for step in range(steps_per_ep):
        t = step * dt

        # 1. N state machines → EE targets + finger commands
        ee_targets:  list[list[float]] = []
        quats_wxyz:  list[list[float]] = []
        finger_cmds: np.ndarray = np.zeros(num_worlds, dtype=np.float32)

        for w in range(num_worlds):
            ee_pos_t, quat, finger_cmd = state_machines[w].get_target(t)
            ee_targets.append(ee_pos_t.tolist())
            quats_wxyz.append(quat.tolist())   # [w, x, y, z]
            finger_cmds[w] = float(finger_cmd)
            if float(finger_cmd) < 0.01 and not gripper_closed[w]:
                gripper_closed[w] = True

        # 2. Batch IK — Stage 1: position-only ANALYTIC
        pos_arr = wp.array([wp.vec3(*t_) for t_ in ee_targets], dtype=wp.vec3)
        pos_obj_1.target_positions = pos_arr   # updated before step()
        stage1_solver.step(joint_q_ik, joint_q_ik, iterations=60)

        # Stage 2: position + rotation AUTODIFF
        # waypoint_ik uses [w, x, y, z]; warp vec4 stores [x, y, z, w]
        rot_arr = wp.array(
            [wp.vec4(q[1], q[2], q[3], q[0]) for q in quats_wxyz], dtype=wp.vec4
        )
        pos_obj_2.target_positions = pos_arr
        rot_obj_2.target_rotations = rot_arr
        stage2_solver.step(joint_q_ik, joint_q_ik, iterations=100)

        # 3. Set joint targets for all worlds (one CPU→GPU write)
        joint_q_ik_np = joint_q_ik.numpy()   # (N, 9) — small sync
        ctrl_np = np.zeros((num_worlds, n_ctrl_per_world), dtype=np.float32)
        ctrl_np[:, :_N_ROBOT_JOINTS] = joint_q_ik_np
        ctrl_np[:, 7] = finger_cmds
        ctrl_np[:, 8] = finger_cmds
        control.joint_target_pos.assign(ctrl_np.reshape(-1))

        # 4. Physics — CUDA graph (collide + 10 substeps) or eager fallback
        if use_graph:
            wp.capture_launch(graph_sim[0])
        else:
            try:
                contacts_fallback = batched_model.collide(state_0)
            except AttributeError:
                pass
            _simulate(solver, state_0, state_1, control, contacts_fallback,
                      sub_dt, _N_SUBSTEPS)

        # 5. Read state — one GPU→CPU sync for all worlds
        newton.eval_fk(batched_model, state_0.joint_q, state_0.joint_qd, state_0)
        body_q_np  = state_0.body_q.numpy()
        joint_q_np = state_0.joint_q.numpy()

        # 6. Per-world extraction + recording
        for w in range(num_worlds):
            global_hand = w * num_bodies_per_world + hand_local_idx
            global_cube = w * num_bodies_per_world + cube_local_idx

            ee_pos_now   = body_q_np[global_hand][:3].tolist()
            cube_pos_now = body_q_np[global_cube][:3].tolist()

            base = w * n_coord_per_world
            robot_q_now = joint_q_np[base : base + _N_ROBOT_JOINTS].astype(np.float32)
            joint_vel_w = ((robot_q_now - prev_robot_q[w]) / dt).tolist()

            cube_arr   = np.array(cube_pos_now, dtype=np.float32)
            cube_delta = float(np.linalg.norm(cube_arr - prev_cube_pos[w]))
            if cube_delta > 0.01 and not knock_logged[w]:
                print(
                    f"\n    [KNOCK] {seq_ids[w]} t={t:.3f}s  delta={cube_delta*100:.1f}cm"
                    f"  cube=[{cube_pos_now[0]:.3f},{cube_pos_now[1]:.3f},{cube_pos_now[2]:.3f}]"
                )
                knock_logged[w] = True
            prev_cube_pos[w] = cube_arr

            if step % record_every == 0:
                frames_per_world[w].append({
                    "step":           step,
                    "t":              round(t, 4),
                    "joint_pos_cmd":  ctrl_np[w, :_N_ROBOT_JOINTS].tolist(),  # actual cmd (with finger override)
                    "joint_pos":      robot_q_now.tolist(),
                    "joint_vel":      joint_vel_w,
                    "gripper_closed": gripper_closed[w],
                    "robot_pos_w":    [0.0, 0.0, 0.0],
                    "ee_pos_w":       ee_pos_now,
                    "cube_pos_w":     cube_pos_now,
                })

            prev_robot_q[w] = robot_q_now.copy()

    return frames_per_world


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_sequences(args):
    import torch

    random.seed(args.seed + args.seq_id_start)

    cfg = DEFAULT_CONFIG
    dt = 0.02         # 50 Hz
    T_END = 10.0
    steps_per_ep = int(T_END / dt)   # 500
    RECORD_EVERY = 1                  # 500 frames at 50 Hz — one frame per outer step

    num_worlds = args.num_worlds

    urdf_path = _find_panda_urdf()
    print(f"[generate] Panda URDF: {urdf_path}")

    # ---- Build models (once for the entire run) --------------------------------
    print(f"[generate] Building models (num_worlds={num_worlds}) ...")

    robot_mb, hand_local_idx, robot_body_count = build_robot_builder(urdf_path)
    batched_model, solver = build_batched_model(robot_mb, num_worlds)
    # single_model: finalized from robot_builder, used for batch IK
    single_model = robot_mb.finalize()

    nbpw = batched_model.body_count // num_worlds
    ncpw = batched_model.joint_coord_count // num_worlds
    print(
        f"[generate] Batched model: {batched_model.body_count} bodies, "
        f"{batched_model.joint_coord_count} coords  "
        f"({nbpw} bodies/world, {ncpw} coords/world)"
    )
    print(f"[generate] hand_local_idx={hand_local_idx}  robot_body_count={robot_body_count}")
    print(f"[generate] Solver: {type(solver).__name__}")

    # ---- Build batch IK solvers (once) -----------------------------------------
    (joint_q_ik, pos_obj_1, stage1_solver,
     pos_obj_2, rot_obj_2, stage2_solver) = build_batch_ik(
        single_model, hand_local_idx, num_worlds
    )

    # ---- Allocate shared states (once, reused across batches) ------------------
    state_0  = batched_model.state()
    state_1  = batched_model.state()
    control  = batched_model.control()
    graph_sim = [None]   # mutable container; populated on first batch

    total_seqs = args.num_sequences
    id_range   = (f"seq_{args.seq_id_start:04d}.."
                  f"seq_{args.seq_id_start + total_seqs - 1:04d}")
    print(
        f"[generate] Simulating {total_seqs} sequences ({id_range}) "
        f"in batches of {num_worlds} | "
        f"{steps_per_ep} steps × {dt:.3f}s = {T_END:.1f}s | "
        f"{steps_per_ep // RECORD_EVERY} frames"
    )

    all_sequences: list[dict] = []
    seq_counter = 0

    while seq_counter < total_seqs:
        batch_size = min(num_worlds, total_seqs - seq_counter)

        # Sample labels + cube positions for this batch
        batch_labels   = [sample_label(args.reachable_ratio, args.success_ratio)
                          for _ in range(batch_size)]
        batch_cube_pos = [sample_cube_pos(lbl, cfg, torch.device("cpu"))
                          for lbl in batch_labels]
        batch_seq_ids  = [
            f"seq_{args.seq_id_start + seq_counter + i:04d}"
            for i in range(batch_size)
        ]

        # Pad last batch to full num_worlds (extra worlds run but are discarded)
        while len(batch_labels) < num_worlds:
            batch_labels.append(batch_labels[-1])
            batch_cube_pos.append(batch_cube_pos[-1])
            batch_seq_ids.append("seq_pad")

        cube_init_np = np.array(
            [pos.numpy().astype(np.float32) for pos in batch_cube_pos],
            dtype=np.float32,
        )
        state_machines = [
            WaypointStateMachine(batch_cube_pos[w], batch_labels[w], cfg,
                                 torch.device("cpu"))
            for w in range(num_worlds)
        ]

        lbl_strs = [label_description(batch_labels[w]) for w in range(batch_size)]
        batch_num = seq_counter // num_worlds + 1
        total_batches = (total_seqs + num_worlds - 1) // num_worlds
        print(
            f"[generate] batch {batch_num}/{total_batches} "
            f"({batch_size} active, {num_worlds - batch_size} pad): "
            + " | ".join(f"{batch_seq_ids[w]}({lbl_strs[w]})"
                         for w in range(min(batch_size, 4)))
            + ("..." if batch_size > 4 else ""),
            end="", flush=True,
        )

        frames_per_world = run_batch_episode(
            batched_model=batched_model,
            solver=solver,
            single_model=single_model,
            hand_local_idx=hand_local_idx,
            robot_body_count=robot_body_count,
            num_worlds=num_worlds,
            state_0=state_0,
            state_1=state_1,
            control=control,
            joint_q_ik=joint_q_ik,
            pos_obj_1=pos_obj_1,
            stage1_solver=stage1_solver,
            pos_obj_2=pos_obj_2,
            rot_obj_2=rot_obj_2,
            stage2_solver=stage2_solver,
            state_machines=state_machines,
            cube_init_positions_np=cube_init_np,
            seq_ids=batch_seq_ids,
            dt=dt,
            steps_per_ep=steps_per_ep,
            record_every=RECORD_EVERY,
            graph_sim=graph_sim,
        )

        for w in range(batch_size):
            cube_zs = [fr["cube_pos_w"][2] for fr in frames_per_world[w]]
            peak_z  = max(cube_zs) if cube_zs else 0.0
            cleared = "LIFTED" if peak_z >= cfg["lift_height"] else f"peak_z={peak_z:.3f}"
            print(
                f"\n  [{batch_seq_ids[w]}] frames={len(frames_per_world[w])}  {cleared}",
                end="",
            )
            all_sequences.append({
                "id":                   batch_seq_ids[w],
                "label":                batch_labels[w],
                "cube_init_pos_w":      batch_cube_pos[w].tolist(),
                "cube_horizontal_dist": round(float(batch_cube_pos[w][:2].norm()), 4),
                "frames":               frames_per_world[w],
            })

        print()
        seq_counter += batch_size

    output_data = {
        "version": "1.0",
        "generated_at": datetime.datetime.now().isoformat(),
        "args": {
            "num_sequences":   total_seqs,
            "num_worlds":      num_worlds,
            "reachable_ratio": args.reachable_ratio,
            "success_ratio":   args.success_ratio,
            "seed":            args.seed,
            "seq_id_start":    args.seq_id_start,
        },
        "config": cfg,
        "sequences": all_sequences,
    }
    save_sequences(output_data, args.output)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Franka cube pick validation sequences (batched Newton physics)."
    )
    parser.add_argument("--num_sequences",   type=int,   default=100)
    parser.add_argument("--num-worlds",      type=int,   default=16,
                        help="Worlds per batch (batch size). Default 16. "
                             "All worlds run simultaneously on GPU.")
    parser.add_argument("--output",          type=str,   default=str(_OUTPUTS_DIR / "reference_ik_sequences.json"))
    parser.add_argument("--reachable_ratio", type=float, default=0.7)
    parser.add_argument("--success_ratio",   type=float, default=0.5)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--seq_id_start",    type=int,   default=0,
                        help="Shift seq IDs and seed by this offset "
                             "(for partial regeneration; not needed for performance).")
    args = parser.parse_args()
    generate_sequences(args)


if __name__ == "__main__":
    main()
