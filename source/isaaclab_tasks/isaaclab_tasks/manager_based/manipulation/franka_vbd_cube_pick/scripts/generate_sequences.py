# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tool 1 — VBD Sequence Generator (batched multi-world).

Generates N scripted simulation sequences for the franka_vbd_cube_pick task.
The cube is a VBD (Vertex Block Descent) deformable body — no rigid cube body.

Sequences cover all four scenario types:

  reachable   + success   (70% × 50% = 35%)
  reachable   + failure   (70% × 50% = 35%)
  unreachable + success   (30% × 50% = 15%)
  unreachable + failure   (30% × 50% = 15%)

Architecture
------------
Runs fully standalone — no Isaac Sim / AppLauncher required.
Uses the patched Newton package at /home/horde/projects/newton with:
  - create_soft_contacts_batched (int32-overflow-safe batch collision)
  - clamp_particle_inertia (particle velocity clamping in SolverVBD)

Two Newton models:
  single_model  — robot only (9 DOF). Used for batch IK.
  batched_model — N × robot + VBD soft grid particles.  Robot worlds are tiled
                  via begin_world/add_builder/end_world; particles added after,
                  tiled via numpy broadcast.

Two solvers per batched_model:
  rigid_solver  — SolverMuJoCo (robot + ground contacts)
  vbd_solver    — SolverVBD (deformable cube)
  collision     — CollisionPipeline (particle–rigid soft contacts)

Two-phase substep:
  1. clear_forces()
  2. CollisionPipeline.collide() → soft contacts
  3. apply_soft_body_reactions() → write reaction forces into state.body_f
  4. rigid_solver.step()          → robot reads body_f
  5. vbd_solver.step()            → cube particles

Usage
-----
micromamba run -n env_isaaclab python scripts/generate_sequences.py \\
    --num_sequences 100 \\
    --num-worlds 4 \\
    --output data/validation/vbd_sequences.json
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.ik as nik
from newton._src.sim.collide import CollisionPipeline
from newton._src.sim.graph_coloring import color_graph, color_rigid_bodies
from newton.solvers.vbd import SolverVBD

sys.path.insert(0, str(Path(__file__).parent))
from _common.sampling import sample_cube_pos, sample_label
from _common.sequence_schema import DEFAULT_CONFIG, label_description, save_sequences
from _common.waypoint_ik import WaypointStateMachine

# Add physics/ to path for vbd_coupling
sys.path.insert(0, str(Path(__file__).parent.parent))
from physics.vbd_coupling import apply_soft_body_reactions

_TASK_ROOT   = Path(__file__).parent.parent
_OUTPUTS_DIR = _TASK_ROOT / "data" / "validation"

# ---------------------------------------------------------------------------
# Panda URDF path
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
    raise FileNotFoundError("Panda URDF not found. Install Isaac Sim pip package.")


# ---------------------------------------------------------------------------
# Physics constants — match FrankaVbdCubePickNewtonCfg + manager
# ---------------------------------------------------------------------------

_PANDA_HAND_BODY_LABEL = "panda_hand"

# Robot PD gains (from example_ik_cube_stacking.py)
_ARM_KE          = [4500, 4500, 3500, 3500, 2000, 2000, 2000]
_ARM_KD          = [450,  450,  350,  350,  200,  200,  200]
_FINGER_KE       = [100,  100]
_FINGER_KD       = [10,   10]
_ARM_ARMATURE    = [0.30, 0.30, 0.30, 0.30, 0.11, 0.11, 0.11]
_FINGER_ARMATURE = [0.15, 0.15]
_ARM_EFFORT      = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
_FINGER_EFFORT   = [100.0, 100.0]

_CONTACT_KE = 5.0e4
_CONTACT_KD = 5.0e2
_CONTACT_KF = 1.0e3
_CONTACT_MU = 0.75

_N_ROBOT_JOINTS = 9
_N_SUBSTEPS     = 10
_SUB_DT         = 0.002   # 2 ms substep → 20 ms / frame (50 Hz)

_HOME_JOINT_Q = [
    -3.6802115e-03,
     2.3901723e-02,
     3.6804110e-03,
    -2.3683236e+00,
    -1.2918962e-04,
     2.3922248e+00,
     7.8549200e-01,
     0.04,
     0.04,
]

# VBD cube material — matches FrankaVbdCubePickNewtonCfg
_CUBE_SIZE       = 0.05      # m
_CUBE_HALF       = _CUBE_SIZE / 2.0
_CUBE_RESOLUTION = 3         # (R+1)³ = 64 particles, 5R³ = 135 tets
_CUBE_CELL       = _CUBE_SIZE / _CUBE_RESOLUTION
_CUBE_DENSITY    = 400.0     # kg/m³
_YOUNG_MODULUS   = 2e4       # Pa
_POISSON_RATIO   = 0.4
_K_MU            = _YOUNG_MODULUS / (2.0 * (1.0 + _POISSON_RATIO))
_K_LAMBDA        = (_YOUNG_MODULUS * _POISSON_RATIO
                    / ((1.0 + _POISSON_RATIO) * (1.0 - 2.0 * _POISSON_RATIO)))
_K_DAMP          = 1e-5

_PARTICLE_RADIUS     = 0.015   # m
_SOFT_CONTACT_KE     = 1e4
_SOFT_CONTACT_KD     = 1e-5
_SOFT_CONTACT_MU     = 1.5
_VBD_ITERATIONS      = 20
_MAX_CONTACTS_PER_ENV = 200
_MAX_PARTICLE_VEL    = 10.0   # m/s (clamps inertia displacement)


# ---------------------------------------------------------------------------
# Robot builder (shared with rigid task — identical)
# ---------------------------------------------------------------------------

def build_robot_builder(urdf_path: Path) -> tuple:
    """Build unfinalized robot-only ModelBuilder for N-world tiling."""
    mb = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(mb)
    mb.add_urdf(str(urdf_path), floating=False, enable_self_collisions=False,
                parse_visuals_as_colliders=False)

    hand_local_idx    = next(i for i, k in enumerate(mb.body_label) if _PANDA_HAND_BODY_LABEL in k)
    left_finger_body  = next(i for i, k in enumerate(mb.body_label) if "leftfinger"  in k)
    right_finger_body = next(i for i, k in enumerate(mb.body_label) if "rightfinger" in k)

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

    for i in range(len(mb.shape_body)):
        if mb.shape_body[i] >= 0:
            mb.shape_collision_group[i] = 2

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

    all_ke     = _ARM_KE + _FINGER_KE
    all_kd     = _ARM_KD + _FINGER_KD
    all_arm    = _ARM_ARMATURE + _FINGER_ARMATURE
    all_effort = _ARM_EFFORT + _FINGER_EFFORT
    for i in range(_N_ROBOT_JOINTS):
        mb.joint_target_ke[i]    = float(all_ke[i])
        mb.joint_target_kd[i]    = float(all_kd[i])
        mb.joint_armature[i]     = float(all_arm[i])
        mb.joint_effort_limit[i] = float(all_effort[i])

    for i in range(_N_ROBOT_JOINTS):
        mb.joint_q[i]          = float(_HOME_JOINT_Q[i])
        mb.joint_target_pos[i] = float(_HOME_JOINT_Q[i])

    robot_body_count = mb.body_count
    return mb, hand_local_idx, robot_body_count


# ---------------------------------------------------------------------------
# VBD batched model builder
# ---------------------------------------------------------------------------

def build_vbd_batched_model(robot_builder: newton.ModelBuilder, num_worlds: int) -> tuple:
    """Build N worlds (robot rigid bodies) + VBD soft grid particles.

    Returns:
        model:             finalized Newton Model.
        rigid_solver:      SolverMuJoCo for robot + ground contacts.
        vbd_solver:        SolverVBD for deformable cube.
        collision:         CollisionPipeline for particle–shape contacts.
        soft_contacts:     Contacts object from CollisionPipeline.
        particles_per_env: Number of particles per world.
        particle_rest_q:   numpy (PPE, 3) relative rest positions for reset.
    """
    scene = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)

    # Phase 1: add N robot worlds
    for w in range(num_worlds):
        scene.begin_world()
        scene.add_builder(robot_builder)
        scene.end_world()
    scene.add_ground_plane()

    # Phase 2: add VBD soft grid for env-0, tile to envs 1..N-1
    # Place env-0 cube at a default spawn position; actual positions set at reset.
    default_cube_pos = np.array([0.4, 0.0, _CUBE_HALF], dtype=np.float32)
    grid_origin = default_cube_pos - np.array([_CUBE_HALF, _CUBE_HALF, 0.0])

    scene.default_particle_radius = _PARTICLE_RADIUS
    snap_particle = len(scene.particle_q)
    snap_tet      = len(scene.tet_indices)
    snap_tet_pose = len(scene.tet_poses)
    snap_tet_mat  = len(scene.tet_materials)
    snap_tet_act  = len(scene.tet_activations)

    scene.add_soft_grid(
        pos=wp.vec3(float(grid_origin[0]), float(grid_origin[1]), float(grid_origin[2])),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=_CUBE_RESOLUTION,
        dim_y=_CUBE_RESOLUTION,
        dim_z=_CUBE_RESOLUTION,
        cell_x=_CUBE_CELL, cell_y=_CUBE_CELL, cell_z=_CUBE_CELL,
        density=_CUBE_DENSITY,
        k_mu=_K_MU,
        k_lambda=_K_LAMBDA,
        k_damp=_K_DAMP,
    )

    n_particles = len(scene.particle_q) - snap_particle
    pq_env0     = np.array([[r[0], r[1], r[2]] for r in scene.particle_q[snap_particle:]], dtype=np.float32)
    pqd_env0    = np.array([[r[0], r[1], r[2]] for r in scene.particle_qd[snap_particle:]], dtype=np.float32)
    pm_env0     = list(scene.particle_mass[snap_particle:])
    pr_env0     = list(scene.particle_radius[snap_particle:])
    pf_env0     = list(scene.particle_flags[snap_particle:])

    tets_env0      = np.array(scene.tet_indices[snap_tet:], dtype=np.int32).reshape(-1, 4)
    tet_poses_env0 = list(scene.tet_poses[snap_tet_pose:])
    tet_mats_env0  = list(scene.tet_materials[snap_tet_mat:])
    tet_acts_env0  = list(scene.tet_activations[snap_tet_act:])

    # Set particle_world for env-0 (world 0)
    while len(scene.particle_world) < snap_particle + n_particles:
        scene.particle_world.append(0)
    for p in range(n_particles):
        scene.particle_world[snap_particle + p] = 0

    # Tile envs 1..N-1
    env_offsets_world = np.arange(1, num_worlds, dtype=np.int32)  # world indices 1..N-1
    for w_idx, w in enumerate(range(1, num_worlds)):
        # Particle positions for env w — same world-relative positions (all worlds at origin)
        # Each world in Newton is independent, so same absolute positions work.
        extra_pq = pq_env0.copy()
        for pi in range(n_particles):
            scene.particle_q.append(wp.vec3(float(extra_pq[pi, 0]),
                                             float(extra_pq[pi, 1]),
                                             float(extra_pq[pi, 2])))
            scene.particle_qd.append(wp.vec3(0.0, 0.0, 0.0))
        scene.particle_mass.extend(pm_env0)
        scene.particle_radius.extend(pr_env0)
        scene.particle_flags.extend(pf_env0)
        while len(scene.particle_world) < snap_particle + (w + 1) * n_particles:
            scene.particle_world.append(w)
        for p in range(n_particles):
            scene.particle_world[snap_particle + w * n_particles + p] = w

        # Tetrahedra with particle index offset
        env_tet_offset = snap_particle + w * n_particles
        tets_w = tets_env0 + env_tet_offset  # offset all tet vertex indices
        scene.tet_indices.extend(map(tuple, tets_w.tolist()))
        scene.tet_poses.extend(tet_poses_env0)
        scene.tet_materials.extend(tet_mats_env0)
        scene.tet_activations.extend(tet_acts_env0)

    # Graph coloring: build tet edges for env-0 (local 0-based), color, tile
    edge_set = set()
    for row in tets_env0:
        for a, b in itertools.combinations(row.tolist(), 2):
            edge_set.add((min(a, b), max(a, b)))
    edge_np = np.array(sorted(edge_set), dtype=np.int32)
    edge_wp = wp.array(edge_np, dtype=int, device="cpu")

    single_colors = color_graph(n_particles, edge_wp, None)
    tiled_colors = [
        np.concatenate([
            c + snap_particle + w * n_particles for w in range(num_worlds)
        ])
        for c in single_colors
    ]
    scene.set_coloring(tiled_colors)
    scene.body_color_groups = color_rigid_bodies(
        scene.body_count, scene.joint_parent, scene.joint_child
    )

    # Finalize
    model = scene.finalize()

    shapes_per_env = model.shape_count // num_worlds
    practical_max  = _MAX_CONTACTS_PER_ENV * num_worlds

    # Rigid solver (robot + ground)
    rigid_solver = newton.solvers.SolverMuJoCo(
        model,
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

    # VBD solver (soft cube)
    vbd_solver = SolverVBD(
        model,
        iterations=_VBD_ITERATIONS,
        integrate_with_external_rigid_solver=True,
        max_soft_contacts=practical_max,
        particle_max_velocity=_MAX_PARTICLE_VEL,
    )

    # Collision pipeline (particle–shape soft contacts)
    collision = CollisionPipeline(
        model,
        soft_contact_margin=_PARTICLE_RADIUS * 3.0,
        soft_contact_max=practical_max,
        particles_per_world=n_particles,
        shapes_per_world=shapes_per_env,
    )
    soft_contacts = collision.contacts()
    # Warm-up: first collide() does d2h shape-type copies (illegal inside CUDA graph)
    state_warmup = model.state()
    collision.collide(state_warmup, soft_contacts)

    # Rest positions relative to CoM (for particle reset)
    com = pq_env0.mean(axis=0)
    particle_rest_q = pq_env0 - com   # (PPE, 3) shape-relative

    print(
        f"[vbd_model] {num_worlds} worlds | "
        f"{model.body_count} bodies ({model.body_count // num_worlds}/world) | "
        f"{model.particle_count} particles ({n_particles}/world) | "
        f"{len(tets_env0)} tets/world | "
        f"{shapes_per_env} shapes/world | "
        f"practical_max={practical_max}"
    )

    return model, rigid_solver, vbd_solver, collision, soft_contacts, n_particles, particle_rest_q


# ---------------------------------------------------------------------------
# Batch IK (identical to rigid task)
# ---------------------------------------------------------------------------

def build_batch_ik(single_model, hand_local_idx: int, num_worlds: int) -> tuple:
    """Create persistent batch IK solvers (position + rotation, two stages)."""
    ik_dofs = single_model.joint_coord_count

    dummy_pos = wp.zeros(num_worlds, dtype=wp.vec3)
    dummy_rot = wp.array([wp.vec4(0.0, 0.0, 0.0, 1.0)] * num_worlds, dtype=wp.vec4)

    joint_limit_lower = single_model.joint_limit_lower
    joint_limit_upper = single_model.joint_limit_upper

    pos_obj_1   = nik.IKObjectivePosition(
        link_index=hand_local_idx, link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=dummy_pos,
    )
    limit_obj_1 = nik.IKObjectiveJointLimit(
        joint_limit_lower=joint_limit_lower, joint_limit_upper=joint_limit_upper, weight=10.0,
    )
    stage1_solver = nik.IKSolver(
        model=single_model, n_problems=num_worlds,
        objectives=[pos_obj_1, limit_obj_1],
        jacobian_mode=nik.IKJacobianType.ANALYTIC,
    )

    pos_obj_2   = nik.IKObjectivePosition(
        link_index=hand_local_idx, link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=dummy_pos,
    )
    limit_obj_2 = nik.IKObjectiveJointLimit(
        joint_limit_lower=joint_limit_lower, joint_limit_upper=joint_limit_upper, weight=10.0,
    )
    rot_obj_2   = nik.IKObjectiveRotation(
        link_index=hand_local_idx, link_offset_rotation=wp.quat_identity(),
        target_rotations=dummy_rot, weight=0.3,
    )
    stage2_solver = nik.IKSolver(
        model=single_model, n_problems=num_worlds,
        objectives=[pos_obj_2, limit_obj_2, rot_obj_2],
        jacobian_mode=nik.IKJacobianType.AUTODIFF,
    )

    home_q = single_model.joint_q.numpy()[:_N_ROBOT_JOINTS].astype(np.float32)
    joint_q_ik = wp.array(
        np.tile(home_q, (num_worlds, 1)).astype(np.float32), dtype=wp.float32,
    )
    return joint_q_ik, pos_obj_1, stage1_solver, pos_obj_2, rot_obj_2, stage2_solver


# ---------------------------------------------------------------------------
# State reset
# ---------------------------------------------------------------------------

def reset_batch_state(
    model,
    state_0,
    state_1,
    cube_positions_np: np.ndarray,
    default_robot_q: np.ndarray,
    joint_q_ik,
    num_worlds: int,
    n_particles: int,
    particle_rest_q: np.ndarray,
) -> None:
    """Reset all worlds: robot → home, cubes → sampled positions, particles → new cube CoM."""
    n_coord = model.joint_coord_count // num_worlds
    n_dof   = model.joint_dof_count   // num_worlds

    joint_q  = np.zeros(model.joint_coord_count, dtype=np.float32)
    joint_qd = np.zeros(model.joint_dof_count,   dtype=np.float32)

    for w in range(num_worlds):
        base = w * n_coord
        joint_q[base : base + _N_ROBOT_JOINTS] = default_robot_q

    state_0.joint_q.assign(joint_q)
    state_0.joint_qd.assign(joint_qd)
    state_1.joint_q.assign(joint_q)
    state_1.joint_qd.assign(joint_qd)
    joint_q_ik.assign(np.tile(default_robot_q, (num_worlds, 1)).astype(np.float32))

    # Particle positions: particle_rest_q (relative) + new CoM per env
    total_particles = num_worlds * n_particles
    pq_new  = np.zeros((total_particles, 3), dtype=np.float32)
    pqd_new = np.zeros((total_particles, 3), dtype=np.float32)
    for w in range(num_worlds):
        new_com = cube_positions_np[w]  # [x, y, z] with z = cube_half
        start = w * n_particles
        pq_new[start : start + n_particles] = particle_rest_q + new_com
    state_0.particle_q.assign([wp.vec3(float(r[0]), float(r[1]), float(r[2])) for r in pq_new])
    state_0.particle_qd.assign([wp.vec3(0.0, 0.0, 0.0)] * total_particles)
    state_1.particle_q.assign(state_0.particle_q)
    state_1.particle_qd.assign(state_0.particle_qd)


# ---------------------------------------------------------------------------
# Particle CoM extraction
# ---------------------------------------------------------------------------

def get_cube_coms_np(particle_q_np: np.ndarray, num_worlds: int, n_particles: int) -> np.ndarray:
    """Compute CoM position per world from flat particle array.

    Args:
        particle_q_np: (num_worlds * n_particles, 3) numpy array.
        num_worlds:    Number of worlds.
        n_particles:   Particles per world.

    Returns:
        (num_worlds, 3) numpy array of CoM positions.
    """
    pq = particle_q_np.reshape(num_worlds, n_particles, 3)
    return pq.mean(axis=1)   # (num_worlds, 3)


# ---------------------------------------------------------------------------
# Two-phase simulation step
# ---------------------------------------------------------------------------

def simulate_two_phase(
    state_0, state_1,
    rigid_solver, vbd_solver, collision, soft_contacts,
    rigid_contacts, control,
    sub_dt: float, n_substeps: int,
):
    """Run n_substeps of two-phase (rigid + VBD) physics.

    n_substeps MUST be even — with an even count, state_0 holds the current
    result after the loop (matching CUDA graph correctness convention).
    """
    for _ in range(n_substeps):
        state_0.clear_forces()
        # 1. Soft-rigid contact detection
        collision.collide(state_0, soft_contacts)
        # 2. Inject reaction forces into body_f (same substep, two-way coupling)
        apply_soft_body_reactions(
            soft_contacts, state_0, rigid_solver.model,
            soft_contacts.soft_contact_max,
            particle_q_prev=state_1.particle_q,
            friction_epsilon=vbd_solver.friction_epsilon if hasattr(vbd_solver, "friction_epsilon") else 1e-2,
            dt=sub_dt,
        )
        # 3. Rigid solver (reads body_f)
        rigid_contacts_step = rigid_solver.model.collide(state_0)
        rigid_solver.step(state_0, state_1, control, rigid_contacts_step, sub_dt)
        # 4. VBD solver (deformable cube)
        vbd_solver.step(state_0, state_1, control, soft_contacts, sub_dt)
        state_0, state_1 = state_1, state_0
    return state_0, state_1


# ---------------------------------------------------------------------------
# Batch episode runner
# ---------------------------------------------------------------------------

def run_batch_episode(
    model,
    rigid_solver, vbd_solver, collision, soft_contacts,
    single_model,
    hand_local_idx: int,
    num_worlds: int,
    state_0, state_1, control,
    joint_q_ik,
    pos_obj_1, stage1_solver,
    pos_obj_2, rot_obj_2, stage2_solver,
    state_machines: list,
    cube_init_positions_np: np.ndarray,
    seq_ids: list,
    dt: float,
    steps_per_ep: int,
    record_every: int,
    n_particles: int,
    particle_rest_q: np.ndarray,
) -> list:
    """Run one batch of num_worlds episodes simultaneously. Returns frames_per_world."""
    n_coord_per_world    = model.joint_coord_count // num_worlds
    n_ctrl_per_world     = len(control.joint_target_pos) // num_worlds
    num_bodies_per_world = model.body_count // num_worlds

    sub_dt = dt / _N_SUBSTEPS

    # Reset
    default_robot_q = single_model.joint_q.numpy()[:_N_ROBOT_JOINTS].astype(np.float32)
    reset_batch_state(
        model, state_0, state_1,
        cube_init_positions_np, default_robot_q, joint_q_ik,
        num_worlds, n_particles, particle_rest_q,
    )
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    gripper_closed   = [False] * num_worlds
    prev_robot_q     = np.tile(default_robot_q, (num_worlds, 1))
    frames_per_world = [[] for _ in range(num_worlds)]

    for step in range(steps_per_ep):
        t = step * dt

        # 1. N state machines → EE targets + finger commands
        ee_targets:  list[list[float]] = []
        quats_wxyz:  list[list[float]] = []
        finger_cmds: np.ndarray = np.zeros(num_worlds, dtype=np.float32)

        for w in range(num_worlds):
            ee_pos_t, quat, finger_cmd = state_machines[w].get_target(t)
            ee_targets.append(ee_pos_t.tolist())
            quats_wxyz.append(quat.tolist())
            finger_cmds[w] = float(finger_cmd)
            if float(finger_cmd) < 0.01 and not gripper_closed[w]:
                gripper_closed[w] = True

        # 2. Batch IK
        pos_arr = wp.array([wp.vec3(*t_) for t_ in ee_targets], dtype=wp.vec3)
        pos_obj_1.target_positions = pos_arr
        stage1_solver.step(joint_q_ik, joint_q_ik, iterations=60)

        rot_arr = wp.array(
            [wp.vec4(q[1], q[2], q[3], q[0]) for q in quats_wxyz], dtype=wp.vec4
        )
        pos_obj_2.target_positions = pos_arr
        rot_obj_2.target_rotations = rot_arr
        stage2_solver.step(joint_q_ik, joint_q_ik, iterations=100)

        # 3. Set joint targets
        joint_q_ik_np = joint_q_ik.numpy()
        ctrl_np = np.zeros((num_worlds, n_ctrl_per_world), dtype=np.float32)
        ctrl_np[:, :_N_ROBOT_JOINTS] = joint_q_ik_np
        ctrl_np[:, 7] = finger_cmds
        ctrl_np[:, 8] = finger_cmds
        control.joint_target_pos.assign(ctrl_np.reshape(-1))

        # 4. Two-phase physics (eager — no CUDA graph for VBD validation)
        state_0, state_1 = simulate_two_phase(
            state_0, state_1,
            rigid_solver, vbd_solver, collision, soft_contacts,
            None, control, sub_dt, _N_SUBSTEPS,
        )

        # 5. Read state
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        body_q_np    = state_0.body_q.numpy()
        joint_q_np   = state_0.joint_q.numpy()
        particle_q_np = state_0.particle_q.numpy()  # (N*PPE,) of wp.vec3 → numpy via reshape
        # particle_q is wp.array of wp.vec3; .numpy() returns (N*PPE, 3) on modern warp
        if particle_q_np.ndim == 1:
            particle_q_np = particle_q_np.reshape(-1, 3)

        cube_coms = get_cube_coms_np(particle_q_np, num_worlds, n_particles)

        # 6. Record
        for w in range(num_worlds):
            global_hand = w * num_bodies_per_world + hand_local_idx
            ee_pos_now  = body_q_np[global_hand][:3].tolist()
            cube_pos_now = cube_coms[w].tolist()

            base = w * n_coord_per_world
            robot_q_now = joint_q_np[base : base + _N_ROBOT_JOINTS].astype(np.float32)
            joint_vel_w = ((robot_q_now - prev_robot_q[w]) / dt).tolist()

            if step % record_every == 0:
                frames_per_world[w].append({
                    "step":           step,
                    "t":              round(t, 4),
                    "joint_pos_cmd":  ctrl_np[w, :_N_ROBOT_JOINTS].tolist(),
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
    steps_per_ep = int(T_END / dt)
    RECORD_EVERY = 1

    num_worlds = args.num_worlds

    urdf_path = _find_panda_urdf()
    print(f"[vbd_gen] Panda URDF: {urdf_path}")
    print(f"[vbd_gen] Building models (num_worlds={num_worlds}) ...")

    robot_mb, hand_local_idx, robot_body_count = build_robot_builder(urdf_path)
    (model, rigid_solver, vbd_solver, collision, soft_contacts,
     n_particles, particle_rest_q) = build_vbd_batched_model(robot_mb, num_worlds)
    single_model = robot_mb.finalize()

    (joint_q_ik, pos_obj_1, stage1_solver,
     pos_obj_2, rot_obj_2, stage2_solver) = build_batch_ik(
        single_model, hand_local_idx, num_worlds
    )

    state_0  = model.state()
    state_1  = model.state()
    control  = model.control()

    total_seqs = args.num_sequences
    print(
        f"[vbd_gen] Simulating {total_seqs} sequences in batches of {num_worlds} | "
        f"{steps_per_ep} steps × {dt:.3f}s = {T_END:.1f}s"
    )

    all_sequences: list[dict] = []
    seq_counter = 0

    while seq_counter < total_seqs:
        batch_size = min(num_worlds, total_seqs - seq_counter)

        batch_labels   = [sample_label(args.reachable_ratio, args.success_ratio)
                          for _ in range(batch_size)]
        batch_cube_pos = [sample_cube_pos(lbl, cfg, torch.device("cpu"))
                          for lbl in batch_labels]
        batch_seq_ids  = [
            f"seq_{args.seq_id_start + seq_counter + i:04d}"
            for i in range(batch_size)
        ]

        while len(batch_labels) < num_worlds:
            batch_labels.append(batch_labels[-1])
            batch_cube_pos.append(batch_cube_pos[-1])
            batch_seq_ids.append("seq_pad")

        cube_init_np = np.array(
            [pos.numpy().astype(np.float32) for pos in batch_cube_pos],
            dtype=np.float32,
        )
        state_machines = [
            WaypointStateMachine(batch_cube_pos[w], batch_labels[w], cfg, torch.device("cpu"))
            for w in range(num_worlds)
        ]

        lbl_strs = [label_description(batch_labels[w]) for w in range(batch_size)]
        batch_num = seq_counter // num_worlds + 1
        total_batches = (total_seqs + num_worlds - 1) // num_worlds
        print(
            f"[vbd_gen] batch {batch_num}/{total_batches} "
            f"({batch_size} active): "
            + " | ".join(f"{batch_seq_ids[w]}({lbl_strs[w]})" for w in range(min(batch_size, 4)))
            + ("..." if batch_size > 4 else ""),
            end="", flush=True,
        )

        frames_per_world = run_batch_episode(
            model=model,
            rigid_solver=rigid_solver, vbd_solver=vbd_solver,
            collision=collision, soft_contacts=soft_contacts,
            single_model=single_model,
            hand_local_idx=hand_local_idx,
            num_worlds=num_worlds,
            state_0=state_0, state_1=state_1, control=control,
            joint_q_ik=joint_q_ik,
            pos_obj_1=pos_obj_1, stage1_solver=stage1_solver,
            pos_obj_2=pos_obj_2, rot_obj_2=rot_obj_2, stage2_solver=stage2_solver,
            state_machines=state_machines,
            cube_init_positions_np=cube_init_np,
            seq_ids=batch_seq_ids,
            dt=dt, steps_per_ep=steps_per_ep, record_every=RECORD_EVERY,
            n_particles=n_particles, particle_rest_q=particle_rest_q,
        )

        for w in range(batch_size):
            cube_zs = [fr["cube_pos_w"][2] for fr in frames_per_world[w]]
            peak_z  = max(cube_zs) if cube_zs else 0.0
            cleared = "LIFTED" if peak_z >= cfg["lift_height"] else f"peak_z={peak_z:.3f}"
            has_nan = any(
                not all(np.isfinite(v) for v in fr["cube_pos_w"])
                for fr in frames_per_world[w]
            )
            status = "NaN!" if has_nan else cleared
            print(f"\n  [{batch_seq_ids[w]}] frames={len(frames_per_world[w])}  {status}", end="")
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
        description="Generate Franka VBD cube pick validation sequences."
    )
    parser.add_argument("--num_sequences",   type=int,   default=100)
    parser.add_argument("--num-worlds",      type=int,   default=4,
                        help="Worlds per batch. Default 4 (VBD is more compute-heavy than rigid).")
    parser.add_argument("--output",          type=str,
                        default=str(_OUTPUTS_DIR / "vbd_sequences.json"))
    parser.add_argument("--reachable_ratio", type=float, default=0.7)
    parser.add_argument("--success_ratio",   type=float, default=0.5)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--seq_id_start",    type=int,   default=0)
    args = parser.parse_args()
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_sequences(args)


if __name__ == "__main__":
    main()
