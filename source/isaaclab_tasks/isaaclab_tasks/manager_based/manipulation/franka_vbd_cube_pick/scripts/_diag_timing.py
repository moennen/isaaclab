#!/usr/bin/env python3
"""Time the per-step overhead in generate_sequences.py to find the bottleneck."""

import sys
import time
from pathlib import Path
import numpy as np
import warp as wp
import newton

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_sequences import (
    _find_panda_urdf, _HOME_JOINT_Q, _N_ROBOT_JOINTS, _N_SUBSTEPS, _SUB_DT,
    _CUBE_HALF, _PARTICLE_RADIUS, _CUBE_REST_Z,
    build_robot_builder, build_vbd_batched_model, reset_batch_state,
    build_batch_ik, simulate_two_phase,
)

NUM_WORLDS = 16
dt = 0.020
sub_dt = dt / _N_SUBSTEPS

CUBE_INIT = np.array([
    [0.4 + 0.1*i, 0.0, _CUBE_REST_Z] for i in range(NUM_WORLDS)
], dtype=np.float32)


def main():
    urdf = _find_panda_urdf()
    robot_mb, hand_local_idx, _ = build_robot_builder(urdf)
    (model, rigid_solver, vbd_solver, collision, soft_contacts,
     n_particles, particle_rest_q) = build_vbd_batched_model(robot_mb, NUM_WORLDS)
    robot_mb_ik, _, _ = build_robot_builder(urdf)
    single_model = robot_mb_ik.finalize()

    (joint_q_ik, pos_obj_1, stage1_solver,
     pos_obj_2, rot_obj_2, stage2_solver) = build_batch_ik(
        single_model, hand_local_idx, NUM_WORLDS
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    default_q = np.array(_HOME_JOINT_Q, dtype=np.float32)
    reset_batch_state(
        model, state_0, state_1,
        CUBE_INIT, default_q, joint_q_ik,
        NUM_WORLDS, n_particles, particle_rest_q,
    )
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    # Warm-up: run a few steps to get past JIT compilation
    print("Warming up...")
    ee_target = [np.array([0.5, 0.0, 0.5], dtype=np.float32)] * NUM_WORLDS
    quat_wxyz  = [[1.0, 0.0, 0.0, 0.0]] * NUM_WORLDS
    pos_arr = wp.array([wp.vec3(*t) for t in ee_target], dtype=wp.vec3)
    rot_arr = wp.array([wp.vec4(q[1], q[2], q[3], q[0]) for q in quat_wxyz], dtype=wp.vec4)

    for _ in range(3):
        pos_obj_1.target_positions = pos_arr
        stage1_solver.step(joint_q_ik, joint_q_ik, iterations=60)
        pos_obj_2.target_positions = pos_arr
        rot_obj_2.target_rotations = rot_arr
        stage2_solver.step(joint_q_ik, joint_q_ik, iterations=100)
        iq_np = joint_q_ik.numpy()
        n_ctrl = len(control.joint_target_pos) // NUM_WORLDS
        ctrl_np = np.zeros((NUM_WORLDS, n_ctrl), dtype=np.float32)
        ctrl_np[:, :_N_ROBOT_JOINTS] = iq_np
        control.joint_target_pos.assign(ctrl_np.reshape(-1))
        state_0, state_1 = simulate_two_phase(
            state_0, state_1, rigid_solver, vbd_solver, collision, soft_contacts,
            None, control, sub_dt, _N_SUBSTEPS, model=model,
        )
    wp.synchronize()
    print("Warm-up done.")

    # Time individual components
    N_BENCH = 20
    print(f"\nTiming {N_BENCH} steps with {NUM_WORLDS} worlds...")

    t0 = time.perf_counter()
    for step in range(N_BENCH):
        pos_obj_1.target_positions = pos_arr
        stage1_solver.step(joint_q_ik, joint_q_ik, iterations=20)
    wp.synchronize()
    t_ik1 = (time.perf_counter() - t0) / N_BENCH * 1000
    print(f"  stage1 IK (20 iters):  {t_ik1:.1f} ms/step")

    t0 = time.perf_counter()
    for step in range(N_BENCH):
        pos_obj_2.target_positions = pos_arr
        rot_obj_2.target_rotations = rot_arr
        stage2_solver.step(joint_q_ik, joint_q_ik, iterations=30)
    wp.synchronize()
    t_ik2 = (time.perf_counter() - t0) / N_BENCH * 1000
    print(f"  stage2 IK (30 iters):  {t_ik2:.1f} ms/step")

    t0 = time.perf_counter()
    for step in range(N_BENCH):
        iq_np = joint_q_ik.numpy()
    t_np = (time.perf_counter() - t0) / N_BENCH * 1000
    print(f"  joint_q_ik.numpy():    {t_np:.1f} ms/step")

    t0 = time.perf_counter()
    for step in range(N_BENCH):
        n_ctrl = len(control.joint_target_pos) // NUM_WORLDS
        ctrl_np = np.zeros((NUM_WORLDS, n_ctrl), dtype=np.float32)
        ctrl_np[:, :_N_ROBOT_JOINTS] = iq_np
        control.joint_target_pos.assign(ctrl_np.reshape(-1))
    t_ctrl = (time.perf_counter() - t0) / N_BENCH * 1000
    print(f"  set control targets:   {t_ctrl:.1f} ms/step")

    t0 = time.perf_counter()
    for step in range(N_BENCH):
        state_0, state_1 = simulate_two_phase(
            state_0, state_1, rigid_solver, vbd_solver, collision, soft_contacts,
            None, control, sub_dt, _N_SUBSTEPS, model=model,
        )
    wp.synchronize()
    t_sim = (time.perf_counter() - t0) / N_BENCH * 1000
    print(f"  simulate_two_phase:    {t_sim:.1f} ms/step  ({_N_SUBSTEPS} substeps)")
    print(f"  TOTAL per step:        {t_ik1+t_ik2+t_np+t_ctrl+t_sim:.1f} ms")
    n_batches_100 = (100 + NUM_WORLDS - 1) // NUM_WORLDS
    t_per_batch = (t_ik1+t_ik2+t_np+t_ctrl+t_sim)*500/1000
    print(f"  Estimated for 500 steps × {NUM_WORLDS} worlds: "
          f"{t_per_batch:.0f} s/batch → {t_per_batch*n_batches_100/60:.0f} min for 100 seqs")


if __name__ == "__main__":
    main()
