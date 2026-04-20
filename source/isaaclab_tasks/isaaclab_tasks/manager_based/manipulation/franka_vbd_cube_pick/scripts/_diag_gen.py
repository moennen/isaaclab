#!/usr/bin/env python3
"""Reproduce generate_sequences.py crash: pinpoint which operation breaks collision.collide."""

import sys
from pathlib import Path
import numpy as np
import warp as wp
import newton
import torch

wp.config.verify_cuda = True

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_sequences import (
    _find_panda_urdf, _HOME_JOINT_Q, _N_ROBOT_JOINTS, _N_SUBSTEPS, _SUB_DT,
    _CUBE_HALF, _PARTICLE_RADIUS, _CUBE_REST_Z,
    build_robot_builder, build_vbd_batched_model, reset_batch_state,
    build_batch_ik, simulate_two_phase,
)

NUM_WORLDS = 2
dt = 0.020
sub_dt = dt / _N_SUBSTEPS

CUBE_INIT = np.array(
    [[0.4, 0.0, _CUBE_REST_Z],
     [0.4, 0.1, _CUBE_REST_Z]], dtype=np.float32
)[:NUM_WORLDS]


def sync_check(label):
    try:
        dummy = wp.zeros(1, dtype=wp.float32, device="cuda:0")
        _ = dummy.numpy()
        print(f"  [ok] {label}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


def test_collide(model, collision, state_0, state_1, rigid_solver, vbd_solver,
                 control, soft_contacts, label):
    """Try collision.collide and report outcome."""
    print(f"\n  --- Collision test: {label} ---")
    try:
        state_0.clear_forces()
        collision.collide(model, state_0)
        pq = state_0.particle_q.numpy().reshape(-1, 3)
        cnt = int(collision.contacts.soft_contact_count.numpy()[0])
        print(f"  [PASS] contacts={cnt}, particle CoM={pq.mean(axis=0)}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def main():
    urdf = _find_panda_urdf()
    robot_mb, hand_local_idx, _ = build_robot_builder(urdf)
    (model, rigid_solver, vbd_solver, collision, soft_contacts,
     n_particles, particle_rest_q) = build_vbd_batched_model(robot_mb, NUM_WORLDS)
    # NOTE: intentionally NOT calling robot_mb.finalize() here to test if
    # that call corrupts model's GPU geometry (shape_source_ptr).
    # In generate_sequences.py, single_model = robot_mb.finalize() is called AFTER
    # build_vbd_batched_model. Testing if that's the bug.

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    default_q = np.array(_HOME_JOINT_Q, dtype=np.float32)

    reset_batch_state(
        model, state_0, state_1,
        CUBE_INIT, default_q, wp.array(
            np.tile(default_q, (NUM_WORLDS, 1)).astype(np.float32), dtype=wp.float32),
        NUM_WORLDS, n_particles, particle_rest_q,
    )
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    n_ctrl = len(control.joint_target_pos) // NUM_WORLDS
    ctrl_np = np.zeros((NUM_WORLDS, n_ctrl), dtype=np.float32)
    ctrl_np[:, :_N_ROBOT_JOINTS] = default_q
    control.joint_target_pos.assign(ctrl_np.reshape(-1))

    print("\n=== TEST A: collision.collide BEFORE building IK ===")
    if not test_collide(model, collision, state_0, state_1, rigid_solver, vbd_solver,
                        control, soft_contacts, "before IK build"):
        print("FAILED even before IK build — deeper issue")
        return

    # Build IK solvers (single_model.finalize already done above)
    print("\n=== TEST B: build IK solvers ===")
    (joint_q_ik, pos_obj_1, stage1_solver,
     pos_obj_2, rot_obj_2, stage2_solver) = build_batch_ik(
        single_model, hand_local_idx, NUM_WORLDS
    )
    sync_check("after build_batch_ik")

    if not test_collide(model, collision, state_0, state_1, rigid_solver, vbd_solver,
                        control, soft_contacts, "after IK build, before IK step"):
        print("build_batch_ik corrupts collision")
        return

    # Run stage1 IK
    print("\n=== TEST C: stage1 IK step ===")
    ee_target = [np.array([0.5, 0.0, 0.5], dtype=np.float32)] * NUM_WORLDS
    pos_arr = wp.array([wp.vec3(*t) for t in ee_target], dtype=wp.vec3)
    pos_obj_1.target_positions = pos_arr
    stage1_solver.step(joint_q_ik, joint_q_ik, iterations=60)
    sync_check("after stage1_solver.step")

    if not test_collide(model, collision, state_0, state_1, rigid_solver, vbd_solver,
                        control, soft_contacts, "after stage1 IK"):
        print("stage1 IK corrupts collision")
        return

    # Run stage2 IK
    print("\n=== TEST D: stage2 IK step ===")
    quat_wxyz = [[1.0, 0.0, 0.0, 0.0]] * NUM_WORLDS
    rot_arr = wp.array(
        [wp.vec4(q[1], q[2], q[3], q[0]) for q in quat_wxyz], dtype=wp.vec4
    )
    pos_obj_2.target_positions = pos_arr
    rot_obj_2.target_rotations = rot_arr
    stage2_solver.step(joint_q_ik, joint_q_ik, iterations=100)
    sync_check("after stage2_solver.step")

    if not test_collide(model, collision, state_0, state_1, rigid_solver, vbd_solver,
                        control, soft_contacts, "after stage2 IK"):
        print("stage2 IK corrupts collision")
        return

    # Transfer result to CPU
    print("\n=== TEST E: joint_q_ik.numpy() ===")
    joint_q_ik_np = joint_q_ik.numpy()
    print(f"  IK joint angles (world 0): {joint_q_ik_np[0, :7]}")
    sync_check("after joint_q_ik.numpy()")

    if not test_collide(model, collision, state_0, state_1, rigid_solver, vbd_solver,
                        control, soft_contacts, "after numpy transfer"):
        print("numpy transfer corrupts collision")
        return

    # Set control targets
    print("\n=== TEST F: set control targets ===")
    n_ctrl_per_world = len(control.joint_target_pos) // NUM_WORLDS
    ctrl_np2 = np.zeros((NUM_WORLDS, n_ctrl_per_world), dtype=np.float32)
    ctrl_np2[:, :_N_ROBOT_JOINTS] = joint_q_ik_np
    control.joint_target_pos.assign(ctrl_np2.reshape(-1))
    sync_check("after assign control targets")

    if not test_collide(model, collision, state_0, state_1, rigid_solver, vbd_solver,
                        control, soft_contacts, "after control targets assigned"):
        print("control target assignment corrupts collision")
        return

    print("\n=== ALL TESTS PASSED ===")
    print("The bug is elsewhere — perhaps collision.collide fails on 2nd+ call?")


if __name__ == "__main__":
    main()
