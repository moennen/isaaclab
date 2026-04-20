#!/usr/bin/env python3
"""Diagnostic: run one frame one substep at a time, checking for NaN/divergence."""

import sys
from pathlib import Path
import numpy as np
import warp as wp
import newton

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_sequences import (
    _find_panda_urdf, _HOME_JOINT_Q, _N_ROBOT_JOINTS, _N_SUBSTEPS, _SUB_DT,
    _CUBE_HALF, _PARTICLE_RADIUS,
    build_robot_builder, build_vbd_batched_model, reset_batch_state,
    simulate_two_phase, get_cube_coms_np,
)

NUM_WORLDS = 2
# COM z = _CUBE_HALF + _PARTICLE_RADIUS so bottom particles sit AT z=radius (not inside ground)
_CUBE_REST_Z = _CUBE_HALF + _PARTICLE_RADIUS
CUBE_INIT = np.array([[0.4, 0.0, _CUBE_REST_Z], [0.4, 0.1, _CUBE_REST_Z]], dtype=np.float32)

def main():
    urdf = _find_panda_urdf()
    print(f"[diag] URDF: {urdf}")

    robot_mb, hand_local_idx, _ = build_robot_builder(urdf)
    (model, rigid_solver, vbd_solver, collision, soft_contacts,
     n_particles, particle_rest_q) = build_vbd_batched_model(robot_mb, NUM_WORLDS)
    print(f"[diag] n_particles/world={n_particles}")
    print(f"[diag] model.gravity shape={model.gravity.shape}, values={model.gravity.numpy()}")
    print(f"[diag] model.tri_count={model.tri_count}")
    print(f"[diag] model.particle_count={model.particle_count}")
    pw = model.particle_world.numpy()
    print(f"[diag] particle_world unique values={np.unique(pw)}  shape={pw.shape}")

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    default_q = np.array(_HOME_JOINT_Q, dtype=np.float32)
    joint_q_ik = wp.array(np.tile(default_q, (NUM_WORLDS, 1)).astype(np.float32), dtype=wp.float32)
    reset_batch_state(
        model, state_0, state_1,
        CUBE_INIT, default_q, joint_q_ik,
        NUM_WORLDS, n_particles, particle_rest_q,
    )
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    # Set control targets to home
    n_ctrl = len(control.joint_target_pos) // NUM_WORLDS
    ctrl_np = np.zeros((NUM_WORLDS, n_ctrl), dtype=np.float32)
    ctrl_np[:, :_N_ROBOT_JOINTS] = default_q
    control.joint_target_pos.assign(ctrl_np.reshape(-1))

    print(f"\n[diag] Running {_N_SUBSTEPS} substeps one at a time ...")
    for substep in range(_N_SUBSTEPS):
        state_0, state_1 = simulate_two_phase(
            state_0, state_1,
            rigid_solver, vbd_solver, collision, soft_contacts,
            None, control, _SUB_DT, 1,
            model=model,
        )
        try:
            wp.synchronize()
        except Exception as e:
            print(f"  Substep {substep+1}: CUDA ERROR during sync: {e}")
            break

        pq_np = state_0.particle_q.numpy()
        if pq_np.ndim == 1:
            pq_np = pq_np.reshape(-1, 3)

        has_nan  = bool(np.any(np.isnan(pq_np)) or np.any(np.isinf(pq_np)))
        max_pos  = float(np.abs(pq_np).max())
        coms     = get_cube_coms_np(pq_np, NUM_WORLDS, n_particles)
        print(f"  Substep {substep+1:2d}: has_nan={has_nan}  max_pos={max_pos:.4f}  "
              f"CoM[0]={coms[0].tolist()}  CoM[1]={coms[1].tolist()}")
        if has_nan or max_pos > 100:
            print("  [diag] DIVERGENCE DETECTED — stopping")
            break

    print("\n[diag] Now running a full 10-substep frame ...")
    reset_batch_state(
        model, state_0, state_1,
        CUBE_INIT, default_q, joint_q_ik,
        NUM_WORLDS, n_particles, particle_rest_q,
    )
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control.joint_target_pos.assign(ctrl_np.reshape(-1))

    try:
        state_0, state_1 = simulate_two_phase(
            state_0, state_1,
            rigid_solver, vbd_solver, collision, soft_contacts,
            None, control, _SUB_DT, _N_SUBSTEPS,
            model=model,
        )
        wp.synchronize()
        pq_np = state_0.particle_q.numpy()
        if pq_np.ndim == 1:
            pq_np = pq_np.reshape(-1, 3)
        coms = get_cube_coms_np(pq_np, NUM_WORLDS, n_particles)
        print(f"  Full frame OK!  CoM[0]={coms[0].tolist()}")
    except Exception as e:
        print(f"  Full frame FAILED: {e}")

    print("\n[diag] Done.")

if __name__ == "__main__":
    main()
