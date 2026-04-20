# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnostic: test cube settling consistency across batches with CUDA graph.

Checks whether cube_z at frame 0 is ~0.032m for all 8 worlds across 3
consecutive batches, using the exact same CUDA graph infrastructure as
generate_sequences.py and replay_sequences.py.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import warp as wp
import newton

sys.path.insert(0, str(Path(__file__).parent))
from generate_sequences import (
    _HOME_JOINT_Q,
    _N_SUBSTEPS,
    _SUB_DT,
    _CUBE_REST_Z,
    build_robot_builder,
    build_vbd_batched_model,
    get_cube_coms_np,
    reset_batch_state,
    simulate_two_phase,
)

NUM_WORLDS = 8

# Multiple distinct cube XY positions to simulate different batches
BATCH_CUBE_POS = [
    # Batch 0 (same as would be loaded from generation data)
    np.array([[0.50, 0.00, _CUBE_REST_Z],
              [0.50, 0.10, _CUBE_REST_Z],
              [0.45, 0.05, _CUBE_REST_Z],
              [0.50, -0.10, _CUBE_REST_Z],
              [0.55, 0.00, _CUBE_REST_Z],
              [0.50, 0.15, _CUBE_REST_Z],
              [0.48, -0.05, _CUBE_REST_Z],
              [0.52, 0.08, _CUBE_REST_Z]], dtype=np.float32),
    # Batch 1 (different cube positions)
    np.array([[0.48, 0.12, _CUBE_REST_Z],
              [0.53, -0.08, _CUBE_REST_Z],
              [0.47, 0.03, _CUBE_REST_Z],
              [0.51, 0.13, _CUBE_REST_Z],
              [0.49, -0.07, _CUBE_REST_Z],
              [0.54, 0.02, _CUBE_REST_Z],
              [0.46, 0.09, _CUBE_REST_Z],
              [0.52, -0.04, _CUBE_REST_Z]], dtype=np.float32),
    # Batch 2
    np.array([[0.52, 0.06, _CUBE_REST_Z],
              [0.47, -0.05, _CUBE_REST_Z],
              [0.50, 0.08, _CUBE_REST_Z],
              [0.53, -0.12, _CUBE_REST_Z],
              [0.48, 0.14, _CUBE_REST_Z],
              [0.51, -0.02, _CUBE_REST_Z],
              [0.49, 0.11, _CUBE_REST_Z],
              [0.55, 0.04, _CUBE_REST_Z]], dtype=np.float32),
]

USE_CUDA_GRAPH = True   # set to False for eager mode test


def run_batch(label, batch_num, model, state_0, state_1, control,
              rigid_solver, vbd_solver, collision, soft_contacts,
              n_particles, particle_rest_q, graph_sim):
    """Reset state, run one frame with this batch's cube positions."""
    cube_positions = BATCH_CUBE_POS[batch_num % len(BATCH_CUBE_POS)]

    reset_batch_state(
        model, state_0, state_1,
        cube_positions,
        np.array(_HOME_JOINT_Q, dtype=np.float32), None,
        NUM_WORLDS, n_particles, particle_rest_q,
    )
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    # CUDA graph capture on first call
    use_graph = graph_sim[0] is not None and graph_sim[0] is not False
    if graph_sim[0] is None and wp.get_device().is_cuda:
        collision.collide(state_0, soft_contacts)
        with wp.ScopedCapture() as capture:
            simulate_two_phase(
                state_0, state_1, rigid_solver, vbd_solver,
                collision, soft_contacts, None, control, _SUB_DT, _N_SUBSTEPS,
                use_cuda_graph=True,
            )
        graph_sim[0] = capture.graph
        use_graph = True
        print("  [CUDA graph captured]")
        # Reset after capture
        reset_batch_state(
            model, state_0, state_1,
            cube_positions,
            np.array(_HOME_JOINT_Q, dtype=np.float32), None,
            NUM_WORLDS, n_particles, particle_rest_q,
        )
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    # Run one frame
    if use_graph:
        wp.capture_launch(graph_sim[0])
    else:
        state_0, state_1 = simulate_two_phase(
            state_0, state_1, rigid_solver, vbd_solver,
            collision, soft_contacts, None, control, _SUB_DT, _N_SUBSTEPS,
        )

    particle_q_np = state_0.particle_q.numpy()
    if particle_q_np.ndim == 1:
        particle_q_np = particle_q_np.reshape(-1, 3)
    cube_coms = get_cube_coms_np(particle_q_np, NUM_WORLDS, n_particles)

    print(f"  {label} (cube_pos_batch={batch_num}):")
    all_ok = True
    for w in range(NUM_WORLDS):
        z = cube_coms[w, 2]
        ok = "OK" if 0.025 < z < 0.040 else "BAD"
        if ok == "BAD":
            all_ok = False
        print(f"    world {w}: cube_z={z:.4f} [{ok}]  init_z={cube_positions[w, 2]:.4f}")
    print(f"  -> All OK: {all_ok}")
    return state_0, state_1


def main():
    no_cuda_graph = "--no-cuda-graph" in sys.argv
    if no_cuda_graph:
        print("=== EAGER MODE (no CUDA graph) ===")
        graph_init = [False]
    else:
        print("=== CUDA GRAPH MODE ===")
        graph_init = [None]

    from generate_sequences import _find_panda_urdf
    urdf_path = _find_panda_urdf()
    robot_mb, _, _ = build_robot_builder(urdf_path)
    (model, rigid_solver, vbd_solver, collision, soft_contacts,
     n_particles, particle_rest_q) = build_vbd_batched_model(robot_mb, NUM_WORLDS)

    # CRITICAL FIX: finalize a fresh single-world builder to properly initialize
    # collision geometry (BVH/mesh GPU state).  Without this, some world indices
    # have wrong ground-contact forces and the cube sinks too far at frame 0.
    # This replicates what generate_sequences.py does via build_batch_ik().
    robot_mb_fix, _, _ = build_robot_builder(urdf_path)
    robot_mb_fix.finalize()

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    graph_sim = graph_init

    # Run 3 consecutive batches
    for batch_num in range(3):
        state_0, state_1 = run_batch(
            f"Batch {batch_num}", batch_num,
            model, state_0, state_1, control,
            rigid_solver, vbd_solver, collision, soft_contacts,
            n_particles, particle_rest_q, graph_sim,
        )


if __name__ == "__main__":
    main()
