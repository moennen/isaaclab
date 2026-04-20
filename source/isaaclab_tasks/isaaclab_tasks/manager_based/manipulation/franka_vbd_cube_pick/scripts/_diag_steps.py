#!/usr/bin/env python3
"""Isolate which substep phase causes CUDA error 700."""

import sys
from pathlib import Path
import numpy as np
import warp as wp
import newton

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_sequences import (
    _find_panda_urdf, _HOME_JOINT_Q, _N_ROBOT_JOINTS, _SUB_DT,
    _CUBE_HALF, _PARTICLE_RADIUS,
    build_robot_builder, build_vbd_batched_model, reset_batch_state,
)
from physics.vbd_coupling import apply_soft_body_reactions

NUM_WORLDS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
# COM z = _CUBE_HALF + _PARTICLE_RADIUS so bottom particles sit AT z=radius (not inside ground)
_CUBE_REST_Z = _CUBE_HALF + _PARTICLE_RADIUS
CUBE_INIT = np.array([[0.4, 0.0, _CUBE_REST_Z], [0.4, 0.1, _CUBE_REST_Z]], dtype=np.float32)[:NUM_WORLDS]


def sync_check(label):
    """Call wp.synchronize() and check for CUDA errors by reading a small array."""
    try:
        # Read a small GPU value — this forces a D2H copy which will fail if CUDA is broken
        dummy = wp.zeros(1, dtype=wp.float32, device="cuda:0")
        _ = dummy.numpy()
        print(f"  [sync_ok] {label}")
        return True
    except Exception as e:
        print(f"  [sync_FAIL] {label}: {e}")
        return False


def main():
    urdf = _find_panda_urdf()
    robot_mb, hand_local_idx, _ = build_robot_builder(urdf)
    (model, rigid_solver, vbd_solver, collision, soft_contacts,
     n_particles, particle_rest_q) = build_vbd_batched_model(robot_mb, NUM_WORLDS)
    print(f"[diag] model.tri_count={model.tri_count} particle_count={model.particle_count}")
    # Check tri_poses for validity
    tri_poses_np = model.tri_poses.numpy()
    print(f"[diag] tri_poses shape={tri_poses_np.shape}, has_nan={np.any(np.isnan(tri_poses_np))}, has_inf={np.any(np.isinf(tri_poses_np))}")
    print(f"[diag] tri_poses min={tri_poses_np.min():.4f} max={tri_poses_np.max():.4f}")
    # Check tri_areas
    tri_areas_np = model.tri_areas.numpy()
    print(f"[diag] tri_areas shape={tri_areas_np.shape}, min={tri_areas_np.min():.4f} max={tri_areas_np.max():.4f}")
    # Check tri_materials
    tri_mats_np = model.tri_materials.numpy()
    print(f"[diag] tri_materials shape={tri_mats_np.shape}, min={tri_mats_np.min():.4f} max={tri_mats_np.max():.4f}")
    # Check tri_indices
    tri_idx_np = model.tri_indices.numpy()
    print(f"[diag] tri_indices shape={tri_idx_np.shape}, min={tri_idx_np.min()} max={tri_idx_np.max()}")

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    default_q = np.array(_HOME_JOINT_Q, dtype=np.float32)
    joint_q_ik = wp.array(np.tile(default_q, (NUM_WORLDS, 1)).astype(np.float32), dtype=wp.float32)
    reset_batch_state(
        model, state_0, state_1, CUBE_INIT, default_q, joint_q_ik,
        NUM_WORLDS, n_particles, particle_rest_q,
    )
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    n_ctrl = len(control.joint_target_pos) // NUM_WORLDS
    ctrl_np = np.zeros((NUM_WORLDS, n_ctrl), dtype=np.float32)
    ctrl_np[:, :_N_ROBOT_JOINTS] = default_q
    control.joint_target_pos.assign(ctrl_np.reshape(-1))

    dt = _SUB_DT

    print("\n[diag] Testing substep phases individually ...")
    sync_check("baseline (before any step)")

    # Test VBD alone (no collision, no rigid) with EMPTY contacts
    # to check if pure VBD is stable
    print("\n--- Test A: VBD alone with empty contacts ---")
    # Create empty contacts (no contact count)
    empty_contact_count = wp.zeros(1, dtype=wp.int32, device="cuda:0")
    empty_soft_contacts = collision.contacts
    empty_soft_contacts.soft_contact_count.zero_()

    vbd_solver.step(state_0, state_1, control, empty_soft_contacts, dt)
    ok = sync_check("after vbd alone (empty contacts)")
    if ok:
        pq = state_1.particle_q.numpy().reshape(-1, 3)
        print(f"  CoM: {pq.mean(axis=0)}  max_pos: {np.abs(pq).max():.4f}")

    # Reset state
    reset_batch_state(
        model, state_0, state_1, CUBE_INIT, default_q, joint_q_ik,
        NUM_WORLDS, n_particles, particle_rest_q,
    )
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    print("\n--- Test B: Full pipeline ---")
    # Phase 1: collision
    state_0.clear_forces()
    collision.collide(model, state_0)
    ok = sync_check("after collision.collide")
    if not ok:
        print("  -> collision is the culprit!")
        return
    soft_contacts = collision.contacts
    cnt = int(soft_contacts.soft_contact_count.numpy()[0])
    print(f"  Detected {cnt} soft contacts")
    if cnt > 0:
        shapes = soft_contacts.soft_contact_shape.numpy()[:cnt]
        particles = soft_contacts.soft_contact_particle.numpy()[:cnt]
        normals = soft_contacts.soft_contact_normal.numpy()[:cnt]
        body_pos = soft_contacts.soft_contact_body_pos.numpy()[:cnt]
        pq_np = state_0.particle_q.numpy().reshape(-1, 3)
        radii = model.particle_radius.numpy()
        shape_bodies = model.shape_body.numpy()
        print(f"  shapes: unique={np.unique(shapes)}, body_idxs={np.unique([shape_bodies[s] for s in shapes])}")
        # compute penetrations
        pens = []
        for i in range(min(cnt, 5)):
            pi = particles[i]
            # For ground (body=-1), bx = body_pos (no transform needed, body_q = identity)
            # Actually penetration = -(dot(n, particle_pos - bx) - radius)
            # bx = transform_point(body_q[body_idx], contact_body_pos) or identity for body=-1
            bx = body_pos[i]  # approximate (ignoring body transform for ground)
            n = normals[i]
            pen = -(np.dot(n, pq_np[pi] - bx) - radii[pi])
            pens.append(pen)
        print(f"  first 5 penetrations (approx): {pens}")

    # Phase 2: apply_soft_body_reactions
    apply_soft_body_reactions(
        soft_contacts, state_0, rigid_solver.model,
        soft_contacts.soft_contact_max,
        particle_q_prev=state_1.particle_q,
        friction_epsilon=1e-2,
        dt=dt,
    )
    ok = sync_check("after apply_soft_body_reactions")
    if not ok:
        print("  -> apply_soft_body_reactions is the culprit!")
        return

    # Phase 3: rigid_solver.step
    rigid_solver.step(state_0, state_1, control, None, dt)
    ok = sync_check("after rigid_solver.step")
    if not ok:
        print("  -> rigid_solver.step is the culprit!")
        return

    # Phase 4: vbd_solver.step
    vbd_solver.step(state_0, state_1, control, soft_contacts, dt)
    ok = sync_check("after vbd_solver.step")
    if not ok:
        print("  -> vbd_solver.step is the culprit!")
        return

    print("\n[diag] All phases passed! Trying to read state ...")
    pq = state_1.particle_q.numpy()
    if pq.ndim == 1:
        pq = pq.reshape(-1, 3)
    print(f"  particle_q shape={pq.shape}")
    print(f"  min xyz: {pq.min(axis=0)}")
    print(f"  max xyz: {pq.max(axis=0)}")
    print(f"  mean xyz (CoM): {pq.mean(axis=0)}")
    has_nan = np.any(np.isnan(pq)) or np.any(np.isinf(pq))
    print(f"  has_nan/inf: {has_nan}")
    print(f"  initial CoM was: {pq.mean(axis=0)}")
    # Check which particles have exploded
    exploded = np.abs(pq).max(axis=1) > 1.0
    print(f"  {exploded.sum()}/{len(pq)} particles with |pos|>1m")
    if exploded.any():
        print(f"  First exploded particle: {pq[exploded][0]}")
    print("[diag] Done — no error!")


if __name__ == "__main__":
    main()
