#!/usr/bin/env python3
"""Check ground plane shape flags and world assignment for soft contact detection."""

import sys
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
)

NUM_WORLDS = 2
CUBE_INIT = np.array(
    [[0.4, 0.0, _CUBE_REST_Z], [0.4, 0.1, _CUBE_REST_Z]], dtype=np.float32
)[:NUM_WORLDS]


def main():
    urdf = _find_panda_urdf()
    robot_mb, hand_local_idx, _ = build_robot_builder(urdf)
    (model, rigid_solver, vbd_solver, collision, soft_contacts,
     n_particles, particle_rest_q) = build_vbd_batched_model(robot_mb, NUM_WORLDS)

    # Print shape info
    shape_world = model.shape_world.numpy()
    shape_flags = model.shape_flags.numpy()
    shape_type  = model.shape_type.numpy()
    shape_body  = model.shape_body.numpy()

    print(f"Total shapes: {model.shape_count}")
    print(f"Ground plane shape types: {shape_type}")
    print(f"\nShape summary:")
    for i in range(model.shape_count):
        flags_val = shape_flags[i]
        # Check particle collision flag (ShapeFlags.COLLIDE_PARTICLES = 4)
        from newton._src.geometry.types import ShapeFlags
        has_particle = bool(flags_val & ShapeFlags.COLLIDE_PARTICLES)
        has_shape    = bool(flags_val & ShapeFlags.COLLIDE_SHAPES)
        from newton._src.geometry.types import GeoType
        geo = GeoType(shape_type[i]).name
        print(f"  shape[{i:2d}]: world={shape_world[i]:2d}, body={shape_body[i]:2d}, "
              f"type={geo}, collide_particles={has_particle}, collide_shapes={has_shape}")

    # Check particle world assignments
    pw = model.particle_world.numpy()
    pq = np.array([[p[0], p[1], p[2]] for p in model.particle_q.numpy().reshape(-1, 3)])
    print(f"\nParticle world assignments: unique={np.unique(pw)}")
    print(f"Particle z-range: min={pq[:,2].min():.4f}, max={pq[:,2].max():.4f}")

    # Reset state and check contacts
    state_0 = model.state()
    state_1 = model.state()
    default_q = np.array(_HOME_JOINT_Q, dtype=np.float32)
    joint_q_ik = wp.array(np.tile(default_q, (NUM_WORLDS, 1)).astype(np.float32), dtype=wp.float32)
    reset_batch_state(
        model, state_0, state_1, CUBE_INIT, default_q, joint_q_ik,
        NUM_WORLDS, n_particles, particle_rest_q,
    )
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    pq_np = state_0.particle_q.numpy()
    if pq_np.ndim > 1:
        pq_np = pq_np.reshape(-1, 3)
    else:
        pq_np = pq_np.reshape(-1, 3)
    print(f"\nAfter reset - particle z: min={pq_np[:,2].min():.4f}, max={pq_np[:,2].max():.4f}")
    print(f"Expected bottom particles at z={_CUBE_REST_Z - _CUBE_HALF:.4f} = particle_radius={_PARTICLE_RADIUS:.4f}")

    # Collision
    state_0.clear_forces()
    collision.collide(model, state_0)
    contacts = collision.contacts
    cnt = int(contacts.soft_contact_count.numpy()[0])
    print(f"\nSoft contacts detected: {cnt}")
    if cnt > 0:
        shapes = contacts.soft_contact_shape.numpy()[:cnt]
        particles = contacts.soft_contact_particle.numpy()[:cnt]
        normals = contacts.soft_contact_normal.numpy()[:cnt]
        print(f"  shapes: {np.unique(shapes)}")
        print(f"  particles: {np.unique(particles)}")
        # Compute distances
        for i in range(min(cnt, 3)):
            pi = particles[i]
            p_pos = pq_np[pi]
            si = shapes[i]
            sw = shape_world[si]
            print(f"  contact {i}: particle[{pi}]@z={p_pos[2]:.4f} vs shape[{si}](world={sw}), "
                  f"normal={normals[i].tolist()}")
    else:
        print("  -> No contacts! Cube will fall through floor.")
        print("  Checking ground plane(s) that should collide with particles...")
        from newton._src.geometry.types import GeoType, ShapeFlags
        ground_shapes = [i for i in range(model.shape_count)
                         if shape_type[i] == GeoType.PLANE and
                            bool(shape_flags[i] & ShapeFlags.COLLIDE_PARTICLES)]
        print(f"  Ground shapes with COLLIDE_PARTICLES: {ground_shapes}")
        if not ground_shapes:
            print("  ERROR: No ground shape has COLLIDE_PARTICLES flag!")
            return

        # Manual distance check for bottom particles
        for si in ground_shapes:
            sw = shape_world[si]
            print(f"\n  Ground shape[{si}], world={sw}")
            # Bottom particles
            bottom_z = pq_np[:,2].min()
            bottom_mask = pq_np[:,2] < bottom_z + 0.001
            bottom_idx = np.where(bottom_mask)[0]
            print(f"  Bottom particles (z={bottom_z:.4f}): indices {bottom_idx[:5]}")
            print(f"  Their particle_world: {[int(pw[idx]) for idx in bottom_idx[:5]]}")
            print(f"  Contact condition: d < margin + radius")
            print(f"    d = z (plane_sdf for z>0) = {bottom_z:.4f}")
            margin = _PARTICLE_RADIUS * 3.0
            radius = _PARTICLE_RADIUS
            print(f"    margin={margin:.4f}, radius={radius:.4f}, margin+radius={margin+radius:.4f}")
            print(f"    {bottom_z:.4f} < {margin+radius:.4f} ? {bottom_z < margin+radius}")
            print(f"    particle_world[{bottom_idx[0]}]={pw[bottom_idx[0]]}, shape_world[{si}]={sw}")
            same_world = (pw[bottom_idx[0]] == sw) or (pw[bottom_idx[0]] == -1) or (sw == -1)
            print(f"    same_world check: {same_world}")


if __name__ == "__main__":
    main()
