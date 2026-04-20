# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extended NewtonManager for the franka_vbd_cube_pick task (VBD soft-body cube).

The deformable cube is created procedurally via ``add_soft_grid()`` inside
``start_simulation()``.  At scale (4096 envs) the particle data for env-0 is
tiled to all remaining envs via numpy broadcast — avoiding an O(N_envs) Python
loop — and the graph coloring is computed once on the single-env topology then
offset-tiled for each env.

Two-phase stepping uses same-substep two-way coupling from ``vbd_coupling.py``:
soft contacts are detected, reactions written to ``state.body_f``, and both
the rigid MuJoCo solver and the VBD solver step with the same contact buffer.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.physics import PhysicsManager
from isaaclab.utils.math import matrix_from_quat

from .vbd_coupling import apply_soft_body_reactions

logger = logging.getLogger("franka_vbd_cube_pick.newton.manager")

# ---------------------------------------------------------------------------
# Warp kernels — run inside the CUDA graph each step (or eagerly after reset).
# One thread per environment.
# ---------------------------------------------------------------------------


@wp.kernel
def _kernel_particle_com_kabsch(
    particle_q:      wp.array(dtype=wp.vec3f),
    particle_rest_q: wp.array(dtype=wp.vec3f),
    particles_per_env: int,
    pos_out:  wp.array(dtype=wp.vec3f),
    quat_out: wp.array(dtype=wp.quatf),
):
    """Compute particle CoM position and Kabsch orientation per environment.

    Args:
        particle_q: Current particle positions [m], shape (num_envs * particles_per_env,).
        particle_rest_q: Rest-pose particle positions [m], same shape.
        particles_per_env: Number of particles per environment.
        pos_out: Output CoM positions [m], shape (num_envs,).
        quat_out: Output orientations (wxyz quaternion), shape (num_envs,).
    """
    env_i = wp.tid()
    start = env_i * particles_per_env
    inv_n = 1.0 / float(particles_per_env)

    com_cur  = wp.vec3(0.0, 0.0, 0.0)
    com_rest = wp.vec3(0.0, 0.0, 0.0)
    for p in range(particles_per_env):
        com_cur  = com_cur  + particle_q[start + p]
        com_rest = com_rest + particle_rest_q[start + p]
    com_cur  = com_cur  * inv_n
    com_rest = com_rest * inv_n
    pos_out[env_i] = com_cur

    # Kabsch H = Σ (rest_centred)^T ⊗ (cur_centred)
    H = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for p in range(particles_per_env):
        dr = particle_rest_q[start + p] - com_rest
        dc = particle_q[start + p]      - com_cur
        H  = H + wp.outer(dr, dc)

    # SVD: H = U * diag(S) * Vᵀ  →  R = V @ Uᵀ
    U, _S, V = wp.svd3(H)
    R = V @ wp.transpose(U)

    # Fix reflection: det(R) must be +1
    if wp.determinant(R) < 0.0:
        V = wp.mat33(
            V[0, 0], V[0, 1], -V[0, 2],
            V[1, 0], V[1, 1], -V[1, 2],
            V[2, 0], V[2, 1], -V[2, 2],
        )
        R = V @ wp.transpose(U)

    quat_out[env_i] = wp.quat_from_matrix(R)


@wp.kernel
def _kernel_particle_com_vel(
    particle_qd:      wp.array(dtype=wp.vec3f),
    particles_per_env: int,
    vel_out:           wp.array(dtype=wp.vec3f),
):
    """Compute particle CoM velocity per environment.

    Args:
        particle_qd: Particle velocities [m/s], shape (num_envs * particles_per_env,).
        particles_per_env: Number of particles per environment.
        vel_out: Output CoM velocities [m/s], shape (num_envs,).
    """
    env_i = wp.tid()
    start = env_i * particles_per_env
    inv_n = 1.0 / float(particles_per_env)
    v = wp.vec3(0.0, 0.0, 0.0)
    for p in range(particles_per_env):
        v = v + particle_qd[start + p]
    vel_out[env_i] = v * inv_n


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_env_origins(num_envs: int, device: str) -> torch.Tensor:
    """Return env origins (num_envs, 3) from the USD stage env prims."""
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    origins = []
    root = stage.GetPrimAtPath("/World/envs")
    if root and root.IsValid():
        children = sorted(
            root.GetChildren(),
            key=lambda c: int(c.GetName().split("_")[1]) if c.GetName().startswith("env_") else -1,
        )
        for c in children:
            if not c.GetName().startswith("env_"):
                continue
            t_attr = c.GetAttribute("xformOp:translate")
            if t_attr:
                t = t_attr.Get()
                origins.append([float(t[0]), float(t[1]), float(t[2])])
            else:
                origins.append([0.0, 0.0, 0.0])
    if len(origins) != num_envs:
        logger.warning(
            "Expected %d env origins but found %d — filling with zeros.", num_envs, len(origins)
        )
        while len(origins) < num_envs:
            origins.append([0.0, 0.0, 0.0])
    return torch.tensor(origins[:num_envs], dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class FrankaVbdCubePickNewtonManager(NewtonManager):
    """Newton manager extended for the Franka VBD cube pick task.

    Creates a procedural deformable cube via ``add_soft_grid()`` and runs
    same-substep two-way coupling between the VBD solver (cube) and the
    MuJoCo rigid solver (Franka robot).

    MDP helpers:

    - :meth:`get_object_pose` — particle CoM position and Kabsch orientation,
      pre-computed every physics step inside the CUDA graph.
    - :meth:`get_object_velocity` — particle CoM linear velocity.
    - :meth:`reset_particles` — teleport cube particles to a new pose after reset.
    - :meth:`reset_particle_velocities` — zero particle velocities after reset.
    """

    _vbd_solver: Any = None
    _soft_collision_pipeline: Any = None
    _soft_contacts: Any = None
    _soft_contact_max: int = 0
    _particles_per_env: int | None = None
    # Build-time particle positions (world space), shape (num_envs * particles_per_env, 3)
    _particle_q_build: torch.Tensor | None = None
    # inv(T_build) per env [num_envs, 4, 4]
    _T_build_inv: torch.Tensor | None = None
    # Rest-pose particle positions (updated on each reset), same shape as _particle_q_build
    _particle_rest_q: torch.Tensor | None = None
    # Warp mirror of _particle_rest_q — kept in sync for use inside Warp kernels.
    _particle_rest_q_wp: Any = None
    # Pre-allocated output arrays written by obs-cache kernels every physics step.
    _particle_pos_out: Any = None   # wp.array(num_envs, dtype=wp.vec3f)
    _particle_quat_out: Any = None  # wp.array(num_envs, dtype=wp.quatf)
    _particle_vel_out: Any = None   # wp.array(num_envs, dtype=wp.vec3f)

    # ------------------------------------------------------------------ #
    # Lifecycle overrides
    # ------------------------------------------------------------------ #

    @classmethod
    def start_simulation(cls) -> None:
        cls._start_simulation_vbd()

    @classmethod
    def _start_simulation_vbd(cls) -> None:
        """Add a deformable cube per env via add_soft_grid(), then call super().start_simulation().

        Fast-path: calls ``add_soft_grid()`` for env-0 only, snapshots the resulting
        builder data, then tiles it to all remaining envs via numpy broadcast.
        Graph coloring is computed once on the single-env topology and tiled with
        per-env particle-index offsets — O(single_env_size) instead of O(num_envs²).
        """
        cfg = PhysicsManager._cfg
        device = PhysicsManager._device

        num_envs = cls._num_envs or 1

        # Lamé parameters computed from Young's modulus and Poisson's ratio.
        E  = float(cfg.young_modulus)
        nu = float(cfg.poisson_ratio)
        k_mu     = E / (2.0 * (1.0 + nu))
        k_lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        cube_size  = float(cfg.cube_size)
        resolution = int(cfg.cube_resolution)
        cell_size  = cube_size / resolution

        # Initial cube position (world frame for env-0 = env-origin + local offset).
        # Cube center is placed at z = cube_size/2 (resting on the ground plane z=0).
        init_local = np.array([0.5, 0.0, cube_size / 2.0], dtype=np.float32)

        env_origins_t  = _discover_env_origins(num_envs, device)
        env_origins    = env_origins_t.cpu().numpy()

        existing_particles = getattr(cls._builder, "particle_count", 0) if cls._builder else 0

        world_pos_0 = env_origins[0] + init_local

        # ------------------------------------------------------------------
        # Step 1: add env-0 via add_soft_grid() so Newton computes tet rest
        #         poses and any surface topology.
        # ------------------------------------------------------------------

        # Set default particle radius BEFORE add_soft_grid (it does not take
        # a per-particle radius parameter unlike add_soft_mesh).
        cls._builder.default_particle_radius = float(cfg.particle_radius)

        snap_before = {
            "particle_q":        len(cls._builder.particle_q),
            "tet_indices":       len(cls._builder.tet_indices),
            "tet_poses":         len(cls._builder.tet_poses),
            "tet_materials":     len(cls._builder.tet_materials),
            "tet_activations":   len(cls._builder.tet_activations),
            "tri_indices":       len(cls._builder.tri_indices),
            "tri_poses":         len(cls._builder.tri_poses) if hasattr(cls._builder, "tri_poses") else 0,
            "tri_materials":     len(cls._builder.tri_materials),
            "edge_indices":      len(cls._builder.edge_indices) if hasattr(cls._builder, "edge_indices") else 0,
            "edge_bending_properties": (
                len(cls._builder.edge_bending_properties)
                if hasattr(cls._builder, "edge_bending_properties") else 0
            ),
        }

        cls._builder.add_soft_grid(
            pos=wp.vec3(float(world_pos_0[0]), float(world_pos_0[1]), float(world_pos_0[2])),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=resolution,
            dim_y=resolution,
            dim_z=resolution,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=float(cfg.density),
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=float(cfg.k_damp),
        )

        # Snapshot what env-0 contributed.
        n_particles = len(cls._builder.particle_q) - snap_before["particle_q"]
        n_tris_per_env = len(cls._builder.tri_indices) - snap_before["tri_indices"]
        has_edges  = hasattr(cls._builder, "edge_indices")
        n_edges_per_env = (
            (len(cls._builder.edge_indices) - snap_before["edge_indices"])
            if has_edges else 0
        )

        tets_env0      = np.array(cls._builder.tet_indices[snap_before["tet_indices"]:], dtype=np.int32)
        tet_poses_env0 = list(cls._builder.tet_poses[snap_before["tet_poses"]:])
        tet_mats_env0  = list(cls._builder.tet_materials[snap_before["tet_materials"]:])
        tet_acts_env0  = list(cls._builder.tet_activations[snap_before["tet_activations"]:])

        tri_env0     = np.array(cls._builder.tri_indices[snap_before["tri_indices"]:], dtype=np.int32) if n_tris_per_env > 0 else None
        tri_mats_env0  = list(cls._builder.tri_materials[snap_before["tri_materials"]:]) if n_tris_per_env > 0 else []
        has_tri_poses  = hasattr(cls._builder, "tri_poses") and len(cls._builder.tri_poses) > snap_before["tri_poses"]
        tri_poses_env0 = list(cls._builder.tri_poses[snap_before["tri_poses"]:]) if has_tri_poses else []

        edges_env0 = (
            np.array(cls._builder.edge_indices[snap_before["edge_indices"]:], dtype=np.int32)
            if n_edges_per_env > 0 else None
        )
        edge_bends_env0 = (
            list(cls._builder.edge_bending_properties[snap_before["edge_bending_properties"]:])
            if n_edges_per_env > 0 else None
        )

        pq_env0  = np.array(cls._builder.particle_q[snap_before["particle_q"]:],  dtype=np.float32)
        pqd_env0 = np.array(cls._builder.particle_qd[snap_before["particle_q"]:], dtype=np.float32)
        pm_env0  = list(cls._builder.particle_mass[snap_before["particle_q"]:])
        pr_env0  = list(cls._builder.particle_radius[snap_before["particle_q"]:])
        pf_env0  = list(cls._builder.particle_flags[snap_before["particle_q"]:])

        n_tets = len(tets_env0) // 4 if tets_env0.ndim == 1 else len(tets_env0)
        logger.info(
            "[VBD] Env-0 grid: %d particles, %d tets, %d tris, %d edges "
            "(resolution=%d, cell_size=%.4f m, k_mu=%.3g, k_lambda=%.3g)",
            n_particles, n_tets, n_tris_per_env, n_edges_per_env,
            resolution, cell_size, k_mu, k_lambda,
        )

        # ------------------------------------------------------------------
        # Step 2: tile envs 1..N-1 via numpy broadcast (no Python loops).
        # ------------------------------------------------------------------
        if num_envs > 1:
            all_world_pos = env_origins[1:] + init_local   # (num_envs-1, 3)
            all_deltas    = all_world_pos - world_pos_0    # (num_envs-1, 3)

            # All particle positions for envs 1..N in one numpy op.
            all_pq_new  = pq_env0[None, :, :] + all_deltas[:, None, :]  # broadcast
            all_pq_flat = all_pq_new.reshape(-1, 3)
            extra_particles = [wp.vec3(float(r[0]), float(r[1]), float(r[2])) for r in all_pq_flat]
            extra_pqd = list(pqd_env0) * (num_envs - 1)
            extra_mass   = pm_env0 * (num_envs - 1)
            extra_radius = pr_env0 * (num_envs - 1)
            extra_flags  = pf_env0 * (num_envs - 1)
            extra_world  = [cls._builder.current_world] * (n_particles * (num_envs - 1))

            cls._builder.particle_q.extend(extra_particles)
            cls._builder.particle_qd.extend(
                [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in extra_pqd]
            )
            cls._builder.particle_mass.extend(extra_mass)
            cls._builder.particle_radius.extend(extra_radius)
            cls._builder.particle_flags.extend(extra_flags)
            cls._builder.particle_world.extend(extra_world)

            # Build index offsets for envs 1..N-1.
            env_offsets = existing_particles + np.arange(1, num_envs, dtype=np.int32) * n_particles

            # Tetrahedra: tile all envs at once.
            all_tets = tets_env0[None, :, :] + env_offsets[:, None, None]
            cls._builder.tet_indices.extend(map(tuple, all_tets.reshape(-1, 4).tolist()))
            cls._builder.tet_poses.extend(tet_poses_env0 * (num_envs - 1))
            cls._builder.tet_materials.extend(tet_mats_env0 * (num_envs - 1))
            cls._builder.tet_activations.extend(tet_acts_env0 * (num_envs - 1))

            # Surface triangles (may be empty for a solid grid).
            if n_tris_per_env > 0 and tri_env0 is not None:
                all_tris = tri_env0[None, :, :] + env_offsets[:, None, None]
                cls._builder.tri_indices.extend(map(tuple, all_tris.reshape(-1, 3).tolist()))
                cls._builder.tri_materials.extend(tri_mats_env0 * (num_envs - 1))
                if tri_poses_env0:
                    cls._builder.tri_poses.extend(tri_poses_env0 * (num_envs - 1))

            # Bending edges.
            if n_edges_per_env > 0 and edges_env0 is not None:
                all_edges = edges_env0[None, :, :] + env_offsets[:, None, None]
                cls._builder.edge_indices.extend(map(tuple, all_edges.reshape(-1, 4).tolist()))
                cls._builder.edge_bending_properties.extend(edge_bends_env0 * (num_envs - 1))

        cls._particles_per_env = n_particles

        # Build inv(T_build) per env for reset_particles().
        all_world_pos_full = np.vstack([world_pos_0[None, :], env_origins[1:] + init_local]) if num_envs > 1 else world_pos_0[None, :]
        T_all = np.eye(4, dtype=np.float32)[None].repeat(num_envs, axis=0)
        T_all[:, :3, 3] = all_world_pos_full
        T_stacked = torch.from_numpy(T_all).to(device)
        cls._T_build_inv = torch.linalg.inv(T_stacked.double()).float()

        # ------------------------------------------------------------------
        # Step 3: graph-color the single-env topology and tile.
        #
        # Build edges from tet adjacency (each tet contributes 6 edges:
        # all pairs among its 4 nodes).  Use color_graph on the env-0
        # topology, then tile by adding (env_idx * n_particles) offset.
        # ------------------------------------------------------------------
        logger.info("[VBD] Coloring single-env graph (%d nodes)...", n_particles)
        from newton._src.sim.graph_coloring import (
            ColoringAlgorithm,
            color_graph,
            color_rigid_bodies,
        )

        tet_np = (tets_env0 - existing_particles).reshape(-1, 4)  # local (0-based) indices

        # Build unique undirected edges from tet adjacency.
        edge_set: set[tuple[int, int]] = set()
        for row in tet_np:
            i, j, k, l = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            for a, b in ((i, j), (i, k), (i, l), (j, k), (j, l), (k, l)):
                edge_set.add((min(a, b), max(a, b)))
        edge_np = np.array(sorted(edge_set), dtype=np.int32)  # (E, 2)
        edge_wp = wp.array(edge_np, dtype=int, device="cpu")

        single_env_colors: list[np.ndarray] = color_graph(
            n_particles,
            edge_wp,
            balance_colors=True,
            target_max_min_color_ratio=1.1,
            algorithm=ColoringAlgorithm.MCS,
        )
        logger.info("[VBD] Single-env coloring: %d colors", len(single_env_colors))

        # Tile: for each color group, stack all env replicas.
        tiled_colors = []
        for color_group in single_env_colors:
            tiled = np.concatenate([
                color_group + existing_particles + env_idx * n_particles
                for env_idx in range(num_envs)
            ])
            tiled_colors.append(tiled)

        cls._builder.set_coloring(tiled_colors)
        cls._builder.body_color_groups = color_rigid_bodies(
            cls._builder.body_count,
            cls._builder.joint_parent,
            cls._builder.joint_child,
        )

        # CRITICAL: finalize a fresh single-world builder with mesh geometry BEFORE the
        # main batched builder is finalized.  Without this warmup, some env indices can
        # have wrong ground-contact forces for VBD particles → cube sinks at rest.
        # Root cause: Newton's BVH/mesh GPU state (wp.Mesh) may not be properly
        # initialized for all worlds when add_builder() shares unfinalized geo objects.
        # Finalizing a minimal single-world model first ensures the required CUDA
        # infrastructure (BVH kernels, device memory) is ready.
        # Equivalent fix applied to replay_sequences.py (lines 290-296).
        try:
            import newton as _nt
            import numpy as _np_warmup
            _wb = _nt.ModelBuilder()
            _mesh = _nt.Mesh(
                _np_warmup.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], dtype=_np_warmup.float32),
                _np_warmup.array([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3], dtype=_np_warmup.int32),
            )
            _wb.add_shape_mesh(body=-1, mesh=_mesh)
            _wb.finalize()
            del _wb, _mesh
            logger.info("[VBD] Collision geometry warmup finalize done.")
        except Exception as exc:
            logger.warning("[VBD] Collision geometry warmup finalize skipped: %s", exc)

        super().start_simulation()

        # Cache build-time particle positions.
        pq = wp.to_torch(cls._state_0.particle_q).float().clone()
        cls._particle_q_build = pq.clone()
        cls._particle_rest_q  = pq.clone()

        if cls._model is not None:
            grav = cls._model.gravity
            logger.info(
                "[VBD] Model gravity=%s  particle_count=%d",
                grav, cls._model.particle_count,
            )

        # Set contact parameters on the finalised model.
        if cls._model is not None:
            cls._model.soft_contact_ke = float(cfg.soft_contact_ke)
            cls._model.soft_contact_kd = float(cfg.soft_contact_kd)
            cls._model.soft_contact_mu = float(cfg.soft_contact_mu)
            logger.info(
                "[VBD] Contact params: ke=%.1f kd=%.1e mu=%.2f",
                cfg.soft_contact_ke, cfg.soft_contact_kd, cfg.soft_contact_mu,
            )

            # Disable COLLIDE_PARTICLES on arm link shapes (collision_group=2).
            # The rigid collision_group mechanism prevents BOX-BOX contact between
            # the arm and the cube, but VBD particle-rigid contacts use a separate
            # COLLIDE_PARTICLES shape flag.  Without clearing this flag, arm links
            # sweep through the VBD cube during HOME→PRE_GRASP (t < 2s) and launch
            # it 2–68m laterally.  Only finger BOX shapes (group=1) and the ground
            # plane (group≠2) should contact VBD particles.
            _FLAG_COLLIDE_PARTICLES = 1 << 2  # ShapeFlags.COLLIDE_PARTICLES
            shape_flags_np = cls._model.shape_flags.numpy().copy()
            shape_cg_np    = cls._model.shape_collision_group.numpy()
            n_disabled = int(np.sum(shape_cg_np == 2))
            shape_flags_np[shape_cg_np == 2] &= ~_FLAG_COLLIDE_PARTICLES
            cls._model.shape_flags.assign(shape_flags_np)
            logger.info(
                "[VBD] Cleared COLLIDE_PARTICLES on %d arm link shapes (collision_group=2).",
                n_disabled,
            )

            # Set shape contact material on finger/gripper shapes (body-attached, not arm links).
            # This matches the ShapeConfig(ke=5e4, kd=5e2, kf=1e3, mu=0.75) applied to finger
            # box shapes in generate_sequences.py.  The intentional ke asymmetry
            # (reaction_ke = soft_contact_ke = 1e3 << shape_ke = 5e4) drives VBD avg_ke ≈ 25500
            # and friction scale ≈ 47800 N/m → tracking efficiency η ≈ 0.96 over the full lift.
            shape_body_np  = cls._model.shape_body.numpy()
            finger_mask    = (shape_cg_np != 2) & (shape_body_np >= 0)
            n_finger       = int(finger_mask.sum())
            if n_finger > 0:
                ke_np = cls._model.shape_material_ke.numpy().copy()
                mu_np = cls._model.shape_material_mu.numpy().copy()
                ke_np[finger_mask] = float(cfg.shape_contact_ke)
                mu_np[finger_mask] = float(cfg.shape_contact_mu)
                cls._model.shape_material_ke.assign(ke_np)
                cls._model.shape_material_mu.assign(mu_np)
                # kd and kf if present (Newton may or may not expose these).
                for attr, val in (
                    ("shape_material_kd", float(cfg.shape_contact_kd)),
                    ("shape_material_kf", float(cfg.shape_contact_kf)),
                ):
                    arr = getattr(cls._model, attr, None)
                    if arr is not None:
                        a = arr.numpy().copy()
                        a[finger_mask] = val
                        arr.assign(a)
                logger.info(
                    "[VBD] Set shape material ke=%.1f mu=%.2f on %d finger shapes "
                    "(effective mu_eff=sqrt(%.1f×%.2f)=%.2f).",
                    cfg.shape_contact_ke, cfg.shape_contact_mu, n_finger,
                    cfg.soft_contact_mu, cfg.shape_contact_mu,
                    (cfg.soft_contact_mu * cfg.shape_contact_mu) ** 0.5,
                )

    @classmethod
    def initialize_solver(cls) -> None:
        # IMPORTANT: create VBD solver BEFORE super().initialize_solver() so that
        # when the base class captures the CUDA graph it calls _simulate_two_phase.
        if cls._model is not None and cls._particles_per_env is not None:
            try:
                from newton.solvers import SolverVBD
            except ImportError:
                raise ImportError("newton.solvers.SolverVBD not found — update Newton.")

            cfg = PhysicsManager._cfg
            num_envs         = cls._num_envs or 1
            particles_per_env = cls._particles_per_env
            shapes_per_env    = cls._model.shape_count // max(num_envs, 1)

            max_contacts_per_env  = getattr(cfg, "vbd_max_contacts_per_env", 200)
            practical_contact_max = min(
                max_contacts_per_env * num_envs,
                shapes_per_env * particles_per_env * num_envs,
            )
            logger.info(
                "[VBD] soft_contact_max=%d (%d/env × %d envs; "
                "theoretical max=%d × %d × %d).",
                practical_contact_max,
                max_contacts_per_env, num_envs,
                shapes_per_env, particles_per_env, num_envs,
            )

            # Pre-allocate collision pipeline and contacts buffer (cloth_franka pattern).
            if cls._state_0 is not None:
                try:
                    import newton as _newton_pkg
                    soft_margin = float(cfg.particle_radius) * 3.0

                    def _make_pipeline(spw: int, contact_max: int):
                        logger.info(
                            "[VBD] Creating CollisionPipeline "
                            "(batched: particles_per_world=%d, shapes_per_world=%d, "
                            "num_worlds=%d, soft_contact_max=%d, margin=%.4f m)...",
                            particles_per_env, spw, num_envs, contact_max, soft_margin,
                        )
                        pipeline = _newton_pkg.CollisionPipeline(
                            cls._model,
                            soft_contact_margin=soft_margin,
                            soft_contact_max=contact_max,
                            particles_per_world=particles_per_env,
                            shapes_per_world=spw,
                        )
                        contacts = pipeline.contacts()
                        pipeline.collide(cls._state_0, contacts)  # warm-up outside graph
                        return pipeline, contacts

                    cls._soft_collision_pipeline, cls._soft_contacts = _make_pipeline(
                        shapes_per_env, practical_contact_max
                    )

                    # Auto-detect tight shapes_per_world.
                    shapes_per_env = cls._auto_detect_shapes_per_world(
                        cls._soft_contacts, shapes_per_env, num_envs,
                        getattr(cfg, "vbd_shapes_per_world", None),
                    )

                    # Rebuild with tighter shapes if auto-detect found a smaller value.
                    if shapes_per_env < (cls._model.shape_count // max(num_envs, 1)):
                        cls._soft_collision_pipeline, cls._soft_contacts = _make_pipeline(
                            shapes_per_env, practical_contact_max
                        )

                    logger.info("[VBD] Soft-body collision pipeline ready.")
                except Exception as exc:
                    logger.warning("[VBD] CollisionPipeline setup failed: %s", exc)
                    cls._soft_collision_pipeline = None
                    cls._soft_contacts = None

            # rigid_contact_k_start: VBD warmstart stiffness for body-particle contacts.
            # Default (100 N/m) is too low — friction ≈ 0.05 N vs gravity 4.9 N.
            # Use avg_ke = 0.5*(soft_ke + shape_ke) so VBD starts at full material stiffness.
            rigid_k_start = cfg.vbd_rigid_contact_k_start
            if rigid_k_start is None:
                rigid_k_start = 0.5 * (float(cfg.soft_contact_ke) + float(cfg.shape_contact_ke))

            cls._vbd_solver = SolverVBD(
                cls._model,
                iterations=int(cfg.vbd_iterations),
                integrate_with_external_rigid_solver=True,
                particle_enable_self_contact=False,
                max_soft_contacts=practical_contact_max,
                particle_max_velocity=float(cfg.vbd_max_particle_velocity),
                rigid_contact_k_start=float(rigid_k_start),
            )
            cls._soft_contact_max = practical_contact_max
            logger.info(
                "[VBD] SolverVBD initialized (iterations=%d, max_soft_contacts=%d, "
                "max_vel=%.1f m/s, rigid_k_start=%.1f).",
                cfg.vbd_iterations, practical_contact_max,
                cfg.vbd_max_particle_velocity, rigid_k_start,
            )

            # Allocate persistent obs-cache arrays.
            device_str = str(PhysicsManager._device)
            n_obs_envs = num_envs
            ppe = cls._particles_per_env
            cls._particle_pos_out  = wp.zeros(n_obs_envs, dtype=wp.vec3f, device=device_str)
            cls._particle_quat_out = wp.zeros(n_obs_envs, dtype=wp.quatf, device=device_str)
            cls._particle_vel_out  = wp.zeros(n_obs_envs, dtype=wp.vec3f, device=device_str)

            rest_np = cls._particle_rest_q[:n_obs_envs * ppe].view(-1, 3).cpu().numpy()
            cls._particle_rest_q_wp = wp.array(rest_np, dtype=wp.vec3f, device=device_str)
            logger.info("[VBD] Obs-cache arrays allocated (%d envs).", n_obs_envs)

        super().initialize_solver()  # creates rigid solver + captures CUDA graph

    @classmethod
    def _auto_detect_shapes_per_world(
        cls,
        contacts,
        shapes_per_env: int,
        num_envs: int,
        override: int | None,
    ) -> int:
        """Return a tight shapes_per_world value based on warmup contact shape indices."""
        if override is not None:
            logger.info("[VBD] shapes_per_world override=%d (from config).", override)
            return min(override, shapes_per_env)

        try:
            count_t = wp.to_torch(contacts.soft_contact_count)
            count = int(count_t.item())
            if count == 0:
                logger.info(
                    "[VBD] No contacts in warmup — keeping shapes_per_world=%d.",
                    shapes_per_env,
                )
                return shapes_per_env

            shape_t = wp.to_torch(contacts.soft_contact_shape)[:count]
            max_global = int(shape_t.max().item())
            max_local  = max_global % shapes_per_env

            tight    = max(min((max_local + 1) * 4, shapes_per_env), 1)
            speedup  = shapes_per_env / tight
            logger.info(
                "[VBD] contact shape range: max local index=%d → "
                "tight shapes_per_world=%d (full=%d, %.1f× smaller). "
                "Set vbd_shapes_per_world=%d in config to hard-code.",
                max_local, tight, shapes_per_env, speedup, tight,
            )
            return tight
        except Exception as exc:
            logger.warning(
                "[VBD] _auto_detect_shapes_per_world failed (%s) — keeping %d.",
                exc, shapes_per_env,
            )
            return shapes_per_env

    @classmethod
    def _refresh_obs_cache(cls) -> None:
        """Launch Warp kernels to update cached pose/velocity from current particle state.

        Called inside the CUDA graph every step and eagerly after reset.
        """
        if cls._particle_pos_out is None or cls._state_0 is None or cls._particles_per_env is None:
            return
        num_envs = cls._num_envs or 1
        ppe = cls._particles_per_env
        wp.launch(
            _kernel_particle_com_kabsch,
            dim=num_envs,
            inputs=[
                cls._state_0.particle_q,
                cls._particle_rest_q_wp,
                ppe,
                cls._particle_pos_out,
                cls._particle_quat_out,
            ],
        )
        wp.launch(
            _kernel_particle_com_vel,
            dim=num_envs,
            inputs=[
                cls._state_0.particle_qd,
                ppe,
                cls._particle_vel_out,
            ],
        )

    @classmethod
    def _simulate(cls) -> None:
        if cls._vbd_solver is not None:
            cls._simulate_two_phase()
            return
        super()._simulate()

    @classmethod
    def _simulate_two_phase(cls) -> None:
        """Two-phase step: rigid MuJoCo (robot) + VBD (cube) with same-substep two-way coupling.

        Substep order:
          1. collide(s0, soft_contacts)  — detect particle-rigid contacts
          2. apply_soft_body_reactions   — write reaction forces into state_0.body_f
          3. rigid.step(s0 → s1)         — reads body_f; writes new rigid positions
          4. vbd.step(s0 → s1)           — uses same contacts
          5. swap / assign
          6. _refresh_obs_cache()        — update CoM pose/vel (captured in CUDA graph)
        """
        if cls._needs_collision_pipeline:
            cls._collision_pipeline.collide(cls._state_0, cls._contacts)
            contacts_rigid = cls._contacts
        else:
            contacts_rigid = None

        cfg = PhysicsManager._cfg
        need_copy = getattr(cfg, "use_cuda_graph", False) and cls._num_substeps % 2 == 1

        for i in range(cls._num_substeps):
            # Detect particle-rigid contacts.
            if cls._soft_collision_pipeline is not None:
                cls._soft_collision_pipeline.collide(cls._state_0, cls._soft_contacts)
                contacts_soft = cls._soft_contacts
            else:
                contacts_soft = cls._model.collide(cls._state_0)

            two_way = getattr(cfg, "vbd_two_way_coupling", True)

            if cls._use_single_state:
                cls._state_0.clear_forces()
                if two_way:
                    if cls._state_0.particle_q is not None:
                        wp.copy(cls._state_1.particle_q, cls._state_0.particle_q)
                    apply_soft_body_reactions(
                        contacts_soft, cls._state_0, cls._model, cls._soft_contact_max,
                        particle_q_prev=cls._state_1.particle_q,
                        friction_epsilon=cls._vbd_solver.friction_epsilon,
                        dt=cls._solver_dt,
                    )
                cls._solver.step(cls._state_0, cls._state_0, cls._control, contacts_rigid, cls._solver_dt)
                cls._vbd_solver.step(
                    cls._state_0, cls._state_0, cls._control, contacts_soft, cls._solver_dt
                )
            else:
                if two_way:
                    apply_soft_body_reactions(
                        contacts_soft, cls._state_0, cls._model, cls._soft_contact_max,
                        particle_q_prev=cls._state_1.particle_q,
                        friction_epsilon=cls._vbd_solver.friction_epsilon,
                        dt=cls._solver_dt,
                    )
                cls._solver.step(cls._state_0, cls._state_1, cls._control, contacts_rigid, cls._solver_dt)
                cls._vbd_solver.step(
                    cls._state_0, cls._state_1, cls._control, contacts_soft, cls._solver_dt
                )
                if need_copy and i == cls._num_substeps - 1:
                    cls._state_0.assign(cls._state_1)
                else:
                    cls._state_0, cls._state_1 = cls._state_1, cls._state_0
                cls._state_0.clear_forces()

        # Update cached pose/velocity — captured inside the CUDA graph.
        cls._refresh_obs_cache()

        if cls._report_contacts:
            eval_contacts = contacts_rigid if contacts_rigid is not None else cls._contacts
            if eval_contacts is not None:
                cls._solver.update_contacts(eval_contacts, cls._state_0)
                for sensor in cls._newton_contact_sensors.values():
                    sensor.update(cls._state_0, eval_contacts)

        if cls._usdrt_stage is not None:
            cls.sync_transforms_to_usd()

    # ------------------------------------------------------------------ #
    # MDP helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def get_object_pose(cls) -> tuple[wp.array, wp.array] | None:
        """Return ``(root_pos_w, root_quat_w)`` for the soft cube per env.

        Position is the particle CoM; orientation is estimated via Kabsch alignment
        of current vs rest-pose particles.  Both are pre-computed every physics step
        by ``_kernel_particle_com_kabsch`` inside the CUDA graph.

        Returns:
            Tuple of ``(pos, quat)`` Warp arrays of shape ``(num_envs,)`` with
            dtypes ``vec3f`` and ``quatf``, or ``None`` if not ready.
        """
        if cls._particle_pos_out is None or cls._particle_quat_out is None:
            return None
        return cls._particle_pos_out, cls._particle_quat_out

    @classmethod
    def get_object_velocity(cls) -> wp.array | None:
        """Return CoM linear velocity per env for the soft cube [m/s].

        Returns:
            Warp array of shape ``(num_envs,)`` with dtype ``vec3f``, or ``None``.
        """
        return cls._particle_vel_out

    @classmethod
    def reset_particles(
        cls,
        env_ids: torch.Tensor,
        root_pose: torch.Tensor,
    ) -> None:
        """Teleport cube particles to match a new root pose after MDP reset.

        Applies the rigid transform:  p_new = T_reset @ T_build_inv @ p_build

        Args:
            env_ids: 1-D tensor of env indices to reset.
            root_pose: ``(N, 7)`` tensor — position (3) + quaternion wxyz (4) [m].
        """
        if cls._particles_per_env is None or cls._state_0 is None:
            return
        if cls._particle_q_build is None or cls._T_build_inv is None:
            return

        pq_dev = cls._state_0.particle_q
        if pq_dev is None or pq_dev.ptr is None:
            return

        device = str(PhysicsManager._device)
        env_ids_t  = torch.as_tensor(env_ids, device=device).long().reshape(-1)
        n          = env_ids_t.shape[0]
        root_pose  = root_pose.to(device=device, dtype=torch.float32)
        ppe        = cls._particles_per_env
        num_envs   = cls._num_envs or 1

        T_inv   = cls._T_build_inv.to(device)           # (num_envs, 4, 4)
        p_build = cls._particle_q_build.to(device)      # (num_envs * ppe, 3)

        R      = matrix_from_quat(root_pose[:, 3:7])    # (n, 3, 3)
        t      = root_pose[:, :3]                        # (n, 3)
        T_reset = torch.eye(4, device=device).unsqueeze(0).expand(n, -1, -1).clone()
        T_reset[:, :3, :3] = R
        T_reset[:, :3, 3]  = t

        delta = T_reset @ T_inv[env_ids_t]               # (n, 4, 4)

        p_build_3d = p_build.view(num_envs, ppe, 3)
        p_sel      = p_build_3d[env_ids_t]               # (n, ppe, 3)

        ones = torch.ones(n, ppe, 1, device=device)
        p_h  = torch.cat([p_sel, ones], dim=-1)          # (n, ppe, 4)
        p_new = (delta @ p_h.transpose(1, 2)).transpose(1, 2)[:, :, :3]  # (n, ppe, 3)

        pq       = wp.to_torch(pq_dev).float().clone()
        pq_3d    = pq.view(num_envs, ppe, 3)
        pq_3d[env_ids_t] = p_new
        pq_flat  = pq_3d.reshape(-1, 3)

        new_pq = wp.from_torch(pq_flat.contiguous(), dtype=wp.vec3f)
        wp.copy(pq_dev, new_pq)
        if cls._state_1 is not None and cls._state_1.particle_q is not None and cls._state_1.particle_q.ptr:
            wp.copy(cls._state_1.particle_q, new_pq)

        cls._particle_rest_q = pq_flat.clone()

        # Keep the Warp mirror of the rest-pose in sync.
        if cls._particle_rest_q_wp is not None:
            rest_wp_new = wp.from_torch(cls._particle_rest_q.contiguous(), dtype=wp.vec3f)
            wp.copy(cls._particle_rest_q_wp, rest_wp_new)

        # Refresh cached pose/velocity so the MDP gets fresh observations.
        cls._refresh_obs_cache()

    @classmethod
    def reset_particle_velocities(cls, env_ids: torch.Tensor) -> None:
        """Zero particle velocities for the given envs after reset."""
        if cls._particles_per_env is None or cls._state_0 is None:
            return
        pqd_dev = getattr(cls._state_0, "particle_qd", None)
        if pqd_dev is None or pqd_dev.ptr is None:
            return

        device    = str(PhysicsManager._device)
        env_ids_t = torch.as_tensor(env_ids, device=device).long().reshape(-1)
        ppe       = cls._particles_per_env
        num_envs  = cls._num_envs or 1

        pqd     = wp.to_torch(pqd_dev).float().clone()
        pqd.view(num_envs, ppe, 3)[env_ids_t] = 0.0
        pqd_flat = pqd.reshape(-1, 3)

        new_pqd = wp.from_torch(pqd_flat.contiguous(), dtype=wp.vec3f)
        wp.copy(pqd_dev, new_pqd)
        if cls._state_1 is not None:
            pqd1 = getattr(cls._state_1, "particle_qd", None)
            if pqd1 is not None and pqd1.ptr:
                wp.copy(pqd1, new_pqd)

    @classmethod
    def clear(cls) -> None:
        cls._vbd_solver = None
        cls._soft_collision_pipeline = None
        cls._soft_contacts = None
        cls._particles_per_env = None
        cls._particle_q_build = None
        cls._T_build_inv = None
        cls._particle_rest_q = None
        cls._particle_rest_q_wp = None
        cls._particle_pos_out = None
        cls._particle_quat_out = None
        cls._particle_vel_out = None
        super().clear()
