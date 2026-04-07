# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extended NewtonManager for dexsuite_3dg_proxy task (VBD soft body mode).

Follows the same pattern as dexsuite_3dg's Dexsuite3dgNewtonManager:
  - Subclasses NewtonManager without modifying it.
  - Patched into the module by the env wrapper before scene construction.
  - When vbd_enabled=True, adds the ragdoll tet mesh as a VBD soft body and
    runs a two-phase step: rigid MuJoCo solver (robot) + VBD solver (soft body).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.physics import PhysicsManager
from isaaclab.utils.math import matrix_from_quat

from .vbd_coupling import apply_soft_body_reactions

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

logger = logging.getLogger("dexsuite_3dg_proxy.newton.manager")

# ---------------------------------------------------------------------------
# Warp kernels — run inside the CUDA graph each step (or eagerly after reset).
# One kernel thread per environment.
# ---------------------------------------------------------------------------

@wp.kernel
def _kernel_particle_com_kabsch(
    particle_q: wp.array(dtype=wp.vec3f),
    particle_rest_q: wp.array(dtype=wp.vec3f),
    particles_per_env: int,
    pos_out: wp.array(dtype=wp.vec3f),
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

    com_cur = wp.vec3(0.0, 0.0, 0.0)
    com_rest = wp.vec3(0.0, 0.0, 0.0)
    for p in range(particles_per_env):
        com_cur = com_cur + particle_q[start + p]
        com_rest = com_rest + particle_rest_q[start + p]
    com_cur = com_cur * inv_n
    com_rest = com_rest * inv_n
    pos_out[env_i] = com_cur

    # Kabsch H = Σ (rest_centred)^T ⊗ (cur_centred)
    H = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for p in range(particles_per_env):
        dr = particle_rest_q[start + p] - com_rest
        dc = particle_q[start + p] - com_cur
        H = H + wp.outer(dr, dc)

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
    particle_qd: wp.array(dtype=wp.vec3f),
    particles_per_env: int,
    vel_out: wp.array(dtype=wp.vec3f),
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


@wp.kernel
def _kernel_fingertip_proximity_contact(
    particle_q: wp.array(dtype=wp.vec3f),
    fingertip_pos: wp.array(dtype=wp.vec3f),
    particles_per_env: int,
    num_tips: int,
    threshold_sq: float,
    signal_magnitude: float,
    contact_out: wp.array(dtype=wp.vec3f),
):
    """Proximity-based contact detection between fingertips and particles.

    One thread per (env, tip) pair. Finds the nearest particle; if closer than
    ``sqrt(threshold_sq)`` the fingertip is marked as in contact.

    Args:
        particle_q: Particle positions [m].
        fingertip_pos: Fingertip world positions [m], shape (num_envs * num_tips,).
        particles_per_env: Number of particles per environment.
        num_tips: Number of fingertips per environment.
        threshold_sq: Squared contact-detection threshold [m²].
        signal_magnitude: Magnitude of the z-component contact signal [N].
        contact_out: Output contact vectors [N], shape (num_envs * num_tips,).
    """
    tid = wp.tid()  # env_i * num_tips + tip_i
    env_i = tid // num_tips
    tip_pos = fingertip_pos[tid]
    start = env_i * particles_per_env
    min_sq = float(1.0e10)
    for p in range(particles_per_env):
        d = tip_pos - particle_q[start + p]
        sq = wp.dot(d, d)
        if sq < min_sq:
            min_sq = sq
    if min_sq < threshold_sq:
        contact_out[tid] = wp.vec3(0.0, 0.0, signal_magnitude)
    else:
        contact_out[tid] = wp.vec3(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENVS_ROOT = "/World/envs"
_OBJECT_RELATIVE_PATH = "Object"


def _vbd_enabled() -> bool:
    cfg = PhysicsManager._cfg
    if not isinstance(cfg, _get_proxy_cfg_class()):
        return False
    return bool(cfg.vbd_enabled and cfg.tet_mesh_path)


def _get_proxy_cfg_class():
    """Lazy import to avoid circular dependency at module load time."""
    from .proxy_newton_cfg import Dexsuite3dgProxyNewtonCfg
    return Dexsuite3dgProxyNewtonCfg


def _discover_env_origins(num_envs: int, device: str) -> torch.Tensor:
    """Return env origins (num_envs, 3) from the USD stage env prims."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Usd

    stage = get_current_stage()
    origins = []
    root = stage.GetPrimAtPath(_ENVS_ROOT)
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


def _load_tet_mesh(path: str) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """Load a tet mesh file and return ``(nodes, tets, mat_params)``.

    Supported formats:

    - ``.msh`` / ``.msh2`` / ``.msh4``: Gmsh format via meshio.
      ``mat_params`` is ``None``; material parameters come from the Python cfg.
    - ``.usd`` / ``.usda``: USD format written by ``vbd_interactive_sim.py``.
      ``mat_params`` is a dict with any subset of the keys
      ``k_mu``, ``k_lambda``, ``k_damp``, ``density``,
      ``soft_contact_ke``, ``soft_contact_kd``, ``soft_contact_mu``.

    Args:
        path: Absolute path to the tet mesh file.

    Returns:
        nodes: Particle positions [m], shape ``(V, 3)``, float32.
        tets: Tetrahedral connectivity, shape ``(T, 4)``, int32.
        mat_params: Material parameter overrides from the file, or ``None``.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".usd", ".usda"):
        return _load_tet_mesh_usd(path)

    try:
        import meshio
    except ImportError:
        raise ImportError("meshio is required for VBD soft body: pip install meshio")

    mesh = meshio.read(path)
    nodes = mesh.points.astype(np.float32)
    tets = mesh.cells_dict.get("tetra")
    if tets is None:
        raise ValueError(f"No tetrahedral cells found in {path}. Run mesh_to_tet.py first.")
    return nodes, tets.astype(np.int32), None


def _load_tet_mesh_usd(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a tet mesh from a ``.usda`` file saved by ``vbd_interactive_sim.py``.

    Reads the ``vbd:vertices``, ``vbd:tet_indices``, and material attributes
    from the ``/TetMesh`` prim.

    Args:
        path: Absolute path to the ``.usd`` or ``.usda`` file.

    Returns:
        nodes: Particle positions [m], shape ``(V, 3)``, float32.
        tets: Tetrahedral connectivity, shape ``(T, 4)``, int32.
        mat_params: Dict of material parameter overrides (always non-empty).
    """
    try:
        from pxr import Usd
    except ImportError:
        raise ImportError("pxr (USD) is required to load .usda tet mesh files")

    stage = Usd.Stage.Open(path)
    prim = stage.GetPrimAtPath("/TetMesh")
    if not prim.IsValid():
        raise ValueError(f"No /TetMesh prim found in {path}")

    verts_vt = prim.GetAttribute("vbd:vertices").Get()
    nodes = np.array([(v[0], v[1], v[2]) for v in verts_vt], dtype=np.float32)

    tets_flat = np.array(list(prim.GetAttribute("vbd:tet_indices").Get()), dtype=np.int32)
    tets = tets_flat.reshape(-1, 4)

    mat_params: dict = {}
    for name in ("k_mu", "k_lambda", "k_damp", "density",
                 "soft_contact_ke", "soft_contact_kd", "soft_contact_mu"):
        attr = prim.GetAttribute(f"vbd:{name}")
        if attr and attr.IsValid():
            mat_params[name] = float(attr.Get())

    return nodes, tets, mat_params


def _transform_points(pts: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Apply 4x4 transform T to (N,3) points."""
    ones = torch.ones(pts.shape[0], 1, device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=1)  # (N, 4)
    return (T @ pts_h.T).T[:, :3]


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class Dexsuite3dgProxyNewtonManager(NewtonManager):
    """Newton manager extended for dexsuite_3dg_proxy (VBD soft body object).

    When :attr:`vbd_enabled` is True (via Dexsuite3dgProxyNewtonCfg):
      - Loads the tet mesh and adds one soft body per environment via add_soft_mesh().
      - Runs two-phase stepping with same-substep **two-way coupling**:
        rigid MuJoCo solver (robot) + VBD solver (soft body).
        Soft-contact reaction forces are injected into ``state.body_f`` before
        the rigid step so the finger joints feel resistance from the deformable
        object in the same substep that VBD applies the contact force to particles.
      - Exposes get_object_pose(), get_object_velocity(), reset_particles(),
        reset_particle_velocities() for the MDP layer.
    """

    _vbd_solver: Any = None
    # Dedicated collision pipeline + contacts buffer for particle-rigid soft contacts.
    # Following the cloth_franka example pattern: pre-allocate once before CUDA graph
    # capture so the same buffer is reused every substep instead of allocating a new
    # Contacts object inside the captured graph.
    _soft_collision_pipeline: Any = None
    _soft_contacts: Any = None
    # Practical upper bound on simultaneous soft contacts (set in initialize_solver).
    # Used as the kernel launch dim for apply_soft_body_reactions so threads
    # beyond the actual contact count early-exit rather than being skipped silently.
    _soft_contact_max: int = 0
    # (num_envs, 2) — start/end particle index per env in state.particle_q
    _per_env_particle_ranges: list[tuple[int, int]] | None = None
    # Number of particles per env — constant since all envs use the same mesh.
    # Stored as a scalar so we can use view(num_envs, _particles_per_env, 3) everywhere
    # instead of looping over per_env_particle_ranges.
    _particles_per_env: int | None = None
    # Build-time particle positions (world space) — shaped (num_envs * particles_per_env, 3)
    _particle_q_build: torch.Tensor | None = None
    # inv(T_build) per env [num_envs, 4, 4]
    _T_build_inv: torch.Tensor | None = None
    # Rest-pose particle positions (updated on each reset) — same shape as _particle_q_build
    _particle_rest_q: torch.Tensor | None = None
    # Warp mirror of _particle_rest_q — kept in sync for use inside Warp kernels.
    _particle_rest_q_wp: Any = None
    # Pre-allocated output arrays written by the obs-cache kernels every physics step.
    # These are captured in the CUDA graph; MDP helpers read them without extra computation.
    _particle_pos_out: Any = None   # wp.array(num_envs, dtype=wp.vec3f) — CoM positions [m]
    _particle_quat_out: Any = None  # wp.array(num_envs, dtype=wp.quatf) — Kabsch orientations
    _particle_vel_out: Any = None   # wp.array(num_envs, dtype=wp.vec3f) — CoM velocities [m/s]
    # Contact output buffer — allocated lazily on first get_fingertip_contact_proxy call.
    _contact_out_wp: Any = None     # wp.array(num_envs * num_tips, dtype=wp.vec3f)

    # ------------------------------------------------------------------ #
    # Lifecycle overrides
    # ------------------------------------------------------------------ #

    @classmethod
    def start_simulation(cls) -> None:
        if _vbd_enabled():
            cls._start_simulation_vbd()
            return
        super().start_simulation()

    @classmethod
    def _start_simulation_vbd(cls) -> None:
        """Add soft body per env to the builder, then call super().start_simulation().

        Fast-path: adds the mesh for env 0 via add_soft_mesh() (to compute tet rest
        poses, surface triangles, and bending edges), then tiles the resulting builder
        data directly for all remaining envs via numpy without calling add_soft_mesh()
        4095 more times.  Graph coloring is done once on the single-env graph (1386
        nodes / ~7400 edges) and tiled — instead of on the full 5.67M-node graph.
        This reduces initialization from O(num_envs²) to O(num_envs + mesh_size).
        """
        cfg = PhysicsManager._cfg
        device = PhysicsManager._device

        nodes_local, tets, mat_override = _load_tet_mesh(cfg.tet_mesh_path)
        logger.info(
            "[Proxy VBD] Loaded tet mesh: %d nodes, %d tets from %s",
            len(nodes_local), len(tets), cfg.tet_mesh_path,
        )

        # Resolve effective material params: USD overrides take precedence over cfg defaults.
        def _mat(key: str) -> float:
            return mat_override[key] if mat_override and key in mat_override else getattr(cfg, key)

        eff_k_mu            = _mat("k_mu")
        eff_k_lambda        = _mat("k_lambda")
        eff_k_damp          = _mat("k_damp")
        eff_density         = _mat("density")
        eff_soft_contact_ke = _mat("soft_contact_ke")
        eff_soft_contact_kd = _mat("soft_contact_kd")
        eff_soft_contact_mu = _mat("soft_contact_mu")

        if mat_override:
            logger.info(
                "[Proxy VBD] USD material overrides: k_mu=%.3g k_lambda=%.3g k_damp=%.3g"
                " density=%.3g ke=%.3g kd=%.3g mu=%.3g",
                eff_k_mu, eff_k_lambda, eff_k_damp, eff_density,
                eff_soft_contact_ke, eff_soft_contact_kd, eff_soft_contact_mu,
            )

        num_envs = cls._num_envs or 1
        n_particles = len(nodes_local)
        n_tets = len(tets)

        init_pos = np.array([-0.55, 0.1, 0.35], dtype=np.float32)

        env_origins_t = _discover_env_origins(num_envs, device)
        env_origins = env_origins_t.cpu().numpy()

        existing_particles = getattr(cls._builder, "particle_count", 0) if cls._builder else 0

        # ------------------------------------------------------------------ #
        # Step 1: add env-0 via the standard path so Newton computes tet
        #         rest poses (inv_Dm) and surface triangles / bending edges.
        # ------------------------------------------------------------------ #
        world_pos_0 = env_origins[0] + init_pos
        vertices_list = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in nodes_local]
        indices_list = [int(i) for i in tets.flatten()]

        snap_before = {
            "particle_q": len(cls._builder.particle_q),
            "tet_indices": len(cls._builder.tet_indices),
            "tet_poses": len(cls._builder.tet_poses),
            "tet_materials": len(cls._builder.tet_materials),
            "tet_activations": len(cls._builder.tet_activations),
            "tri_indices": len(cls._builder.tri_indices),
            "tri_poses": len(cls._builder.tri_poses),
            "tri_materials": len(cls._builder.tri_materials),
            "edge_indices": len(cls._builder.edge_indices),
            "edge_bending_properties": len(cls._builder.edge_bending_properties),
        }

        cls._builder.add_soft_mesh(
            pos=wp.vec3(*world_pos_0.tolist()),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices_list,
            indices=indices_list,
            density=eff_density,
            k_mu=eff_k_mu,
            k_lambda=eff_k_lambda,
            k_damp=eff_k_damp,
            add_surface_mesh_edges=True,
            particle_radius=cfg.particle_radius,
        )

        # Snapshot of what env-0 contributed (as numpy arrays for fast tiling)
        n_tris_per_env = len(cls._builder.tri_indices) - snap_before["tri_indices"]
        n_edges_per_env = (len(cls._builder.edge_indices) - snap_before["edge_indices"]) if hasattr(cls._builder, "edge_indices") else 0
        tets_env0 = np.array(cls._builder.tet_indices[snap_before["tet_indices"]:], dtype=np.int32)
        tet_poses_env0 = list(cls._builder.tet_poses[snap_before["tet_poses"]:])
        tet_mats_env0 = list(cls._builder.tet_materials[snap_before["tet_materials"]:])
        tet_acts_env0 = list(cls._builder.tet_activations[snap_before["tet_activations"]:])

        tri_env0 = np.array(cls._builder.tri_indices[snap_before["tri_indices"]:], dtype=np.int32)
        tri_mats_env0 = list(cls._builder.tri_materials[snap_before["tri_materials"]:])
        tri_poses_env0 = list(cls._builder.tri_poses[snap_before["tri_poses"]:])

        edges_env0 = (
            np.array(cls._builder.edge_indices[snap_before["edge_indices"]:], dtype=np.int32)
            if n_edges_per_env > 0 else None
        )
        edge_bends_env0 = (
            list(cls._builder.edge_bending_properties[snap_before["edge_bending_properties"]:])
            if n_edges_per_env > 0 else None
        )

        # Particle data for env-0 (positions + per-particle scalars)
        pq_env0 = np.array(cls._builder.particle_q[snap_before["particle_q"]:], dtype=np.float32)
        pqd_env0 = np.array(cls._builder.particle_qd[snap_before["particle_q"]:], dtype=np.float32)
        pm_env0 = list(cls._builder.particle_mass[snap_before["particle_q"]:])
        pr_env0 = list(cls._builder.particle_radius[snap_before["particle_q"]:])
        pf_env0 = list(cls._builder.particle_flags[snap_before["particle_q"]:])

        logger.info(
            "[Proxy VBD] Env-0 mesh: %d particles, %d tets, %d tris, %d edges",
            n_particles, n_tets, n_tris_per_env, n_edges_per_env,
        )

        # ------------------------------------------------------------------ #
        # Step 2: tile envs 1..N-1 directly via list extend — no per-tet
        #         Python loops, no matrix inversions, no surface detection.
        # ------------------------------------------------------------------ #
        per_env_ranges = [(existing_particles, existing_particles + n_particles)]

        # Pre-compute all env world positions: (num_envs-1, 3)
        all_world_pos = env_origins[1:] + init_pos   # (num_envs-1, 3)
        all_deltas = all_world_pos - world_pos_0      # (num_envs-1, 3)

        # All particle positions for envs 1..N in one numpy op: (num_envs-1, n_particles, 3)
        # Then convert to flat wp.vec3 list all at once.
        all_pq_new = pq_env0[None, :, :] + all_deltas[:, None, :]  # broadcast
        all_pq_flat = all_pq_new.reshape(-1, 3)  # ((num_envs-1)*n_particles, 3)
        # Build the wp.vec3 list once (fastest path: map over numpy rows)
        extra_particles = [wp.vec3(float(r[0]), float(r[1]), float(r[2])) for r in all_pq_flat]
        extra_pqd = list(pqd_env0) * (num_envs - 1)
        # Scalars that are identical per env
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

        # Build index offsets for envs 1..N-1: (num_envs-1,)
        env_offsets = existing_particles + np.arange(1, num_envs, dtype=np.int32) * n_particles

        # Tetrahedra: tile all envs at once with broadcasting
        # tets_env0: (n_tets, 4), env_offsets: (num_envs-1,)
        # all_tets: (num_envs-1, n_tets, 4) = tets_env0 + offset per env
        all_tets = tets_env0[None, :, :] + env_offsets[:, None, None]  # broadcast
        cls._builder.tet_indices.extend(map(tuple, all_tets.reshape(-1, 4).tolist()))
        cls._builder.tet_poses.extend(tet_poses_env0 * (num_envs - 1))
        cls._builder.tet_materials.extend(tet_mats_env0 * (num_envs - 1))
        cls._builder.tet_activations.extend(tet_acts_env0 * (num_envs - 1))

        # Surface triangles
        if n_tris_per_env > 0:
            all_tris = tri_env0[None, :, :] + env_offsets[:, None, None]
            cls._builder.tri_indices.extend(map(tuple, all_tris.reshape(-1, 3).tolist()))
            cls._builder.tri_materials.extend(tri_mats_env0 * (num_envs - 1))
            if tri_poses_env0:
                cls._builder.tri_poses.extend(tri_poses_env0 * (num_envs - 1))

        # Bending edges
        if n_edges_per_env > 0 and edges_env0 is not None:
            all_edges = edges_env0[None, :, :] + env_offsets[:, None, None]
            cls._builder.edge_indices.extend(map(tuple, all_edges.reshape(-1, 4).tolist()))
            cls._builder.edge_bending_properties.extend(edge_bends_env0 * (num_envs - 1))

        for env_idx in range(1, num_envs):
            offset = existing_particles + env_idx * n_particles
            per_env_ranges.append((offset, offset + n_particles))

        cls._per_env_particle_ranges = per_env_ranges
        cls._particles_per_env = n_particles

        # Build all 4×4 translation matrices in one batched op
        all_world_pos_full = np.vstack([world_pos_0[None, :], all_world_pos])  # (num_envs, 3)
        T_all = np.eye(4, dtype=np.float32)[None].repeat(num_envs, axis=0)     # (num_envs, 4, 4)
        T_all[:, :3, 3] = all_world_pos_full
        T_stacked = torch.from_numpy(T_all).to(device)
        cls._T_build_inv = torch.linalg.inv(T_stacked.double()).float()

        # ------------------------------------------------------------------ #
        # Step 3: graph-color the single-env topology, then tile the result.
        #
        # The full graph (all envs concatenated) has no inter-env edges, so
        # its optimal coloring is identical to the single-env coloring tiled
        # with a per-env particle-index offset.  This is O(single_env_size)
        # vs O(total_particle_count) for the naive call to builder.color().
        # ------------------------------------------------------------------ #
        logger.info("[Proxy VBD] Coloring single-env graph (%d nodes)...", n_particles)
        from newton._src.sim.graph_coloring import (
            color_graph,
            construct_particle_graph,
            ColoringAlgorithm,
        )

        tet_np = tets_env0 - existing_particles  # local (0-based) indices for env-0
        tri_np = tri_env0 - existing_particles if n_tris_per_env > 0 else None

        graph_edges = construct_particle_graph(
            tri_np,
            None,  # tri_active_mask — not needed for plain soft mesh
            None,  # bending_edge_indices
            None,
            tet_np,
            None,  # tet_active_mask
        )

        single_env_colors: list[np.ndarray] = color_graph(
            n_particles,
            graph_edges,
            balance_colors=True,
            target_max_min_color_ratio=1.1,
            algorithm=ColoringAlgorithm.MCS,
        )
        logger.info("[Proxy VBD] Single-env coloring: %d colors", len(single_env_colors))

        # Tile: for each color group, stack all env replicas
        tiled_colors = []
        for color_group in single_env_colors:
            # color_group: local indices for env-0
            # Tile across all envs: each env adds (env_idx * n_particles) offset
            tiled = np.concatenate([
                color_group + existing_particles + env_idx * n_particles
                for env_idx in range(num_envs)
            ])
            tiled_colors.append(tiled)

        cls._builder.set_coloring(tiled_colors)
        # Also color rigid bodies (unchanged — handled by body topology)
        from newton._src.sim.graph_coloring import color_rigid_bodies
        cls._builder.body_color_groups = color_rigid_bodies(
            cls._builder.body_count,
            cls._builder.joint_parent,
            cls._builder.joint_child,
        )

        super().start_simulation()

        # Cache build-time particle positions
        pq = wp.to_torch(cls._state_0.particle_q).float().clone()
        cls._particle_q_build = pq.clone()
        cls._particle_rest_q = pq.clone()

        # Log gravity and particle count after finalization
        if cls._model is not None:
            grav = cls._model.gravity.numpy().tolist() if hasattr(cls._model.gravity, "numpy") else cls._model.gravity
            logger.info("[Proxy VBD] Model gravity=%s  particle_count=%d", grav, cls._model.particle_count)
            inv_m = wp.to_torch(cls._model.particle_inv_mass).float()
            logger.info("[Proxy VBD] particle_inv_mass[0:5]=%s", inv_m[:5].cpu().tolist())

        # Set contact parameters on the finalised model
        if cls._model is not None:
            cls._model.soft_contact_ke = eff_soft_contact_ke
            cls._model.soft_contact_kd = eff_soft_contact_kd
            cls._model.soft_contact_mu = eff_soft_contact_mu
            logger.info(
                "[Proxy VBD] Contact params: ke=%.1f kd=%.1f mu=%.2f",
                eff_soft_contact_ke, eff_soft_contact_kd, eff_soft_contact_mu,
            )

    @classmethod
    def initialize_solver(cls) -> None:
        # IMPORTANT: create VBD solver BEFORE super().initialize_solver() so that when the
        # base class captures the CUDA graph it calls _simulate_two_phase (not the rigid-only
        # base _simulate).  The CUDA graph is captured inside super().initialize_solver() by
        # calling cls._simulate(); if cls._vbd_solver is None at that point the check
        # `_vbd_enabled() and cls._vbd_solver is not None` is False and VBD is skipped.
        if _vbd_enabled() and cls._model is not None:
            try:
                from newton.solvers import SolverVBD
            except ImportError:
                raise ImportError("newton.solvers.SolverVBD not found — update Newton.")
            cfg = PhysicsManager._cfg

            # Pre-allocate pipeline + contacts buffer once (cloth_franka pattern) so
            # the same buffer is reused every substep instead of allocating inside
            # the CUDA graph.
            #
            # Compute per-env counts for the world-batched collision kernel.
            # Particles in env N can only contact shapes in env N (different Newton
            # worlds), so the effective kernel dim is:
            #   particles_per_env × shapes_per_env × num_envs
            # instead of the full cross-product which grows as num_worlds².
            # At 4096 envs:  full = 49×4096 × 1386×4096 ≈ 1.14×10¹²  (overflow)
            #                batched = 49 × 1386 × 4096 ≈ 278M         (safe)
            # The same reduction applies to the SolverVBD AVBD state arrays.
            num_envs = len(cls._per_env_particle_ranges) if cls._per_env_particle_ranges else 1
            particles_per_env = cls._model.particle_count // max(num_envs, 1)
            shapes_per_env = cls._model.shape_count // max(num_envs, 1)

            # Practical contact buffer cap.  The theoretical worst case is
            # particles_per_env × shapes_per_env × num_envs (all pairs in contact),
            # but in practice only ~50-200 contacts/env occur.  Capping at a practical
            # max reduces the VBD contact kernel launch dim from ~69M → ~512K threads
            # at 1024 envs (135× fewer wasted threads per kernel launch).
            # If the actual count ever exceeds this, contacts are silently dropped →
            # increase vbd_max_contacts_per_env in config if tunneling appears.
            max_contacts_per_env = getattr(cfg, "vbd_max_contacts_per_env", 500)
            practical_contact_max = min(
                max_contacts_per_env * num_envs,
                shapes_per_env * particles_per_env * num_envs,
            )
            logger.info(
                "[Proxy VBD] soft_contact_max=%d (practical cap: %d/env × %d envs; "
                "theoretical max=%d × %d × %d).",
                practical_contact_max,
                max_contacts_per_env, num_envs,
                shapes_per_env, particles_per_env, num_envs,
            )

            if cls._state_0 is not None:
                try:
                    import newton as _newton_pkg
                    soft_margin = cfg.particle_radius * 3.0
                    # Use the world-batched CollisionPipeline so create_soft_contacts_batched
                    # launches with dim = particles_per_world × shapes_per_world × num_worlds
                    # (avoiding int32 overflow at 4096 envs).
                    # soft_contact_max is set to the practical cap so VBD contact kernels
                    # launch with that smaller dim rather than the full P×S×W product.
                    def _make_collision_pipeline(spw: int, contact_max: int):
                        logger.info(
                            "[Proxy VBD] Creating soft-body CollisionPipeline "
                            "(batched: particles_per_world=%d, shapes_per_world=%d, "
                            "num_worlds=%d, soft_contact_max=%d, soft_contact_margin=%.4f m)...",
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
                        # Warm up outside CUDA graph capture (first call does d2h
                        # shape-type copies that are illegal inside capture).
                        pipeline.collide(cls._state_0, contacts)
                        return pipeline, contacts

                    cls._soft_collision_pipeline, cls._soft_contacts = _make_collision_pipeline(
                        shapes_per_env, practical_contact_max
                    )

                    # Auto-detect or use override for tight shapes_per_world.
                    shapes_per_env = cls._auto_detect_shapes_per_world(
                        cls._soft_contacts, shapes_per_env, num_envs,
                        getattr(cfg, "vbd_shapes_per_world", None),
                    )

                    # Rebuild with tighter shapes if auto-detect found a smaller value.
                    if shapes_per_env < (cls._model.shape_count // max(num_envs, 1)):
                        cls._soft_collision_pipeline, cls._soft_contacts = _make_collision_pipeline(
                            shapes_per_env, practical_contact_max
                        )

                    logger.info("[Proxy VBD] Soft-body collision pipeline ready.")
                except Exception as exc:
                    logger.warning("[Proxy VBD] CollisionPipeline setup failed: %s", exc)
                    cls._soft_collision_pipeline = None
                    cls._soft_contacts = None

            cls._vbd_solver = SolverVBD(
                cls._model,
                iterations=cfg.vbd_iterations,
                integrate_with_external_rigid_solver=True,
                particle_enable_self_contact=False,
                max_soft_contacts=practical_contact_max,
                particle_max_velocity=cfg.vbd_max_particle_velocity,
            )
            cls._soft_contact_max = practical_contact_max
            logger.info(
                "[Proxy VBD] SolverVBD initialized (iterations=%d, max_soft_contacts=%d, max_vel=%.1f m/s).",
                cfg.vbd_iterations, practical_contact_max, cfg.vbd_max_particle_velocity,
            )

            # Allocate persistent obs-cache arrays (pos, quat, vel).  These are
            # written by Warp kernels inside the CUDA graph every step, so the
            # MDP helpers (get_object_pose / get_object_velocity) can return them
            # without any extra PyTorch ops, eliminating the 90K cudaMemcpyAsync
            # calls that dominated the training loop.
            if cls._state_0 is not None and cls._particles_per_env is not None:
                device_str = str(PhysicsManager._device)
                n_obs_envs = num_envs
                ppe = cls._particles_per_env
                cls._particle_pos_out = wp.zeros(n_obs_envs, dtype=wp.vec3f, device=device_str)
                cls._particle_quat_out = wp.zeros(n_obs_envs, dtype=wp.quatf, device=device_str)
                cls._particle_vel_out = wp.zeros(n_obs_envs, dtype=wp.vec3f, device=device_str)
                # Warp-owned copy of the rest-pose positions (torch tensor → Warp array).
                rest_np = cls._particle_rest_q[:n_obs_envs * ppe].view(-1, 3).cpu().numpy()
                cls._particle_rest_q_wp = wp.array(rest_np, dtype=wp.vec3f, device=device_str)
                logger.info("[Proxy VBD] Obs-cache arrays allocated (%d envs).", n_obs_envs)

        super().initialize_solver()  # creates rigid solver + captures CUDA graph

    @classmethod
    def _auto_detect_shapes_per_world(
        cls,
        contacts,
        shapes_per_env: int,
        num_envs: int,
        override: int | None,
    ) -> int:
        """Return a tight shapes_per_world value based on observed contact shape indices.

        Scans the post-warmup contact list to find the highest local shape index that
        actually appears.  Kuka arm shapes (detailed convex-hull decompositions that
        never touch the doll) inflate the kernel launch dim to ~1386/env by default;
        after tightening this can drop to ~50/env, giving ~27× fewer threads in every
        VBD contact kernel and a 3-5× overall VBD speedup.

        If *override* is set (via ``vbd_shapes_per_world`` in config), that value is
        used directly and the scan is skipped.  The detected value is always logged so
        the user can hard-code it after the first run.
        """
        if override is not None:
            logger.info("[Proxy VBD] shapes_per_world override=%d (from config).", override)
            return min(override, shapes_per_env)

        try:
            count_t = wp.to_torch(contacts.soft_contact_count)
            count = int(count_t.item())
            if count == 0:
                logger.info(
                    "[Proxy VBD] No contacts in warmup — keeping shapes_per_world=%d. "
                    "Set vbd_shapes_per_world in config after observing contacts.",
                    shapes_per_env,
                )
                return shapes_per_env

            shape_t = wp.to_torch(contacts.soft_contact_shape)[:count]
            max_global = int(shape_t.max().item())
            max_local = max_global % shapes_per_env

            # 4× safety margin so shapes contacted after warmup are still covered.
            tight = max(min((max_local + 1) * 4, shapes_per_env), 1)
            speedup = shapes_per_env / tight
            logger.info(
                "[Proxy VBD] contact shape range: max local index=%d → "
                "tight shapes_per_world=%d (full=%d, %.1f× smaller). "
                "To hard-code, set vbd_shapes_per_world=%d in config.",
                max_local, tight, shapes_per_env, speedup, tight,
            )
            return tight
        except Exception as exc:
            logger.warning(
                "[Proxy VBD] _auto_detect_shapes_per_world failed (%s) — keeping %d.",
                exc, shapes_per_env,
            )
            return shapes_per_env

    @classmethod
    def _refresh_obs_cache(cls) -> None:
        """Launch Warp kernels to update cached pose/velocity from current particle state.

        Called inside the CUDA graph (captured in ``_simulate_two_phase``) so the
        output arrays are refreshed every physics step at graph-replay speed.
        Also called eagerly from ``reset_particles`` to ensure fresh observations
        immediately after environment resets.
        """
        if cls._particle_pos_out is None or cls._state_0 is None or cls._particles_per_env is None:
            return
        num_envs = len(cls._per_env_particle_ranges)
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
        if _vbd_enabled() and cls._vbd_solver is not None:
            cls._simulate_two_phase()
            return
        super()._simulate()

    @classmethod
    def _simulate_two_phase(cls) -> None:
        """Two-phase step: rigid MuJoCo (robot) + VBD (soft body) with same-substep two-way coupling.

        Substep order (matches validate_grasp.py simulate()):
          1. clear_forces()              — zero state_0.body_f
          2. collide(s0, soft_contacts)  — detect particle-rigid contacts at current positions
          3. apply_soft_body_reactions   — write reaction forces into state_0.body_f
          4. rigid.step(s0 → s1)         — reads body_f; writes new rigid positions into s1
          5. vbd.step(s0 → s1)           — uses same contacts; reads s1.body_q for contact geometry
          6. swap / assign

        The contact detection (step 2) is moved before the rigid step so that the reaction
        forces are ready in body_f when MuJoCo.step() is called (same-substep coupling,
        no time lag).  VBD then uses the same contact buffer for its own force computation,
        ensuring the action–reaction pair is computed from identical contact geometry.
        """
        if cls._needs_collision_pipeline:
            cls._collision_pipeline.collide(cls._state_0, cls._contacts)
            contacts_rigid = cls._contacts
        else:
            contacts_rigid = None

        cfg = PhysicsManager._cfg
        need_copy = getattr(cfg, "use_cuda_graph", False) and cls._num_substeps % 2 == 1

        for i in range(cls._num_substeps):
            # --- Soft contact detection (must precede the rigid step) -----------
            # Detect particle-rigid contacts from the current state.  Using the
            # pre-allocated pipeline/buffer (cloth_franka pattern) so the same
            # Contacts object is reused every substep inside the CUDA graph.
            if cls._soft_collision_pipeline is not None:
                cls._soft_collision_pipeline.collide(cls._state_0, cls._soft_contacts)
                contacts_soft = cls._soft_contacts
            else:
                contacts_soft = cls._model.collide(cls._state_0)

            two_way = getattr(cfg, "vbd_two_way_coupling", True)

            if cls._use_single_state:
                # Single-buffer mode: s0 == s1, both ops write in-place.
                # state_1.particle_q is unused in this mode — repurpose it as a
                # particle_q_prev scratch buffer so the friction coupling has
                # access to the previous substep's particle positions.
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
                # Two-buffer mode: s0 is the input state, s1 is the output state.
                # After each swap, state_1.particle_q holds the previous substep's
                # particle positions — pass it directly as particle_q_prev.
                if two_way:
                    apply_soft_body_reactions(
                        contacts_soft, cls._state_0, cls._model, cls._soft_contact_max,
                        particle_q_prev=cls._state_1.particle_q,
                        friction_epsilon=cls._vbd_solver.friction_epsilon,
                        dt=cls._solver_dt,
                    )
                # Step 1: rigid solver reads body_f from s0, writes new state to s1.
                cls._solver.step(cls._state_0, cls._state_1, cls._control, contacts_rigid, cls._solver_dt)
                # Step 2: VBD uses the same contacts (same-substep two-way coupling);
                #   reads s1.body_q (updated rigid positions) for contact geometry.
                cls._vbd_solver.step(
                    cls._state_0, cls._state_1, cls._control, contacts_soft, cls._solver_dt
                )
                # Step 3: advance buffers — s0 takes the fully updated state.
                if need_copy and i == cls._num_substeps - 1:
                    cls._state_0.assign(cls._state_1)
                else:
                    cls._state_0, cls._state_1 = cls._state_1, cls._state_0
                # Zero body_f on the new s0 so the next substep starts clean.
                cls._state_0.clear_forces()

        # Update cached pose/velocity — captured in the CUDA graph so this runs
        # at near-zero cost during replay and eliminates per-step PyTorch obs ops.
        cls._refresh_obs_cache()

        if cls._report_contacts:
            eval_contacts = contacts_rigid if contacts_rigid is not None else cls._contacts
            if eval_contacts is not None:
                cls._solver.update_contacts(eval_contacts, cls._state_0)
                for sensor in cls._newton_contact_sensors.values():
                    sensor.update(cls._state_0, eval_contacts)

        if cls._usdrt_stage is not None:
            cls.sync_transforms_to_usd()

    @classmethod
    def _sync_vbd_visual_to_usd(cls) -> None:
        """Write per-env particle CoM to the Object USD prim transforms for visualization."""
        if cls._usdrt_stage is None or cls._particles_per_env is None or cls._state_0 is None:
            return
        pq_wp = cls._state_0.particle_q
        if pq_wp is None or pq_wp.ptr is None:
            return
        try:
            import usdrt
            num_envs = len(cls._per_env_particle_ranges)
            ppe = cls._particles_per_env
            # Batch-compute all CoMs in one GPU call instead of per-env mean
            pq = wp.to_torch(pq_wp).float()[:num_envs * ppe].view(num_envs, ppe, 3)
            coms = pq.mean(dim=1).cpu()  # (num_envs, 3) — single d2h transfer
            for env_idx in range(num_envs):
                prim_path = f"/World/envs/env_{env_idx}/Object"
                prim = cls._usdrt_stage.GetPrimAtPath(prim_path)
                if prim and prim.IsValid():
                    xform_attr = prim.GetAttribute("xformOp:translate")
                    if xform_attr:
                        c = coms[env_idx]
                        xform_attr.Set(usdrt.Gf.Vec3d(float(c[0]), float(c[1]), float(c[2])))
        except Exception as exc:
            logger.debug("[Proxy VBD] _sync_vbd_visual_to_usd: %s", exc)

    # ------------------------------------------------------------------ #
    # MDP helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def get_fingertip_contact_proxy(
        cls,
        fingertip_pos_w: torch.Tensor,
        contact_threshold: float,
        signal_magnitude: float = 1.0,
    ) -> torch.Tensor:
        """Proximity-based contact signal for fingertip↔particle contact.

        Replaces PhysX contact sensors in VBD mode.  For each fingertip and each
        environment, the minimum distance to any particle in that env is computed.
        If the distance is below ``contact_threshold``, the fingertip is considered
        to be in contact with the soft object.

        Args:
            fingertip_pos_w: Fingertip world positions, shape ``(num_envs, num_tips, 3)``.
            contact_threshold: Distance below which a particle is considered in contact [m].
                               Typically ``particle_radius * 2``.
            signal_magnitude: Magnitude of the returned "force" signal when in contact.
                              Returned as the z-component of a vec3 to match the shape
                              expected by ``fingers_contact_force_b``.

        Returns:
            Tensor of shape ``(num_envs, num_tips, 3)`` — a proxy "contact force"
            vector where the z-component equals ``signal_magnitude`` when in contact
            and 0 otherwise.
        """
        if cls._particles_per_env is None or cls._state_0 is None:
            return torch.zeros_like(fingertip_pos_w)
        pq_wp = cls._state_0.particle_q
        if pq_wp is None or pq_wp.ptr is None:
            return torch.zeros_like(fingertip_pos_w)

        num_envs = fingertip_pos_w.shape[0]
        num_tips = fingertip_pos_w.shape[1]
        ppe = cls._particles_per_env
        dev = fingertip_pos_w.device

        total = num_envs * num_tips
        # Allocate (or reuse) persistent contact output buffer.
        if cls._contact_out_wp is None or cls._contact_out_wp.shape[0] != total:
            cls._contact_out_wp = wp.zeros(total, dtype=wp.vec3f, device=str(dev))

        # Zero-copy wrap of fingertip positions — (num_envs * num_tips, 3) float32.
        tips_flat = fingertip_pos_w.reshape(-1, 3).contiguous()
        tips_wp = wp.from_torch(tips_flat, dtype=wp.vec3f)

        wp.launch(
            _kernel_fingertip_proximity_contact,
            dim=total,
            inputs=[
                pq_wp,
                tips_wp,
                ppe,
                num_tips,
                float(contact_threshold ** 2),
                float(signal_magnitude),
                cls._contact_out_wp,
            ],
        )

        # Zero-copy view back to torch: (num_envs, num_tips, 3)
        return wp.to_torch(cls._contact_out_wp).view(num_envs, num_tips, 3)

    @classmethod
    def get_object_pose(cls) -> tuple[wp.array, wp.array] | None:
        """Return (root_pos_w, root_quat_w) for the soft object per env.

        Position is the particle CoM; orientation is estimated via Kabsch alignment
        of current vs rest-pose particles.  Both are pre-computed every physics step
        by ``_kernel_particle_com_kabsch`` inside the CUDA graph, so this method is
        a free array-view lookup with no GPU work.

        Returns:
            Tuple of ``(pos, quat)`` Warp arrays of shape ``(num_envs,)``, or
            ``None`` if VBD is not enabled or the obs cache is not ready.
        """
        if cls._particle_pos_out is None or cls._particle_quat_out is None:
            return None
        return cls._particle_pos_out, cls._particle_quat_out

    @classmethod
    def get_object_velocity(cls) -> wp.array | None:
        """Return CoM linear velocity per env for the soft object.

        Pre-computed every physics step by ``_kernel_particle_com_vel`` inside the
        CUDA graph — this is a free array-view lookup.

        Returns:
            Warp array of shape ``(num_envs,)`` with dtype ``vec3f`` [m/s], or
            ``None`` if the obs cache is not ready.
        """
        return cls._particle_vel_out

    @classmethod
    def reset_particles(
        cls,
        env_ids: torch.Tensor,
        root_pose: torch.Tensor,
    ) -> None:
        """Teleport particles to match a new root pose after MDP reset.

        Applies the rigid transform:  p_new = T_reset @ T_build_inv @ p_build

        Args:
            env_ids: 1-D tensor of env indices to reset.
            root_pose: (N, 7) tensor — position (3) + quaternion wxyz (4) in world frame.
        """
        if not _vbd_enabled() or cls._particles_per_env is None:
            return
        if cls._state_0 is None or cls._particle_q_build is None or cls._T_build_inv is None:
            return

        pq_dev = cls._state_0.particle_q
        if pq_dev is None or pq_dev.ptr is None:
            return

        device = str(PhysicsManager._device)
        env_ids_t = torch.as_tensor(env_ids, device=device).long().reshape(-1)
        n = env_ids_t.shape[0]
        root_pose = root_pose.to(device=device, dtype=torch.float32)
        ppe = cls._particles_per_env
        num_envs = len(cls._per_env_particle_ranges)

        T_inv = cls._T_build_inv.to(device)       # (num_envs, 4, 4)
        p_build = cls._particle_q_build.to(device) # (num_envs * ppe, 3)

        # Build per-reset-env 4×4 transform
        R = matrix_from_quat(root_pose[:, 3:7])   # (n, 3, 3)
        t = root_pose[:, :3]                        # (n, 3)
        T_reset = torch.eye(4, device=device).unsqueeze(0).expand(n, -1, -1).clone()
        T_reset[:, :3, :3] = R
        T_reset[:, :3, 3] = t

        # Compute per-env delta transforms in batch: (n, 4, 4)
        delta = T_reset @ T_inv[env_ids_t]

        # Gather build-time particles for the selected envs: (n, ppe, 3)
        p_build_3d = p_build.view(num_envs, ppe, 3)
        p_sel = p_build_3d[env_ids_t]  # (n, ppe, 3)

        # Batch homogeneous transform: (n, 4, 4) @ (n, 4, ppe) → (n, 4, ppe)
        ones = torch.ones(n, ppe, 1, device=device)
        p_h = torch.cat([p_sel, ones], dim=-1)                        # (n, ppe, 4)
        p_new = (delta @ p_h.transpose(1, 2)).transpose(1, 2)[:, :, :3]  # (n, ppe, 3)

        # Scatter back into the flat particle array
        pq = wp.to_torch(pq_dev).float().clone()
        pq_3d = pq.view(num_envs, ppe, 3)
        pq_3d[env_ids_t] = p_new
        pq_flat = pq_3d.reshape(-1, 3)

        new_pq = wp.from_torch(pq_flat.contiguous(), dtype=wp.vec3f)
        wp.copy(pq_dev, new_pq)
        if cls._state_1 is not None and cls._state_1.particle_q is not None and cls._state_1.particle_q.ptr:
            wp.copy(cls._state_1.particle_q, new_pq)

        cls._particle_rest_q = pq_flat.clone()

        # Keep the Warp mirror of the rest-pose in sync so the obs kernels
        # (and the CUDA graph) see the updated rest positions for Kabsch.
        if cls._particle_rest_q_wp is not None:
            rest_wp_new = wp.from_torch(cls._particle_rest_q.contiguous(), dtype=wp.vec3f)
            wp.copy(cls._particle_rest_q_wp, rest_wp_new)

        # Refresh cached pose/velocity eagerly so the MDP gets fresh observations
        # from the reset state before the next CUDA graph replay.
        cls._refresh_obs_cache()

    @classmethod
    def reset_particle_velocities(cls, env_ids: torch.Tensor) -> None:
        """Zero particle velocities for the given envs after reset."""
        if not _vbd_enabled() or cls._particles_per_env is None or cls._state_0 is None:
            return
        pqd_dev = getattr(cls._state_0, "particle_qd", None)
        if pqd_dev is None or pqd_dev.ptr is None:
            return

        device = str(PhysicsManager._device)
        env_ids_t = torch.as_tensor(env_ids, device=device).long().reshape(-1)
        ppe = cls._particles_per_env
        num_envs = len(cls._per_env_particle_ranges)

        pqd = wp.to_torch(pqd_dev).float().clone()
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
        cls._per_env_particle_ranges = None
        cls._particles_per_env = None
        cls._particle_q_build = None
        cls._T_build_inv = None
        cls._particle_rest_q = None
        cls._particle_rest_q_wp = None
        cls._particle_pos_out = None
        cls._particle_quat_out = None
        cls._particle_vel_out = None
        cls._contact_out_wp = None
        super().clear()
