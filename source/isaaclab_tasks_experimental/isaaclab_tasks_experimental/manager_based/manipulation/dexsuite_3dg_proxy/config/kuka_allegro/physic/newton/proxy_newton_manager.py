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
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.physics import PhysicsManager
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

logger = logging.getLogger("dexsuite_3dg_proxy.newton.manager")

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


def _load_tet_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a Gmsh .msh file and return (nodes [V,3] float32, tets [T,4] int32)."""
    try:
        import meshio
    except ImportError:
        raise ImportError("meshio is required for VBD soft body: pip install meshio")

    mesh = meshio.read(path)
    nodes = mesh.points.astype(np.float32)
    tets = mesh.cells_dict.get("tetra")
    if tets is None:
        raise ValueError(f"No tetrahedral cells found in {path}. Run mesh_to_tet.py first.")
    return nodes, tets.astype(np.int32)


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
      - Runs two-phase stepping: rigid MuJoCo solver (robot) + VBD solver (soft body).
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
    # (num_envs, 2) — start/end particle index per env in state.particle_q
    _per_env_particle_ranges: list[tuple[int, int]] | None = None
    # Build-time particle positions (world space) — used for reset teleport
    _particle_q_build: torch.Tensor | None = None
    # inv(T_build) per env [num_envs, 4, 4]
    _T_build_inv: torch.Tensor | None = None
    # Rest-pose particle positions (updated on each reset) — used for Kabsch orientation
    _particle_rest_q: torch.Tensor | None = None

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
        """Add soft body per env to the builder, then call super().start_simulation()."""
        cfg = PhysicsManager._cfg
        device = PhysicsManager._device

        nodes_local, tets = _load_tet_mesh(cfg.tet_mesh_path)
        logger.info(
            "[Proxy VBD] Loaded tet mesh: %d nodes, %d tets from %s",
            len(nodes_local), len(tets), cfg.tet_mesh_path,
        )

        # We need num_envs before the builder is finalised — read from cfg
        # (NewtonManager._num_envs is set during initialize(), which ran before start_simulation)
        num_envs = cls._num_envs or 1

        # Object initial position from SceneCfg (same offset used for rigid object)
        # Default matches RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35))
        init_pos = np.array([-0.55, 0.1, 0.35], dtype=np.float32)

        env_origins_t = _discover_env_origins(num_envs, device)
        env_origins = env_origins_t.cpu().numpy()

        per_env_ranges = []
        T_build_list = []
        particle_offset = 0

        # Count existing particles in the builder before we add soft bodies
        existing_particles = getattr(cls._builder, "particle_count", 0) if cls._builder else 0
        particle_offset = existing_particles

        vertices_list = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in nodes_local]
        indices_list = [int(i) for i in tets.flatten()]

        for env_idx in range(num_envs):
            origin = env_origins[env_idx]
            world_pos = origin + init_pos  # world position of object CoM at build time

            pos_wp = wp.vec3(float(world_pos[0]), float(world_pos[1]), float(world_pos[2]))
            rot_wp = wp.quat_identity()

            cls._builder.add_soft_mesh(
                pos=pos_wp,
                rot=rot_wp,
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=vertices_list,
                indices=indices_list,
                density=cfg.density,
                k_mu=cfg.k_mu,
                k_lambda=cfg.k_lambda,
                k_damp=cfg.k_damp,
                add_surface_mesh_edges=True,
                particle_radius=cfg.particle_radius,
            )

            n_particles = len(nodes_local)
            per_env_ranges.append((particle_offset, particle_offset + n_particles))

            # Build-time world transform for this env (translation only — no rotation at spawn)
            T = torch.eye(4, dtype=torch.float32)
            T[0, 3] = float(world_pos[0])
            T[1, 3] = float(world_pos[1])
            T[2, 3] = float(world_pos[2])
            T_build_list.append(T)

            particle_offset += n_particles

        cls._per_env_particle_ranges = per_env_ranges

        T_stacked = torch.stack(T_build_list).to(device)
        cls._T_build_inv = torch.linalg.inv(T_stacked.double()).float()

        # VBD requires graph coloring before finalize
        logger.info("[Proxy VBD] Running graph coloring (required for VBD)...")
        cls._builder.color()

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
            cls._model.soft_contact_ke = cfg.soft_contact_ke
            cls._model.soft_contact_kd = cfg.soft_contact_kd
            cls._model.soft_contact_mu = cfg.soft_contact_mu
            logger.info(
                "[Proxy VBD] Contact params: ke=%.1f kd=%.1f mu=%.2f",
                cfg.soft_contact_ke, cfg.soft_contact_kd, cfg.soft_contact_mu,
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
            cls._vbd_solver = SolverVBD(
                cls._model,
                iterations=cfg.vbd_iterations,
                integrate_with_external_rigid_solver=True,
                particle_enable_self_contact=False,
            )
            logger.info("[Proxy VBD] SolverVBD initialized (iterations=%d).", cfg.vbd_iterations)

            # Create a dedicated CollisionPipeline + contacts buffer for particle-rigid
            # soft contacts, following the cloth_franka example pattern.
            #
            # Rationale:  model.collide() lazily creates a CollisionPipeline on first
            # call (doing d2h copies that are illegal during CUDA graph capture) AND
            # allocates a new Contacts buffer on every call.  By pre-allocating both
            # here — before super().initialize_solver() captures the CUDA graph — we
            # reuse a single buffer each substep, which is cleaner and avoids subtle
            # issues with repeated allocations inside the captured graph.
            #
            # soft_contact_margin should be slightly larger than particle_radius so
            # contacts are generated before deep penetration occurs.
            if cls._state_0 is not None:
                try:
                    import newton as _newton_pkg
                    soft_margin = cfg.particle_radius * 2.0
                    logger.info(
                        "[Proxy VBD] Creating soft-body CollisionPipeline "
                        "(soft_contact_margin=%.4f m)...", soft_margin
                    )
                    cls._soft_collision_pipeline = _newton_pkg.CollisionPipeline(
                        cls._model,
                        soft_contact_margin=soft_margin,
                    )
                    cls._soft_contacts = cls._soft_collision_pipeline.contacts()
                    # Warm up the pipeline outside CUDA graph capture (first call does
                    # d2h shape-type copies that are illegal inside graph capture).
                    cls._soft_collision_pipeline.collide(cls._state_0, cls._soft_contacts)
                    logger.info("[Proxy VBD] Soft-body collision pipeline ready.")
                except Exception as exc:
                    logger.warning("[Proxy VBD] CollisionPipeline setup failed: %s", exc)
                    cls._soft_collision_pipeline = None
                    cls._soft_contacts = None

        super().initialize_solver()  # creates rigid solver + captures CUDA graph

    @classmethod
    def _simulate(cls) -> None:
        if _vbd_enabled() and cls._vbd_solver is not None:
            cls._simulate_two_phase()
            return
        super()._simulate()

    @classmethod
    def _simulate_two_phase(cls) -> None:
        """Two-phase step: rigid MuJoCo (robot) then VBD (soft body).

        Follows the cloth_franka pattern:
          1. rigid.step(s0 → s1) — writes new rigid body positions into s1
          2. vbd.step(s0 → s1)   — writes new particle positions into s1;
             with integrate_with_external_rigid_solver=True, reads s1.body_q
             (the JUST-UPDATED rigid positions) for particle-rigid contact
          3. swap s0 ↔ s1         — s0 now has both updated rigid + particles
          4. clear forces
        """
        if cls._needs_collision_pipeline:
            cls._collision_pipeline.collide(cls._state_0, cls._contacts)
            contacts_rigid = cls._contacts
        else:
            contacts_rigid = None

        cfg = PhysicsManager._cfg
        need_copy = getattr(cfg, "use_cuda_graph", False) and cls._num_substeps % 2 == 1

        for i in range(cls._num_substeps):
            if cls._use_single_state:
                # Single-buffer mode: s0 == s1, both ops write in-place.
                cls._solver.step(cls._state_0, cls._state_0, cls._control, contacts_rigid, cls._solver_dt)
                cls._state_0.clear_forces()
                # Detect particle-rigid contacts using the pre-allocated pipeline/buffer
                # (cloth_franka pattern: reuse the same Contacts object every substep).
                if cls._soft_collision_pipeline is not None:
                    cls._soft_collision_pipeline.collide(cls._state_0, cls._soft_contacts)
                    contacts_soft = cls._soft_contacts
                else:
                    contacts_soft = cls._model.collide(cls._state_0)
                cls._vbd_solver.step(
                    cls._state_0, cls._state_0, cls._control, contacts_soft, cls._solver_dt
                )
            else:
                # Step 1: rigid solver s0 → s1 (new rigid body positions in s1).
                cls._solver.step(cls._state_0, cls._state_1, cls._control, contacts_rigid, cls._solver_dt)
                # Step 2: VBD solver s0 → s1 (new particle positions in s1;
                #   reads s1.body_q = newly updated rigid positions for contact).
                if cls._soft_collision_pipeline is not None:
                    cls._soft_collision_pipeline.collide(cls._state_0, cls._soft_contacts)
                    contacts_soft = cls._soft_contacts
                else:
                    contacts_soft = cls._model.collide(cls._state_0)
                cls._vbd_solver.step(
                    cls._state_0, cls._state_1, cls._control, contacts_soft, cls._solver_dt
                )
                # Step 3: advance buffers — s0 takes the fully updated state.
                if need_copy and i == cls._num_substeps - 1:
                    cls._state_0.assign(cls._state_1)
                else:
                    cls._state_0, cls._state_1 = cls._state_1, cls._state_0
                cls._state_0.clear_forces()

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
        if cls._usdrt_stage is None or cls._per_env_particle_ranges is None or cls._state_0 is None:
            return
        pq_wp = cls._state_0.particle_q
        if pq_wp is None or pq_wp.ptr is None:
            return
        try:
            import usdrt
            particle_t = wp.to_torch(pq_wp).float()
            num_envs = len(cls._per_env_particle_ranges)
            for env_idx in range(num_envs):
                start, end = cls._per_env_particle_ranges[env_idx]
                if end <= start:
                    continue
                com = particle_t[start:end].mean(dim=0).cpu().tolist()
                prim_path = f"/World/envs/env_{env_idx}/Object"
                prim = cls._usdrt_stage.GetPrimAtPath(prim_path)
                if prim and prim.IsValid():
                    xform_attr = prim.GetAttribute("xformOp:translate")
                    if xform_attr:
                        xform_attr.Set(usdrt.Gf.Vec3d(com[0], com[1], com[2]))
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
        if cls._per_env_particle_ranges is None or cls._state_0 is None:
            return torch.zeros_like(fingertip_pos_w)
        pq_wp = cls._state_0.particle_q
        if pq_wp is None or pq_wp.ptr is None:
            return torch.zeros_like(fingertip_pos_w)

        pq = wp.to_torch(pq_wp).float()
        num_envs = len(cls._per_env_particle_ranges)
        num_tips = fingertip_pos_w.shape[1]
        result = torch.zeros(
            num_envs, num_tips, 3, device=fingertip_pos_w.device, dtype=torch.float32
        )
        for e in range(num_envs):
            start, end = cls._per_env_particle_ranges[e]
            if end <= start:
                continue
            tips = fingertip_pos_w[e].to(pq.device)           # (num_tips, 3)
            particles = pq[start:end]                          # (N, 3)
            min_dists = torch.cdist(tips, particles).min(dim=1).values  # (num_tips,)
            in_contact = min_dists < contact_threshold
            result[e, in_contact, 2] = signal_magnitude
        return result

    @classmethod
    def get_object_pose(cls) -> tuple[wp.array, wp.array] | None:
        """Return (root_pos_w, root_quat_w) for the soft object per env.

        Position is the particle CoM. Orientation is estimated via Kabsch alignment
        of current vs rest-pose particles.

        Returns None if VBD is not enabled or particles are not initialised.
        """
        if cls._per_env_particle_ranges is None or cls._state_0 is None:
            return None
        pq_wp = cls._state_0.particle_q
        if pq_wp is None or pq_wp.ptr is None:
            return None
        rest = cls._particle_rest_q
        if rest is None:
            return None

        particle_t = wp.to_torch(pq_wp).float()
        if rest.device != particle_t.device:
            rest = rest.to(particle_t.device)

        pos_list, quat_list = [], []
        for start, end in cls._per_env_particle_ranges:
            n = end - start
            if n == 0:
                pos_list.append(torch.zeros(3, device=particle_t.device))
                quat_list.append(torch.tensor([0., 0., 0., 1.], device=particle_t.device))
                continue

            P_cur = particle_t[start:end]
            P_rest = rest[start:end]
            com = P_cur.mean(dim=0)
            pos_list.append(com)

            if n >= 3:
                P_cur_c = P_cur - com
                P_rest_c = P_rest - P_rest.mean(dim=0)
                H = P_rest_c.T @ P_cur_c
                U, _S, Vh = torch.linalg.svd(H)
                R = Vh.T @ U.T
                if torch.linalg.det(R) < 0:
                    Vh = Vh.clone()
                    Vh[-1, :] *= -1
                    R = Vh.T @ U.T
                quat_list.append(quat_from_matrix(R.unsqueeze(0)).squeeze(0))
            else:
                quat_list.append(torch.tensor([0., 0., 0., 1.], device=particle_t.device))

        pos_t = torch.stack(pos_list)
        quat_t = torch.stack(quat_list)
        return (
            wp.from_torch(pos_t.contiguous(), dtype=wp.vec3f),
            wp.from_torch(quat_t.contiguous(), dtype=wp.quatf),
        )

    @classmethod
    def get_object_velocity(cls) -> wp.array | None:
        """Return CoM linear velocity per env for the soft object."""
        if cls._per_env_particle_ranges is None or cls._state_0 is None:
            return None
        pqd_wp = getattr(cls._state_0, "particle_qd", None)
        if pqd_wp is None or pqd_wp.ptr is None:
            return None

        vel_t = wp.to_torch(pqd_wp).float()
        vel_list = []
        for start, end in cls._per_env_particle_ranges:
            n = end - start
            vel_list.append(vel_t[start:end].mean(dim=0) if n > 0 else torch.zeros(3, device=vel_t.device))
        lin_vel_t = torch.stack(vel_list)
        return wp.from_torch(lin_vel_t.contiguous(), dtype=wp.vec3f)

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
        if not _vbd_enabled() or cls._per_env_particle_ranges is None:
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

        T_inv = cls._T_build_inv.to(device)
        p_build = cls._particle_q_build.to(device)

        R = matrix_from_quat(root_pose[:, 3:7])
        t = root_pose[:, :3]
        T_reset = torch.eye(4, device=device).unsqueeze(0).expand(n, -1, -1).clone()
        T_reset[:, :3, :3] = R
        T_reset[:, :3, 3] = t

        pq = wp.to_torch(pq_dev).float().clone()
        for k in range(n):
            e = int(env_ids_t[k].item())
            start, end = cls._per_env_particle_ranges[e]
            if start >= end:
                continue
            delta = T_reset[k] @ T_inv[e]
            pq[start:end] = _transform_points(p_build[start:end], delta)

        new_pq = wp.from_torch(pq.contiguous(), dtype=wp.vec3f)
        wp.copy(pq_dev, new_pq)
        if cls._state_1 is not None and cls._state_1.particle_q is not None and cls._state_1.particle_q.ptr:
            wp.copy(cls._state_1.particle_q, new_pq)

        cls._particle_rest_q = pq.clone()

    @classmethod
    def reset_particle_velocities(cls, env_ids: torch.Tensor) -> None:
        """Zero particle velocities for the given envs after reset."""
        if not _vbd_enabled() or cls._per_env_particle_ranges is None or cls._state_0 is None:
            return
        pqd_dev = getattr(cls._state_0, "particle_qd", None)
        if pqd_dev is None or pqd_dev.ptr is None:
            return

        device = str(PhysicsManager._device)
        env_ids_t = torch.as_tensor(env_ids, device=device).long().reshape(-1)
        pqd = wp.to_torch(pqd_dev).float().clone()
        for k in range(len(env_ids_t)):
            e = int(env_ids_t[k].item())
            start, end = cls._per_env_particle_ranges[e]
            pqd[start:end] = 0.0
        new_pqd = wp.from_torch(pqd.contiguous(), dtype=wp.vec3f)
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
        cls._particle_q_build = None
        cls._T_build_inv = None
        cls._particle_rest_q = None
        super().clear()
