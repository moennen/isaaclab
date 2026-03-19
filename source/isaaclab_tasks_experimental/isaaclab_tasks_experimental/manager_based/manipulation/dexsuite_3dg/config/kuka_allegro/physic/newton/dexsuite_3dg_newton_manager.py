# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extended NewtonManager for Dexsuite 3dg task (Newton mode only).

Use this module together with the env wrapper that patches ``isaaclab_newton.physics.NewtonManager``
before the simulation context is created. See docs/NEWTON_MANAGER_EXTENSION.md.

Step 5: When simplicits_enabled, replace builder with Simplicits model, add SimplicitsSolver,
and run two-phase step (rigid then Simplicits).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.physics import PhysicsManager
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix
from isaaclab.utils.timer import Timer

from ..kaolin import SimplicitsObjectCfg
from ..mesh_from_usd import (
    get_geom_world_transform_4x4,
    get_vertices_faces_from_prim_path,
    transform_points_mat4,
)
from .dexsuite_3dg_newton_cfg import Dexsuite3dgNewtonCfg
from .simplicits_assembly import build_multi_env_simplicits_model

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

logger = logging.getLogger("dexsuite_3dg.newton.manager")

try:
    from kaolin.experimental.newton.solver import SimplicitsSolver
except ImportError:
    SimplicitsSolver = None  # type: ignore[assignment]


# Match reference task Isaac-Dexsuite-Kuka-Allegro: env root and object name are
# hardcoded there too (e.g. prim_path="/World/envs/env_.*/table", "{ENV_REGEX_NS}/Object").
# NewtonManager.instantiate_builder_from_stage() has similar logic but scans /World (not
# /World/envs) and does not expose it as a callable; we cannot change code outside the
# task folder (SIMPLICITS_INTEGRATION_PLAN.md), so we implement discovery here.
_ENVS_ROOT = "/World/envs"
_OBJECT_RELATIVE_PATH = "Object"


def _discover_env_paths_and_xforms(
    stage: Any,
) -> tuple[
    list[str],
    list[tuple[tuple[float, float, float], tuple[float, float, float, float]]],
]:
    """Discover env paths and their world transforms from the stage.

    Returns:
        env_paths: e.g. ["/World/envs/env_0", ...].
        env_proto_xforms: (pos_xyz, quat_xyzw) per env for rigid proto placement.
    """
    root = stage.GetPrimAtPath(_ENVS_ROOT)
    if not root or not root.IsValid():
        return [], []
    children = root.GetChildren()
    indices = []
    for c in children:
        name = c.GetName()
        if name.startswith("env_"):
            try:
                indices.append((int(name.split("_")[1]), c.GetPath().pathString))
            except (IndexError, ValueError):
                continue
    indices.sort(key=lambda x: x[0])
    env_paths = [path for _, path in indices]
    xforms = []
    for _, path in indices:
        prim = stage.GetPrimAtPath(path)
        t_attr = prim.GetAttribute("xformOp:translate") if prim else None
        if t_attr:
            t = t_attr.Get()
            pos = (float(t[0]), float(t[1]), float(t[2]))
        else:
            pos = (0.0, 0.0, 0.0)
        quat = (0.0, 0.0, 0.0, 1.0)
        xforms.append((pos, quat))
    return env_paths, xforms


def _simplicits_enabled() -> bool:
    """True if config is Dexsuite3dgNewtonCfg with simplicits_enabled and simplicits_cfg set."""
    cfg = PhysicsManager._cfg
    if not isinstance(cfg, Dexsuite3dgNewtonCfg):
        return False
    return bool(cfg.simplicits_enabled and cfg.simplicits_cfg is not None)


class _SimplicitsModelBuilderWrapper:
    """Wrapper so super().start_simulation() can call finalize(device) and get our model."""

    def __init__(self, model: Any):
        self._model = model
        self.up_axis = None  # set by NewtonManager.start_simulation before finalize

    def finalize(self, device: str, **kwargs: Any) -> Any:
        return self._model


class Dexsuite3dgNewtonManager(NewtonManager):
    """Newton manager extended for Dexsuite 3dg (Simplicits spawn object, two-phase step).

    When :attr:`simplicits_enabled` is True, overrides :meth:`start_simulation`,
    :meth:`initialize_solver`, and :meth:`_simulate` to use a Simplicits model and
    two-phase stepping (rigid solver then SimplicitsSolver).
    """

    _simplicits_solver: Any = None
    _per_env_particle_ranges: list[tuple[int, int]] | None = None
    _particle_rest_q: torch.Tensor | None = None
    #: ``inv(T_build)`` per env [num_envs, 4, 4]; T_build = Object geom world matrix at model build.
    _simplicits_T_build_inv: torch.Tensor | None = None
    # World-space Simplicits particle positions immediately after build (same indexing as
    # ``state.particle_q``). Never updated: each reset rigid-teleports via
    # ``p_new = T_reset @ T_build_inv @ p_build`` using the per-env slice of this tensor as
    # ``p_build``.
    _simplicits_particle_q_build: torch.Tensor | None = None

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize the manager. No 3dg-specific setup required here."""
        super().initialize(sim_context)

    @classmethod
    def start_simulation(cls) -> None:
        """Start simulation. When simplicits enabled, build Simplicits model and set as builder."""
        if _simplicits_enabled():
            cls._start_simulation_simplicits()
            return
        super().start_simulation()

    @classmethod
    def _start_simulation_simplicits(cls) -> None:
        """Build Simplicits model from stage (env paths, object mesh per env), wrap, set builder."""
        if SimplicitsSolver is None:
            raise RuntimeError(
                "Simplicits is enabled but Kaolin SimplicitsSolver is not available. "
                "Install kaolin with Newton support or set simplicits_enabled=False."
            )
        cfg = PhysicsManager._cfg
        assert isinstance(cfg, Dexsuite3dgNewtonCfg) and cfg.simplicits_cfg is not None
        simplicits_cfg: SimplicitsObjectCfg = cfg.simplicits_cfg
        device = PhysicsManager._device
        stage = get_current_stage()
        env_paths, env_proto_xforms = _discover_env_paths_and_xforms(stage)
        if not env_paths:
            raise RuntimeError(
                "Simplicits enabled but no env prims found under /World/envs. "
                "Ensure the scene is built (e.g. replicate_physics=True) before reset."
            )
        num_envs = len(env_paths)
        env_meshes = []
        # Per-env geom world transform (column-vector: p_w = M[:3,:3] @ p_l + M[:3,3]).
        # Passed to Kaolin as init_transform and stored for rigid-teleport reset (T_build_inv).
        env_transforms = []
        for env_path in env_paths:
            obj_path = f"{env_path}/{_OBJECT_RELATIVE_PATH}"
            try:
                vertices_local, faces = get_vertices_faces_from_prim_path(
                    stage, obj_path, device=device, dtype=torch.float32
                )
                world_from_local = get_geom_world_transform_4x4(stage, obj_path, device=device, dtype=torch.float32)
            except KeyError as e:
                raise RuntimeError(
                    f"Simplicits enabled but could not get object mesh from {obj_path}. "
                    "Ensure the Object prim has Mesh or primitive geometry."
                ) from e
            env_meshes.append((vertices_local, faces))
            env_transforms.append(world_from_local.clone())
        T_stacked = torch.stack(env_transforms)
        cls._simplicits_T_build_inv = torch.linalg.inv(T_stacked.to(torch.float64)).to(
            device=device, dtype=torch.float32
        )
        with Timer(name="dexsuite_3dg_simplicits_build", msg="Simplicits model build took:"):
            model, per_env_ranges = build_multi_env_simplicits_model(
                stage=stage,
                env_paths=env_paths,
                object_relative_path=_OBJECT_RELATIVE_PATH,
                env_meshes=env_meshes,
                simplicits_cfg=simplicits_cfg,
                env_transforms=env_transforms,
                env_proto_xforms=env_proto_xforms,
                device=device,
                up_axis=cls._up_axis,
                solver_type="mujoco_warp",
                collision_particle_radius=simplicits_cfg.collision_particle_radius,
                gravity=abs(cls._gravity_vector[2]) if cls._gravity_vector[2] != 0 else 9.81,
            )
        cls._per_env_particle_ranges = per_env_ranges
        wrapper = _SimplicitsModelBuilderWrapper(model)
        cls._builder = wrapper
        cls._num_envs = num_envs
        super().start_simulation()
        # Build-time particle layout for rigid reset; rest pose for Kabsch (updated each reset).
        pq0 = wp.to_torch(cls._state_0.particle_q).clone().float()
        cls._simplicits_particle_q_build = pq0.clone()
        cls._particle_rest_q = pq0.clone()

    @classmethod
    def initialize_solver(cls) -> None:
        """Initialize solver. When simplicits enabled, add SimplicitsSolver after base solver."""
        super().initialize_solver()
        if _simplicits_enabled() and cls._model is not None and SimplicitsSolver is not None:
            cls._simplicits_solver = SimplicitsSolver(cls._model)

    @classmethod
    def _simulate(cls) -> None:
        """Run one simulation step. When simplicits enabled, two-phase: rigid then Simplicits."""
        if _simplicits_enabled() and cls._simplicits_solver is not None:
            cls._simulate_two_phase()
            return
        super()._simulate()

    @classmethod
    def _simulate_two_phase(cls) -> None:
        """Run one simulation step in two phases: rigid pipeline, then Simplicits pipeline, then state update.

        Pipeline overview:

        1. **Standard Newton (rigid/articulation)** — same as base :meth:`NewtonManager._simulate`:
           - **Collision:** Compute rigid–rigid (and rigid–articulation) contacts via
             :attr:`_collision_pipeline.collide` into :attr:`_contacts` (used only for the rigid step).
           - **Steps:** Run :attr:`_solver.step` for each substep (e.g. MuJoCo), advancing
             :attr:`_state_0` body and articulation DOFs only. Particles in :attr:`_state_0` are
             not modified by the rigid solver.

        2. **Simplicits pipeline:**
           - **Collision:** Call :attr:`_model.collide` on :attr:`_state_0` to compute contacts
             that include rigid–particle (soft–rigid) and particle–particle. Input state has rigid
             bodies already updated from phase 1 and particle positions from the previous step.
           - **Steps:** Run :meth:`SimplicitsSolver.step` to advance Simplicits particle state
             (e.g. from :attr:`_state_0` to :attr:`_state_1`).

        3. **State update:** Copy the result of the Simplicits step back into :attr:`_state_0`
           (:meth:`State.assign`) and clear forces so the next frame sees the combined rigid + particle state.

        After that, contact sensors are updated (from rigid contacts) and transforms are synced to USD
        when needed.
        """
        # Phase 1: rigid step (same as base)
        if cls._needs_collision_pipeline:
            cls._collision_pipeline.collide(cls._state_0, cls._contacts)
            contacts_rigid = cls._contacts
        else:
            contacts_rigid = None

        def step_fn(state_0, state_1):
            cls._solver.step(state_0, state_1, cls._control, contacts_rigid, cls._solver_dt)

        if cls._use_single_state:
            for _ in range(cls._num_substeps):
                step_fn(cls._state_0, cls._state_0)
                cls._state_0.clear_forces()
        else:
            cfg = PhysicsManager._cfg
            need_copy = cfg is not None and getattr(cfg, "use_cuda_graph", False) and cls._num_substeps % 2 == 1
            for i in range(cls._num_substeps):
                step_fn(cls._state_0, cls._state_1)
                if need_copy and i == cls._num_substeps - 1:
                    cls._state_0.assign(cls._state_1)
                else:
                    cls._state_0, cls._state_1 = cls._state_1, cls._state_0
                cls._state_0.clear_forces()

        # Phase 2: collide (rigid + Simplicits particles) then SimplicitsSolver
        #
        # State layout entering phase 2:
        #   state_0 — fully up-to-date: rigid fields (joint_q, body_q, joint_qd, …)
        #             advanced by phase 1; Simplicits fields (sim_z, particle_q, …)
        #             still hold values from the *previous* frame.
        #   state_1 — untouched since model.state() initialisation: rigid fields
        #             are at their initial values; Simplicits fields are stale.
        #
        # SimplicitsSolver.step(state_0 → state_1) reads sim_z / sim_z_dot from
        # state_0, runs one Simplicits Newton step, then writes the four updated
        # Simplicits fields into state_1:
        #   state_1.sim_z, state_1.sim_z_dot          (reduced DOFs + velocities)
        #   state_1.particle_q, state_1.particle_qd   (world-space positions / vel)
        # Rigid fields in state_1 are NOT written by the Simplicits solver and
        # therefore remain at their stale initial values.
        #
        # After the step we must NOT use state_0.assign(state_1) — that would copy
        # the stale rigid fields from state_1 back onto the correctly-advanced
        # state_0.  Instead we copy only the four Simplicits fields (see below).
        with wp.ScopedDevice(PhysicsManager._device):
            contacts = cls._model.collide(cls._state_0)
            cls._simplicits_solver.step(
                cls._state_0,
                cls._state_1,
                cls._control,
                contacts,
                cls._solver_dt,
            )
            # Copy only the Simplicits-updated fields from state_1 back to state_0.
            # Using state_0.assign(state_1) would overwrite rigid body fields (joint_q,
            # body_q, etc.) with stale initial values since state_1 is never updated by
            # the rigid solver step.
            wp.copy(cls._state_0.sim_z, cls._state_1.sim_z)
            wp.copy(cls._state_0.sim_z_dot, cls._state_1.sim_z_dot)
            _s_start = cls._model.simplicits_particle_start
            _s_end = cls._model.simplicits_particle_end
            wp.copy(
                dest=cls._state_0.particle_q,
                src=cls._state_1.particle_q,
                dest_offset=_s_start,
                src_offset=_s_start,
                count=_s_end - _s_start,
            )
            wp.copy(
                dest=cls._state_0.particle_qd,
                src=cls._state_1.particle_qd,
                dest_offset=_s_start,
                src_offset=_s_start,
                count=_s_end - _s_start,
            )
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
    def get_simplicits_object_pose(cls) -> tuple[wp.array, wp.array] | None:
        """Return (root_pos_w, root_quat_w) for the Simplicits object per env: one pose per object.

        Position is the center of mass of the particle cloud for that env. Orientation is the
        rotation from the canonical (rest) particle positions to the current positions, computed
        via Kabsch (best-fit rotation); if rest positions are missing or there are fewer than 3
        particles, orientation falls back to identity.

        Returns None if Simplicits is not enabled or particle ranges are not set.
        root_pos_w: (num_envs,) wp.vec3f; root_quat_w: (num_envs,) wp.quatf (x, y, z, w).
        """
        if cls._per_env_particle_ranges is None or cls._state_0 is None:
            return None
        particle_q = cls._state_0.particle_q
        if particle_q is None or particle_q.ptr is None:
            return None
        rest_t = cls._particle_rest_q
        if rest_t is None:
            return None
        particle_t = wp.to_torch(particle_q).float()
        if rest_t.device != particle_t.device:
            rest_t = rest_t.to(particle_t.device)
        pos_list = []
        quat_list = []
        for start, end in cls._per_env_particle_ranges:
            n = end - start
            if n > 0:
                P_cur = particle_t[start:end]
                P_rest = rest_t[start:end]
                com_cur = P_cur.mean(dim=0)
                pos_list.append(com_cur)
                if n >= 3:
                    com_rest = P_rest.mean(dim=0)
                    P_cur_c = P_cur - com_cur
                    P_rest_c = P_rest - com_rest
                    H = P_rest_c.T @ P_cur_c
                    U, _S, Vh = torch.linalg.svd(H)
                    R = Vh.T @ U.T
                    if torch.linalg.det(R) < 0:
                        Vh = Vh.clone()
                        Vh[-1, :] *= -1
                        R = Vh.T @ U.T
                    quat_list.append(quat_from_matrix(R.unsqueeze(0)).squeeze(0))
                else:
                    quat_list.append(
                        torch.tensor([0.0, 0.0, 0.0, 1.0], device=particle_t.device, dtype=particle_t.dtype)
                    )
            else:
                pos_list.append(torch.zeros(3, device=particle_t.device, dtype=particle_t.dtype))
                quat_list.append(torch.tensor([0.0, 0.0, 0.0, 1.0], device=particle_t.device, dtype=particle_t.dtype))
        pos_t = torch.stack(pos_list)
        quat_t = torch.stack(quat_list)
        pos_w = wp.from_torch(pos_t.contiguous(), dtype=wp.vec3f)
        quat_w = wp.from_torch(quat_t.contiguous(), dtype=wp.quatf)
        return (pos_w, quat_w)

    @classmethod
    def get_simplicits_object_velocity(cls) -> wp.array | None:
        """Return root linear velocity (CoM of particle velocities) per env for the Simplicits object.

        Uses state.particle_qd; angular velocity is not computed and callers should treat it as zero.
        Returns None if Simplicits is not enabled or particle_qd is not available.

        Returns:
            lin_vel_w: (num_envs,) wp.vec3f, or None.
        """
        if cls._per_env_particle_ranges is None or cls._state_0 is None:
            return None
        particle_qd = getattr(cls._state_0, "particle_qd", None)
        if particle_qd is None or particle_qd.ptr is None:
            return None
        vel_t = wp.to_torch(particle_qd).float()
        device = vel_t.device
        vel_list = []
        for start, end in cls._per_env_particle_ranges:
            n = end - start
            if n > 0:
                vel_list.append(vel_t[start:end].mean(dim=0))
            else:
                vel_list.append(torch.zeros(3, device=device, dtype=vel_t.dtype))
        lin_vel_t = torch.stack(vel_list)
        return wp.from_torch(lin_vel_t.contiguous(), dtype=wp.vec3f)

    @classmethod
    def apply_simplicits_particles_pose_reset(
        cls,
        env_ids: torch.Tensor | Any,
        root_pose: torch.Tensor,
    ) -> None:
        """Rigid-teleport Simplicits particles to match MDP object root pose after reset.

        For each env, ``p_new = T_reset @ T_build^{-1} @ p_build`` where ``p_build`` are
        build-time particle positions and ``T_reset`` is the 4x4 from ``root_pose``
        (pos + quat wxyz). Assumes Object geom world frame matches root link frame.

        Args:
            env_ids: Env indices (1D), same length as ``root_pose`` leading dim.
            root_pose: (N, 7) world pose: position [m] then quaternion **(x,y,z,w)** (same as
                :meth:`reset_root_state_uniform` / rigid ``write_root_pose_to_sim_index``).
        """
        if not _simplicits_enabled():
            return
        if cls._per_env_particle_ranges is None:
            return
        if cls._simplicits_particle_q_build is None:
            return
        if cls._simplicits_T_build_inv is None:
            return
        if cls._state_0 is None:
            return
        pq_dev = cls._state_0.particle_q
        if pq_dev is None or pq_dev.ptr is None:
            return
        env_ids_t = torch.as_tensor(env_ids, device=root_pose.device).long().reshape(-1)
        n = env_ids_t.shape[0]
        if root_pose.shape[0] != n:
            raise ValueError("root_pose batch dim must match len(env_ids)")
        device = str(PhysicsManager._device)
        T_inv = cls._simplicits_T_build_inv.to(device=device, dtype=torch.float32)
        p_build = cls._simplicits_particle_q_build.to(device=device, dtype=torch.float32)
        R = matrix_from_quat(root_pose[:, 3:7].to(device=device, dtype=torch.float32))
        t = root_pose[:, :3].to(torch.float32)
        T_reset = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1, -1).clone()
        T_reset[:, :3, :3] = R
        T_reset[:, :3, 3] = t
        pq = wp.to_torch(pq_dev).float().clone()
        for k in range(n):
            e = int(env_ids_t[k].item())
            start, end = cls._per_env_particle_ranges[e]
            if start >= end:
                continue
            delta = T_reset[k] @ T_inv[e]
            pts = p_build[start:end]
            pq[start:end] = transform_points_mat4(pts, delta)
        pq_w = wp.from_torch(pq.contiguous(), dtype=wp.vec3f)
        wp.copy(pq_dev, pq_w)
        if cls._state_1 is not None and cls._state_1.particle_q is not None and cls._state_1.particle_q.ptr:
            wp.copy(cls._state_1.particle_q, pq_w)
        cls._particle_rest_q = pq.clone()

        # Update sim_z and sim_z_dot so the Simplicits solver starts from the reset
        # transform, not from wherever sim_z was at the end of the previous episode.
        # Without this, the first post-reset step overwrites particle_q with positions
        # derived from the stale sim_z, undoing the teleport above.
        #
        # Encoding: sim_z[e*12:(e+1)*12] = [[R-I | t]] flattened (row-major, 3×4).
        # One rigid handle (12 DOFs) per env, objects added in env order 0,1,...
        if cls._state_0.sim_z is not None:
            I_pad = torch.zeros(3, 4, device=device, dtype=torch.float32)
            I_pad[:3, :3] = torch.eye(3, device=device, dtype=torch.float32)
            # z_new[k] = [[R_k - I | t_k]] flattened → shape (n, 12)
            z_new = (T_reset[:, :3, :] - I_pad).reshape(n, 12)
            sim_z_t = wp.to_torch(cls._state_0.sim_z).clone()
            for k in range(n):
                e = int(env_ids_t[k].item())
                sim_z_t[e * 12 : (e + 1) * 12] = z_new[k]
            new_sim_z = wp.from_torch(sim_z_t.contiguous())
            wp.copy(cls._state_0.sim_z, new_sim_z)
            if cls._state_1 is not None and cls._state_1.sim_z is not None:
                wp.copy(cls._state_1.sim_z, new_sim_z)

        if cls._state_0.sim_z_dot is not None:
            sim_z_dot_t = wp.to_torch(cls._state_0.sim_z_dot).clone()
            for k in range(n):
                e = int(env_ids_t[k].item())
                sim_z_dot_t[e * 12 : (e + 1) * 12] = 0.0
            new_sim_z_dot = wp.from_torch(sim_z_dot_t.contiguous())
            wp.copy(cls._state_0.sim_z_dot, new_sim_z_dot)
            if cls._state_1 is not None and cls._state_1.sim_z_dot is not None:
                wp.copy(cls._state_1.sim_z_dot, new_sim_z_dot)

    @classmethod
    def apply_simplicits_particles_velocity_reset(
        cls,
        env_ids: torch.Tensor | Any,
        root_velocity: torch.Tensor,
    ) -> None:
        """Set particle velocities from rigid root twist (lin + ang) about particle CoM.

        Args:
            env_ids: Env indices (1D), same length as ``root_velocity`` leading dim.
            root_velocity: (N, 6) [m/s, rad/s] linear then angular (world).
        """
        if not _simplicits_enabled() or cls._per_env_particle_ranges is None or cls._state_0 is None:
            return
        pqd_dev = getattr(cls._state_0, "particle_qd", None)
        if pqd_dev is None or pqd_dev.ptr is None:
            return
        env_ids_t = torch.as_tensor(env_ids, device=root_velocity.device).long().reshape(-1)
        n = env_ids_t.shape[0]
        if root_velocity.shape[0] != n or root_velocity.shape[-1] != 6:
            raise ValueError("root_velocity must be (N, 6)")
        device = str(PhysicsManager._device)
        pqd = wp.to_torch(pqd_dev).float().clone()
        pq = wp.to_torch(cls._state_0.particle_q).float()
        v_lin = root_velocity[:, :3].to(device=device, dtype=torch.float32)
        omega = root_velocity[:, 3:6].to(device=device, dtype=torch.float32)
        for k in range(n):
            e = int(env_ids_t[k].item())
            start, end = cls._per_env_particle_ranges[e]
            if start >= end:
                continue
            rel = pq[start:end] - pq[start:end].mean(dim=0, keepdim=True)
            w = omega[k].unsqueeze(0).expand_as(rel)
            pqd[start:end] = v_lin[k].unsqueeze(0) + torch.cross(w, rel, dim=-1)
        pqd_w = wp.from_torch(pqd.contiguous(), dtype=wp.vec3f)
        wp.copy(pqd_dev, pqd_w)
        if cls._state_1 is not None and getattr(cls._state_1, "particle_qd", None) is not None:
            pqd1 = cls._state_1.particle_qd
            if pqd1 is not None and pqd1.ptr:
                wp.copy(pqd1, pqd_w)

    @classmethod
    def clear(cls) -> None:
        """Clear manager state. Reset Simplicits-related attributes."""
        cls._simplicits_solver = None
        cls._per_env_particle_ranges = None
        cls._particle_rest_q = None
        cls._simplicits_T_build_inv = None
        cls._simplicits_particle_q_build = None
        super().clear()
