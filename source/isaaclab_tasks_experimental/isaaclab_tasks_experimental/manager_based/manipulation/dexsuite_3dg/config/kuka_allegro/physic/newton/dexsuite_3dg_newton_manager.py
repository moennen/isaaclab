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
from isaaclab.utils.math import quat_from_matrix
from isaaclab.utils.timer import Timer

from ..kaolin import SimplicitsObjectCfg
from ..mesh_from_usd import get_vertices_faces_from_prim_path
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
        env_transforms = []
        for _, env_path in enumerate(env_paths):
            obj_path = f"{env_path}/{_OBJECT_RELATIVE_PATH}"
            try:
                vertices, faces = get_vertices_faces_from_prim_path(stage, obj_path, device=device, dtype=torch.float32)
            except KeyError as e:
                raise RuntimeError(
                    f"Simplicits enabled but could not get object mesh from {obj_path}. "
                    "Ensure the Object prim has Mesh or primitive geometry."
                ) from e
            env_meshes.append((vertices, faces))
            env_transforms.append(torch.eye(4, device=vertices.device, dtype=torch.float32))
        logger.debug("simplicits mode enabled: building model for %s envs", num_envs)
        with Timer(name="dexsuite_3dg_simplicits_build", msg="Simplicits model build took:"):
            logger.debug(
                f"[DexSuite 3DG : Newton :] Starting Simplicits model build with {cls._up_axis} up axis and gravity {cls._gravity_vector}"
            )
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
                verbose=logger.isEnabledFor(logging.DEBUG),
            )
        cls._per_env_particle_ranges = per_env_ranges
        wrapper = _SimplicitsModelBuilderWrapper(model)
        cls._builder = wrapper
        cls._num_envs = num_envs
        super().start_simulation()
        # Cache canonical (rest) particle positions for rotation-from-rest in get_simplicits_object_pose.
        cls._particle_rest_q = wp.to_torch(cls._state_0.particle_q).clone().float()
        logger.info(
            "Simplicits model ready: %s envs, particle ranges %s",
            num_envs,
            per_env_ranges,
        )

    @classmethod
    def initialize_solver(cls) -> None:
        """Initialize solver. When simplicits enabled, add SimplicitsSolver after base solver."""
        super().initialize_solver()
        if _simplicits_enabled() and cls._model is not None and SimplicitsSolver is not None:
            cls._simplicits_solver = SimplicitsSolver(cls._model)
            logger.debug("SimplicitsSolver added (two-phase step)")

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
        with wp.ScopedDevice(PhysicsManager._device):
            contacts = cls._model.collide(cls._state_0)
            cls._simplicits_solver.step(
                cls._state_0,
                cls._state_1,
                cls._control,
                contacts,
                cls._solver_dt,
            )
            cls._state_0.assign(cls._state_1)
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
    def clear(cls) -> None:
        """Clear manager state. Reset Simplicits-related attributes."""
        cls._simplicits_solver = None
        cls._per_env_particle_ranges = None
        cls._particle_rest_q = None
        super().clear()
