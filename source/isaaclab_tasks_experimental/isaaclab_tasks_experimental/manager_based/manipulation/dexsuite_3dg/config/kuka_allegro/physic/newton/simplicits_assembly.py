# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-env SimplicitsModelBuilder assembly (Step 3/4).

The builder and resulting model are owned by the extended Newton manager (Step 5).
Use :func:`build_multi_env_simplicits_model` with one env for single-env cases.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import warp as wp
from newton import Axis

from ..kaolin import (
    SimplicitsObjectCfg,
    compute_collision_particle_radius_from_mesh,
    create_rigid_simplicits_object_from_mesh,
)
from .dexsuite_3dg_builder_utils import build_rigid_proto_excluding_object

logger = logging.getLogger("dexsuite_3dg.simplicits.assembly")

try:
    from kaolin.experimental.newton.builder import SimplicitsModelBuilder
    from kaolin.experimental.newton.collisions import SimplicitsParticleNewtonShapeSoftContact
    from kaolin.experimental.newton.model import SimplicitsModel
except ImportError:
    SimplicitsModelBuilder = None  # type: ignore[assignment]
    SimplicitsParticleNewtonShapeSoftContact = None  # type: ignore[assignment]
    SimplicitsModel = None  # type: ignore[assignment]


def _resolve_collision_particle_radius(
    explicit: float | None,
    simplicits_cfg: Any,
    vertices: torch.Tensor,
    num_samples: int,
) -> float:
    """Resolve collision particle radius: explicit > config > computed from mesh."""
    if explicit is not None and explicit > 0.0:
        return float(explicit)
    cfg_radius = simplicits_cfg.collision_particle_radius
    if cfg_radius is not None and cfg_radius > 0.0:
        return float(cfg_radius)
    return compute_collision_particle_radius_from_mesh(vertices, num_samples)


def build_multi_env_simplicits_model(
    stage: Any,
    env_paths: list[str],
    object_relative_path: str,
    env_meshes: list[tuple[torch.Tensor, torch.Tensor]],
    simplicits_cfg: Any,
    env_transforms: list[torch.Tensor] | None = None,
    env_proto_xforms: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]] | None = None,
    device: str = "cuda",
    up_axis: str = "Z",
    solver_type: str | None = None,
    num_qp: int | None = None,
    collision_particle_radius: float | None = None,
    detection_ratio: float = 1.5,
    gravity: float = 9.81,
    verbose: bool = False,
) -> tuple[Any, list[tuple[int, int]]]:
    """Build a multi-env SimplicitsModel with one world per env; return model and particle ranges.

    For each env: begin_world() → add rigid proto with env xform → add one Simplicits object
    → end_world(). Then add_simplicits_collisions(), add_ground_plane(), finalize().
    Particles are registered per world so collision is isolated per env.
    Use the Newton visualizer's show_particles option to display Simplicits particles.

    Args:
        stage: USD stage.
        env_paths: Env root paths (e.g. /World/envs/env_0, ...).
        object_relative_path: Prim name to exclude from proto (e.g. Object).
        env_meshes: One (vertices, faces) per env for the Simplicits object.
        simplicits_cfg: SimplicitsObjectCfg for mesh→Simplicits factory.
        env_transforms: 4x4 initial transform per env for the Simplicits object; identity if None.
        env_proto_xforms: (pos_xyz, quat_xyzw) per env for rigid proto; None for identity.
        device: Target device for finalize.
        up_axis: Newton up axis (e.g. Z).
        solver_type: Solver type for rigid proto (e.g. mujoco_warp); None to skip solver-specific registration.
        num_qp: Quadrature points per object; defaults to simplicits_cfg.num_samples.
        collision_particle_radius: Radius [m]; if None, uses cfg or computed from first env mesh and num_samples.
        detection_ratio: Passed to add_simplicits_collisions.
        gravity: Gravity magnitude [m/s²] (positive).
        verbose: If True, log assembly steps.

    Returns:
        (model, per_env_particle_ranges). per_env_particle_ranges[i] = (start, end)
        into state.particle_q for env i.
    """
    if SimplicitsModelBuilder is None:
        raise ImportError(
            "Kaolin experimental Newton (SimplicitsModelBuilder) is required for assembly. "
            "Install kaolin with newton support."
        )

    if not isinstance(simplicits_cfg, SimplicitsObjectCfg):
        raise TypeError("simplicits_cfg must be SimplicitsObjectCfg")
    n_envs = len(env_paths)
    if n_envs != len(env_meshes):
        raise ValueError("env_paths and env_meshes must have the same length")
    if env_transforms is not None and len(env_transforms) != n_envs:
        raise ValueError("env_transforms length must match env_paths")
    if env_proto_xforms is not None and len(env_proto_xforms) != n_envs:
        raise ValueError("env_proto_xforms length must match env_paths")

    axis = Axis.from_string(up_axis) if isinstance(up_axis, str) else up_axis
    n_qp = num_qp if num_qp is not None else simplicits_cfg.num_samples
    vertices0 = env_meshes[0][0]
    collision_radius = _resolve_collision_particle_radius(collision_particle_radius, simplicits_cfg, vertices0, n_qp)

    # Build Simplicits scene (all objects) first so we have sim_pts; then build Newton with
    # one world per env: rigid proto + that env's particle slice, so particles get correct world ID.
    smb = _MultiWorldSimplicitsModelBuilder(up_axis=axis, gravity=-float(gravity))

    # Phase 1: add all Simplicits objects to the scene (no Newton worlds yet)
    for i, env_path in enumerate(env_paths):
        vertices, faces = env_meshes[i]
        sim_obj = create_rigid_simplicits_object_from_mesh(vertices, faces, simplicits_cfg, device=device)
        init_t = (
            env_transforms[i]
            if env_transforms is not None
            else torch.eye(4, device=vertices.device, dtype=vertices.dtype)
        )
        # Normalize 3x4 (rotation + translation) to 4x4 by appending homogeneous row [0, 0, 0, 1].
        if init_t.dim() == 2 and init_t.shape[0] == 3:
            row = torch.zeros(1, 4, device=init_t.device, dtype=init_t.dtype)
            row[0, 3] = 1.0
            init_t = torch.cat([init_t, row], dim=0)
        smb.add_simplicits_object_only(sim_obj, num_qp=n_qp, init_transform=init_t)
        if verbose:
            logger.debug("assembly: added simplicits object for env_path=%s", env_path)

    smb.add_simplicits_collisions(
        collision_particle_radius=collision_radius,
        detection_ratio=detection_ratio,
    )

    # Phase 2: build Newton worlds with rigid proto + particle slice per env (uses scene sim_pts)
    smb.build_worlds_with_particles(
        stage=stage,
        env_paths=env_paths,
        object_relative_path=object_relative_path,
        up_axis=up_axis,
        solver_type=solver_type,
        env_proto_xforms=env_proto_xforms,
        particle_radius=collision_radius,
        device=str(device) if device is not None else "cuda",
    )
    smb.add_ground_plane()

    model, per_env_particle_ranges = smb.finalize_multi_world(device=device, requires_grad=False, n_envs=n_envs)
    if verbose:
        logger.info(
            "assembly: multi-env finalize done; %s envs, ranges=%s",
            len(per_env_particle_ranges),
            per_env_particle_ranges,
        )
    return model, per_env_particle_ranges


class _MultiWorldSimplicitsModelBuilder:
    """Builds multi-env Simplicits model: scene first, then Newton worlds with rigid proto + particle slice per env."""

    def __init__(self, up_axis: Any, gravity: float):
        if SimplicitsModelBuilder is None:
            raise ImportError("Kaolin SimplicitsModelBuilder required")
        self._base = SimplicitsModelBuilder(up_axis=up_axis, gravity=gravity)

    def add_simplicits_object_only(
        self,
        sim_object: Any,
        num_qp: int = 1000,
        init_transform: torch.Tensor | None = None,
    ) -> int:
        """Add Simplicits object to the scene only (no Newton world)."""
        return self._base.add_simplicits_object(
            sim_object, num_qp=num_qp, init_transform=init_transform, is_kinematic=False
        )

    def build_worlds_with_particles(
        self,
        stage: Any,
        env_paths: list[str],
        object_relative_path: str,
        up_axis: str,
        solver_type: str | None,
        env_proto_xforms: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]] | None,
        particle_radius: float = 0.05,
        device: str = "cuda",
    ) -> None:
        """Setup scene to get sim_pts, then add one Newton world per env: rigid proto + that env's particle slice."""
        if SimplicitsModel is None:
            raise ImportError("Kaolin SimplicitsModel required for multi-env assembly")
        pending = getattr(self._base, "_pending_objects", [])
        if not pending:
            return
        # Kaolin SimplicitsModelBuilder defers add_simplicits_object until finalize(); there is no
        # .model on the builder beforehand. Mirror finalize()'s first phase to materialize sim_pts.
        self._base.model = SimplicitsModel(device)
        model = self._base.model
        for sim_object, num_qp, init_transform, is_kinematic in pending:
            model.simplicits_scene.add_object(sim_object, num_qp, init_transform, is_kinematic)
        scene = model.simplicits_scene
        if len(scene.sim_obj_dict) == 0:
            return
        # Scene setup so sim_pts is built
        acc_gravity = torch.zeros(3)
        acc_gravity[self._base.up_axis.value] = -self._base.gravity
        scene.set_scene_gravity(acc_gravity)
        for obj_idx, name, fcn, bdry_penalty, pinned_x in self._base._pending_boundary_conditions:
            scene.set_object_boundary_condition(obj_idx, name, fcn, bdry_penalty, pinned_x)
        if self._base._pending_collisions is not None:
            scene.enable_collisions(*self._base._pending_collisions)

        sim_pts = scene.sim_pts.numpy()
        sim_masses = scene.sim_M.values.numpy()[::3]
        n_pts = sim_pts.shape[0]
        assert sim_masses.shape[0] == n_pts
        obj_ids = sorted(scene.sim_obj_dict.keys())
        offsets = [0]
        for oid in obj_ids:
            offsets.append(offsets[-1] + scene.sim_obj_dict[oid].num_qp)
        assert offsets[-1] == n_pts

        n_envs = len(env_paths)
        self._simplicits_particle_start = len(self._base.particle_q)
        for i in range(n_envs):
            self._base.begin_world()
            proto = build_rigid_proto_excluding_object(
                stage,
                env_path=env_paths[i],
                object_relative_path=object_relative_path,
                up_axis=up_axis,
                solver_type=solver_type,
            )
            if env_proto_xforms is not None:
                pos, quat = env_proto_xforms[i]
                xform = wp.transform(pos, quat)
            else:
                xform = wp.transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
            self._base.add_builder(proto, xform=xform)
            start, end = offsets[i], offsets[i + 1]
            pos_list = []
            vel_list = []
            mass_list = []
            for j in range(start, end):
                p = sim_pts[j]
                pos_list.append((float(p[0]), float(p[1]), float(p[2])))
                vel_list.append((0.0, 0.0, 0.0))
                mass_list.append(float(sim_masses[j]))
            self._base.add_particles(
                pos=pos_list,
                vel=vel_list,
                mass=mass_list,
                radius=[particle_radius] * len(pos_list),
            )
            self._base.end_world()
        self._simplicits_particle_end = len(self._base.particle_q)

    def add_simplicits_collisions(self, **kwargs: Any) -> None:
        self._base.add_simplicits_collisions(**kwargs)

    def add_ground_plane(self) -> None:
        self._base.add_ground_plane()

    def finalize_multi_world(
        self, device: str = "cuda", requires_grad: bool = False, n_envs: int = 0
    ) -> tuple[Any, list[tuple[int, int]]]:
        """Finalize Newton model and return (model, per_env_particle_ranges)."""
        if SimplicitsParticleNewtonShapeSoftContact is None:
            raise ImportError("Kaolin SimplicitsParticleNewtonShapeSoftContact required")
        if not getattr(self._base, "model", None):
            model = self._base.finalize(device=device, requires_grad=requires_grad)
            return model, []
        scene = self._base.model.simplicits_scene
        has_simplicits = len(scene.sim_obj_dict) > 0
        if not has_simplicits:
            model = self._base.finalize(device=device, requires_grad=requires_grad)
            return model, []

        simplicits_particle_start = self._simplicits_particle_start
        simplicits_particle_end = self._simplicits_particle_end
        base_m = self._base.__class__.__mro__[1].finalize(self._base, device, requires_grad)
        self._base.model.__dict__.update(base_m.__dict__)
        self._base.model.simplicits_particle_start = simplicits_particle_start
        self._base.model.simplicits_particle_end = simplicits_particle_end
        if "newton_soft_collisions" not in scene.force_dict["pt_wise"]:
            scene.force_dict["pt_wise"]["newton_soft_collisions"] = {
                "object": SimplicitsParticleNewtonShapeSoftContact(
                    self._base.model,
                    wp.ones_like(scene.sim_vols),
                    dt=scene.timestep,
                    friction_use_lagged_body_contact_force_norm=False,
                ),
                "coeff": 0.001,
            }

        # Per-env ranges: we added one slice per env in build_worlds_with_particles in order
        obj_ids = sorted(scene.sim_obj_dict.keys())
        offsets = [0]
        for oid in obj_ids:
            offsets.append(offsets[-1] + scene.sim_obj_dict[oid].num_qp)
        running = simplicits_particle_start
        per_env_ranges = []
        for i in range(min(n_envs, len(obj_ids))):
            count = offsets[i + 1] - offsets[i]
            per_env_ranges.append((running, running + count))
            running += count
        return self._base.model, per_env_ranges
