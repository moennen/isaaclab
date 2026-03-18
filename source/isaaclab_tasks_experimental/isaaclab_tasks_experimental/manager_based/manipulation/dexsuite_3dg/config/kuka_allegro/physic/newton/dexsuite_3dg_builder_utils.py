# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a Newton rigid proto from USD for one env, excluding the Object prim (Step 2)."""

from __future__ import annotations

import logging
from typing import Any

from newton import ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
from newton.solvers import SolverMuJoCo

logger = logging.getLogger("dexsuite_3dg.newton.builder_utils")


def get_builder_body_articulation_labels(builder: ModelBuilder) -> tuple[list[Any], list[Any]]:
    """Return (body_labels, articulation_labels) from a Newton ModelBuilder.

    Newton may expose body_label or body_key (and similarly for articulations);
    this helper normalizes to lists for compatibility across versions.
    """
    body_labels = getattr(builder, "body_label", None) or getattr(builder, "body_key", [])
    art_labels = getattr(builder, "articulation_label", None) or getattr(builder, "articulation_key", [])
    body_list = list(body_labels) if body_labels is not None else []
    art_list = list(art_labels) if art_labels is not None else []
    return (body_list, art_list)


def _register_solver_attributes(builder: ModelBuilder, solver_type: str) -> None:
    """Register solver-specific attributes on the builder when required by the solver.

    In the Newton API, only SolverMuJoCo exposes ``register_custom_attributes(builder)``;
    SolverXPBD and SolverFeatherstone do not require (or provide) builder registration.
    We are solver-agnostic in the sense that we dispatch on ``solver_type`` (from the
    physics config); today only ``"mujoco_warp"`` has an implementation. Unknown or
    other solver types are no-op.
    """
    if solver_type == "mujoco_warp":
        SolverMuJoCo.register_custom_attributes(builder)
    # "xpbd", "featherstone", or any other solver_type: no builder registration in Newton API


def build_rigid_proto_excluding_object(
    stage: Any,
    env_path: str,
    object_relative_path: str,
    up_axis: str = "Z",
    load_visual_shapes: bool = True,
    skip_mesh_approximation: bool = True,
    solver_type: str | None = None,
    verbose: bool = False,
) -> ModelBuilder:
    """Build a Newton ModelBuilder (rigid only) for one env, per-env content only, excluding the Object prim.

    Adds only scene elements **under the env** (Robot, table, etc.); **does not** add global
    content (e.g. GroundPlane). This matches the existing Newton pattern: global content is
    shared and must be added once by the caller (Step 3/4 assembly); the proto is duplicated
    per world with an env-specific xform. The Object prim is never included (it will be
    added as a Simplicits object in Step 1).

    The solver used at runtime is chosen by NewtonManager from the physics config
    (``cfg.solver_cfg``); the config's ``solver_type`` (e.g. ``"mujoco_warp"``,
    ``"xpbd"``, ``"featherstone"``) is read in :meth:`NewtonManager.initialize_solver`.
    Pass the same ``solver_type`` here so that any solver-specific builder registration
    (e.g. MuJoCo custom attributes) is applied to the proto. If ``solver_type`` is
    None, no solver-specific registration is performed (caller must ensure the final
    model is built with the correct solver attributes before finalize).

    Args:
        stage: USD stage (e.g. from get_current_stage()).
        env_path: Full path to the env root (e.g. ``/World/envs/env_0``).
        object_relative_path: Prim name under env_path to exclude (e.g. ``Object``).
        up_axis: Newton model up axis. Defaults to ``"Z"``.
        load_visual_shapes: Whether to load visual shapes. Defaults to True.
        skip_mesh_approximation: Whether to skip mesh approximation. Defaults to True.
        solver_type: Solver type from the physics config (e.g. ``"mujoco_warp"``,
            ``"xpbd"``, ``"featherstone"``). Only ``"mujoco_warp"`` triggers builder
            registration (MuJoCo custom attributes); other types are no-op. Pass None
            to skip any solver-specific registration.
        verbose: If True, log added root_paths and body/articulation counts.

    Returns:
        A Newton ModelBuilder containing rigid content for this env only (no global).
        The caller must add global content (e.g. ground plane) once when building
        the full model (Step 3 single-env or Step 4 multi-env).
    """
    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    builder = ModelBuilder(up_axis=up_axis)

    if solver_type is not None:
        _register_solver_attributes(builder, solver_type)

    # Per-env content only: add each direct child of env_path that is not the Object
    env_prim = stage.GetPrimAtPath(env_path)
    if not env_prim.IsValid():
        logger.warning("[DexSuite 3DG : Newton :] builder_utils: env_path %s is not valid on stage", env_path)
        return builder

    object_path = f"{env_path}/{object_relative_path}"
    for child in env_prim.GetChildren():
        child_name = child.GetName()
        if child_name == object_relative_path:
            logger.debug("[DexSuite 3DG : Newton :] builder_utils: skipping Object prim %s", object_path)
            continue
        child_path = child.GetPath().pathString
        builder.add_usd(
            stage,
            root_path=child_path,
            load_visual_shapes=load_visual_shapes,
            skip_mesh_approximation=skip_mesh_approximation,
            schema_resolvers=schema_resolvers,
        )
        if verbose:
            body_list, art_list = get_builder_body_articulation_labels(builder)
            logger.debug(
                "[DexSuite 3DG : Newton :] builder_utils: after add_usd root_path=%s body_count=%s articulation_count=%s",
                child_path,
                len(body_list),
                len(art_list),
            )

    return builder
