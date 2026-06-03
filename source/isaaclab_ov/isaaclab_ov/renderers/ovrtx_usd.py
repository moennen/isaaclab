# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD manipulation for OVRTX: RenderProduct authoring, camera injection, and stage prim activation."""

import logging
import math

from pxr import Gf, Sdf, Usd, UsdGeom

logger = logging.getLogger(__name__)


def get_render_var_config(data_types: list[str]) -> tuple[str, str, str]:
    """Return (render_var_path, render_var_name, source_name) from data_types."""
    use_depth = any(dt in ["depth", "distance_to_image_plane", "distance_to_camera"] for dt in data_types)
    use_albedo = "albedo" in data_types
    use_semantic = "semantic_segmentation" in data_types
    use_rgb = any(dt in ["rgb", "rgba"] for dt in data_types)
    use_hdr = "rgb_hdr" in data_types

    if use_depth and not (use_rgb or use_albedo or use_semantic):
        return "/Render/Vars/depth", "depth", "DistanceToImagePlaneSD"
    if use_albedo and not (use_rgb or use_semantic):
        return "/Render/Vars/albedo", "albedo", "DiffuseAlbedoSD"
    if use_semantic and not (use_rgb or use_albedo):
        return "/Render/Vars/semantic", "semantic", "SemanticSegmentation"
    if use_hdr and not use_rgb:
        return "/Render/Vars/HdrColor", "HdrColor", "HdrColor"
    return "/Render/Vars/LdrColor", "LdrColor", "LdrColor"


def get_render_var_configs(data_types: list[str]) -> list[tuple[str, str, str]]:
    """Return render var configs needed for the requested data types.

    Returns the single render var resolved by :func:`get_render_var_config`,
    plus ``HdrColor`` when both ``"rgb"`` (or ``"rgba"``) and ``"rgb_hdr"`` are
    in ``data_types`` so PPISP can consume the HDR AOV alongside the LDR
    destination on the same render product. Other multi-AOV combinations are
    not supported.
    """
    data_types = data_types if data_types else ["rgb"]
    render_vars: list[tuple[str, str, str]] = [get_render_var_config(data_types)]
    use_rgb = any(dt in ["rgb", "rgba"] for dt in data_types)
    if use_rgb and "rgb_hdr" in data_types:
        render_vars.append(("/Render/Vars/HdrColor", "HdrColor", "HdrColor"))
    return render_vars


def _tiled_resolution(num_envs: int, width: int, height: int) -> tuple[int, int]:
    """Compute tiled width and height from env count and per-env resolution (same as Camera)."""
    num_cols = math.ceil(math.sqrt(num_envs))
    num_rows = math.ceil(num_envs / num_cols)
    return num_cols * width, num_rows * height


def build_render_product_on_stage(
    stage: Usd.Stage,
    width: int,
    height: int,
    num_envs: int,
    data_types: list[str],
    minimal_mode: int | None = None,
    camera_rel_path: str = "Camera",
    render_product_name: str = "RenderProduct",
) -> str:
    """Author the OVRTX render product directly on ``stage``.

    Callers author this on an anonymous copy of the exported simulation stage
    before exporting the complete stage string consumed by OVRTX.
    """
    data_types = data_types if data_types else ["rgb"]
    tiled_width, tiled_height = _tiled_resolution(num_envs, width, height)

    camera_paths = [f"/World/envs/env_{i}/{camera_rel_path}" for i in range(num_envs)]
    render_product_path = f"/Render/{render_product_name}"
    render_var_configs = get_render_var_configs(data_types)

    stage.DefinePrim("/Render", "Scope")
    render_product = stage.DefinePrim(render_product_path, "RenderProduct")
    _prepend_api_schema(render_product, "OmniRtxSettingsCommonAdvancedAPI_1")
    render_product.CreateRelationship("camera").SetTargets([Sdf.Path(path) for path in camera_paths])
    render_product.CreateAttribute("omni:rtx:background:source:type", Sdf.ValueTypeNames.Token).Set("domeLight")
    render_product.CreateAttribute("omni:rtx:rt:ambientLight:intensity", Sdf.ValueTypeNames.Float).Set(1.0)
    render_product.CreateAttribute("omni:rtx:rendermode", Sdf.ValueTypeNames.Token).Set(
        "RealTimePathTracing" if minimal_mode is None else "Minimal"
    )
    if minimal_mode is not None:
        render_product.CreateAttribute("omni:rtx:minimal:mode", Sdf.ValueTypeNames.Int).Set(minimal_mode)
    render_product.CreateAttribute("omni:rtx:waitForEvents", Sdf.ValueTypeNames.TokenArray).Set(
        ["AllLoadingFinished", "OnlyOnFirstRequest"]
    )
    render_product.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(tiled_width, tiled_height))

    stage.DefinePrim("/Render/Vars")
    render_var_targets = []
    for render_var_path, _, source_name in render_var_configs:
        render_var = stage.DefinePrim(render_var_path, "RenderVar")
        render_var.CreateAttribute(
            "sourceName",
            Sdf.ValueTypeNames.String,
            custom=False,
            variability=Sdf.VariabilityUniform,
        ).Set(source_name)
        render_var_targets.append(Sdf.Path(render_var_path))
    render_product.CreateRelationship("orderedVars").SetTargets(render_var_targets)
    return render_product_path


def stage_from_string(root_layer_content: str) -> Usd.Stage:
    """Create an anonymous USD stage from exported root-layer content."""
    root_layer = Sdf.Layer.CreateAnonymous("ovrtx_renderer_stage.usda")
    if not root_layer.ImportFromString(root_layer_content):
        raise RuntimeError("Failed to import exported OVRTX USD string into a temporary USD stage.")
    stage = Usd.Stage.Open(root_layer)
    if stage is None:
        raise RuntimeError("Failed to open temporary OVRTX USD stage from imported string.")
    return stage


def _prepend_api_schema(prim: Usd.Prim, schema_name: str) -> None:
    schemas = Sdf.TokenListOp()
    current = prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
    items = list(current.prependedItems) if current.prependedItems else []
    if schema_name not in items:
        items.append(schema_name)
    schemas.prependedItems = items
    prim.SetMetadata("apiSchemas", schemas)


def create_scene_partition_attributes(
    stage,
    num_envs: int = 1,
    use_ovrtx_cloning: bool = True,
) -> None:
    """Create scene partition attributes for env roots and cameras.

    If use_ovrtx_cloning is True, only env_0 is exported for OVRTX; env_1..env_{n-1} are deactivated before export.
    OVRTX clones env_0 internally and _update_scene_partitions_after_clone sets partition attributes on the clones.
    So we only need to set attributes on env_0 here.

    Camera prims are discovered by USD type (``UsdGeom.Camera``) rather than by name, so this works regardless of
    where the camera is placed in the hierarchy.

    Args:
        stage: USD stage to modify.
        num_envs: Number of environments.
        use_ovrtx_cloning: Whether OVRTX cloning is enabled.
    """
    env_indices = [0] if use_ovrtx_cloning else range(num_envs)
    for env_idx in env_indices:
        env_path = f"/World/envs/env_{env_idx}"
        env_prim = stage.GetPrimAtPath(env_path)
        if not env_prim.IsValid():
            logger.warning("Failed to get env root prim at '%s'", env_path)
            continue

        scene_partition = f"env_{env_idx}"
        env_prim.CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token).Set(scene_partition)
        logger.debug("Set scene partition '%s' on env root '%s'", scene_partition, env_prim.GetPath())

        for prim in Usd.PrimRange(env_prim):
            if prim.GetPath() == env_prim.GetPath():
                continue

            if not prim.IsA(UsdGeom.Camera):
                continue

            prim.CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set(scene_partition)
            logger.debug("Set scene partition '%s' on camera '%s'", scene_partition, prim.GetPath())


def export_stage_to_string(stage, num_envs: int, use_ovrtx_cloning: bool = True) -> str:
    """Export the stage to a string; when num_envs > 1, only env_0 is exported for OVRTX cloning.

    When num_envs > 1, deactivates env_1..env_{num_envs-1} before export and reactivates
    them after, so the exported content contains only env_0. The stage is modified in place.

    Args:
        stage: USD stage to export.
        num_envs: Number of environments.
        use_ovrtx_cloning: Whether OVRTX cloning is enabled.

    Returns:
        The exported stage as a string.
    """
    deactivated_prims = []
    if use_ovrtx_cloning and num_envs > 1:
        logger.info("Deactivating %d environment roots...", num_envs - 1)
        for env_idx in range(1, num_envs):
            env_path = f"/World/envs/env_{env_idx}"
            prim = stage.GetPrimAtPath(env_path)
            if prim.IsValid() and prim.IsActive():
                prim.SetActive(False)
                deactivated_prims.append(prim)
                logger.debug("Deactivated environment root: %s", env_path)

        logger.info("Deactivated %d environment roots in total", len(deactivated_prims))

    try:
        return stage.ExportToString()
    finally:
        if deactivated_prims:
            logger.info("Reactivating %d environment roots...", len(deactivated_prims))
            for prim in deactivated_prims:
                if prim.IsValid():
                    prim.SetActive(True)
