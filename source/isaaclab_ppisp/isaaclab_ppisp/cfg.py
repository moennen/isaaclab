# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPISP configuration and USD/shader parsing helpers.

The implementation follows the physically plausible ISP model described in
https://arxiv.org/abs/2601.18336.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any

from isaaclab.utils.configclass import configclass

PPISP_SHADER_NAME = "PPISP"
"""Conventional prim name for a PPISP shader child under a ``RenderProduct``."""

PPISP_AUTO_SHADER_NAME = "PPISPAuto"
"""Conventional prim name for a controller-driven PPISP shader child under a ``RenderProduct``."""

PPISP_SHADER_NAMES = (PPISP_SHADER_NAME, PPISP_AUTO_SHADER_NAME)
"""PPISP shader prim names understood by the USD discovery helpers, in preference order."""

PPISP_CONTROLLER_PARAMS_INPUT = "ControllerParams"
"""Input name on ``PPISPAuto`` that carries controller-predicted exposure/color parameters."""

PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN = 241_961

PPISP_FLOAT2_INPUTS = {
    "vignettingCenterR",
    "vignettingCenterG",
    "vignettingCenterB",
    "colorLatentBlue",
    "colorLatentRed",
    "colorLatentGreen",
    "colorLatentNeutral",
}

PPISP_DEFAULT_INPUTS: dict[str, float | tuple[float, float]] = {
    "responsivity": 1.0,
    "exposureOffset": 0.0,
    "vignettingCenterR": (0.0, 0.0),
    "vignettingAlpha1R": 0.0,
    "vignettingAlpha2R": 0.0,
    "vignettingAlpha3R": 0.0,
    "vignettingCenterG": (0.0, 0.0),
    "vignettingAlpha1G": 0.0,
    "vignettingAlpha2G": 0.0,
    "vignettingAlpha3G": 0.0,
    "vignettingCenterB": (0.0, 0.0),
    "vignettingAlpha1B": 0.0,
    "vignettingAlpha2B": 0.0,
    "vignettingAlpha3B": 0.0,
    "colorLatentBlue": (0.0, 0.0),
    "colorLatentRed": (0.0, 0.0),
    "colorLatentGreen": (0.0, 0.0),
    "colorLatentNeutral": (0.0, 0.0),
    "crfToeR": 0.013659,
    "crfShoulderR": 0.013659,
    "crfGammaR": 0.378165,
    "crfCenterR": 0.0,
    "crfToeG": 0.013659,
    "crfShoulderG": 0.013659,
    "crfGammaG": 0.378165,
    "crfCenterG": 0.0,
    "crfToeB": 0.013659,
    "crfShoulderB": 0.013659,
    "crfGammaB": 0.378165,
    "crfCenterB": 0.0,
}


def default_ppisp_inputs() -> dict[str, float | tuple[float, float]]:
    """Return a copy of the PPISP identity/default input dictionary."""
    return dict(PPISP_DEFAULT_INPUTS)


@configclass
class PpispControllerCfg:
    """Configuration for an exported embedded-weight PPISP controller."""

    prior_exposure: float = 0.0
    """Controller prior exposure scalar authored as ``inputs:priorExposure``."""

    weights: tuple[float, ...] | None = None
    """Flattened embedded controller weights parsed from ``kControllerWeights``."""


@configclass
class PpispCfg:
    """Configuration for PPISP post-processing.

    PPISP inputs are static in IsaacLab. If imported from animated USD shader inputs,
    the first authored time sample is used and later samples are ignored.
    """

    spg_render_product_prim_path: str | None = None
    """Optional source RenderProduct prim path containing an authored PPISP SPG graph.

    Only the composed ``UsdShade.Shader`` and intermediate ``RenderVar`` graph
    children are consumed from this container. Camera, resolution, and render
    settings from the source RenderProduct are intentionally ignored.
    """

    inputs: dict[str, float | tuple[float, float]] = field(default_factory=default_ppisp_inputs)
    """Flat PPISP shader input values keyed by USD input name.

    Coordinate conventions for spatial inputs:

    * ``vignettingCenter{R,G,B}`` is a 2D offset in UV space normalised by
      ``max(width, height)`` with the image center at ``(0.0, 0.0)``. The
      :data:`PPISP_DEFAULT_INPUTS` defaults place every channel's
      optical center at the image center.
    * Radial vignetting coefficients ``vignettingAlpha{1,2,3}{R,G,B}`` are
      polynomial coefficients in the same normalised radius, applied as
      ``factor = clamp(1 + a1*r^2 + a2*r^4 + a3*r^6, 0, 1)``; with a square
      frame the image corners sit at ``r^2 = 0.5``.
    """

    controller: PpispControllerCfg | None = None
    """Optional controller graph parsed from a ``PPISPAuto`` SPG shader.

    When present, the Newton/Warp fallback predicts ``exposureOffset`` and the
    four color latents from the HDR image each frame. Static PPISP inputs still
    provide responsivity, vignetting, and CRF.
    """


def normalize_ppisp_cfg(
    ppisp_cfg: PpispCfg | None,
    stage: Any | None = None,
    *,
    load_controller_weights: bool = True,
) -> PpispCfg | None:
    """Normalise a :class:`PpispCfg` for downstream consumption.

    * If ``ppisp_cfg`` is ``None``, returns ``None``.
    * If ``ppisp_cfg.spg_render_product_prim_path`` is set and ``stage`` is
      supplied, merges the shader-authored values with the cfg's explicit
      overrides (see :func:`_merge_shader_inputs_with_cfg`).
    * Otherwise validates ``ppisp_cfg.inputs`` and fills in defaults.
    """
    if ppisp_cfg is None:
        return None
    if not isinstance(ppisp_cfg, PpispCfg):
        raise TypeError(f"Unsupported PPISP configuration type: {type(ppisp_cfg)!r}")
    input_overrides = dict(ppisp_cfg.inputs)
    if ppisp_cfg.spg_render_product_prim_path and stage is not None:
        return _merge_shader_inputs_with_cfg(
            ppisp_cfg, stage, input_overrides, load_controller_weights=load_controller_weights
        )
    ppisp_cfg.inputs = _normalized_inputs(input_overrides)
    return ppisp_cfg


def ppisp_cfg_from_usd_shader(shader: Any, *, load_controller_weights: bool = True) -> PpispCfg:
    """Create :class:`PpispCfg` from a ``UsdShade.Shader`` prim.

    Animated inputs are collapsed to their first authored time sample.
    """
    from .controller_source import ppisp_controller_cfg_from_auto_shader
    from .spg import ppisp_spg_prim_paths

    prim = shader.GetPrim()
    parent = prim.GetParent()
    has_native_spg = bool(ppisp_spg_prim_paths(parent, prim)) if parent else False
    cfg = PpispCfg(
        spg_render_product_prim_path=str(parent.GetPath()) if has_native_spg else None,
    )
    values = default_ppisp_inputs()
    for input_name in values:
        shader_input = shader.GetInput(input_name)
        if not shader_input:
            continue
        attr = shader_input.GetAttr()
        value = _read_first_authored_value(attr)
        if value is not None:
            values[input_name] = _normalize_input_value(input_name, value)
    cfg.inputs = values
    cfg.controller = ppisp_controller_cfg_from_auto_shader(shader, load_controller_weights=load_controller_weights)
    return cfg


def ppisp_cfg_from_usd_stage(stage: Any, shader_prim_path: str, *, load_controller_weights: bool = True) -> PpispCfg:
    """Create :class:`PpispCfg` from a shader prim path in a USD stage."""
    from pxr import UsdShade

    shader = UsdShade.Shader(stage.GetPrimAtPath(shader_prim_path))
    if not shader:
        raise ValueError(f"PPISP shader prim not found at path: {shader_prim_path}")
    return ppisp_cfg_from_usd_shader(shader, load_controller_weights=load_controller_weights)


def ppisp_cfg_from_usd_render_product(
    stage: Any, render_product_prim_path: str, *, load_controller_weights: bool = True
) -> PpispCfg:
    """Create :class:`PpispCfg` from a RenderProduct prim containing a PPISP SPG graph."""
    from pxr import UsdShade

    from .spg import find_ppisp_shader_prim

    render_product_prim = stage.GetPrimAtPath(render_product_prim_path)
    if not render_product_prim or not render_product_prim.IsValid():
        raise ValueError(f"PPISP RenderProduct prim not found at path: {render_product_prim_path}")
    shader_prim = find_ppisp_shader_prim(stage, render_product_prim)
    if shader_prim is None:
        raise ValueError(f"PPISP shader prim not found under RenderProduct: {render_product_prim_path}")
    return ppisp_cfg_from_usd_shader(UsdShade.Shader(shader_prim), load_controller_weights=load_controller_weights)


def _normalized_inputs(inputs: dict[str, Any]) -> dict[str, float | tuple[float, float]]:
    values = default_ppisp_inputs()
    for input_name, value in inputs.items():
        if input_name not in values:
            raise ValueError(f"Unknown PPISP input: {input_name}")
        values[input_name] = _normalize_input_value(input_name, value)
    return values


def _merge_shader_inputs_with_cfg(
    ppisp_cfg: PpispCfg,
    stage: Any,
    input_overrides: dict[str, Any],
    *,
    load_controller_weights: bool,
) -> PpispCfg:
    parsed_cfg = ppisp_cfg_from_usd_render_product(
        stage, ppisp_cfg.spg_render_product_prim_path, load_controller_weights=load_controller_weights
    )
    if input_overrides != PPISP_DEFAULT_INPUTS:
        parsed_cfg.inputs.update(_normalized_input_overrides(input_overrides))
    return parsed_cfg


def _normalized_input_overrides(inputs: dict[str, Any]) -> dict[str, float | tuple[float, float]]:
    values = {}
    for input_name, value in inputs.items():
        if input_name not in PPISP_DEFAULT_INPUTS:
            raise ValueError(f"Unknown PPISP input: {input_name}")
        values[input_name] = _normalize_input_value(input_name, value)
    return values


def _normalize_input_value(input_name: str, value: Any) -> float | tuple[float, float]:
    if input_name in PPISP_FLOAT2_INPUTS:
        if len(value) != 2:
            raise ValueError(f"PPISP input '{input_name}' expects two values.")
        return (float(value[0]), float(value[1]))
    return float(value)


def _read_first_authored_value(attr: Any) -> Any:
    time_samples = attr.GetTimeSamples()
    if time_samples:
        return attr.Get(time_samples[0])
    return attr.Get()


def resolve_and_normalize(isp_cfg: Any, stage: Any, camera_prim_path: str) -> PpispCfg | None:
    """Resolve a Camera sensor batch's ``isp_cfg`` to a normalised cfg or ``None``.

    Handles all three legal forms of :attr:`~isaaclab.sensors.camera.CameraCfg.isp_cfg`:

    * ``None`` → returns ``None``.
    * :class:`~isaaclab.sensors.camera.CameraISPMode` sentinel — walks the stage
      via :func:`auto_camera_ppisp_cfg` (and :func:`auto_any_ppisp_cfg` for
      ``AUTO_ANY``) to discover a PPISP shader. Returns the parsed +
      normalised :class:`PpispCfg`, or ``None`` if no shader matched.
    * Concrete :class:`PpispCfg` — normalises in place (validates input keys,
      fills defaults, and merges shader-authored values when
      ``spg_render_product_prim_path`` is set).

    This is the single entry point renderer backends call inside their
    ``prepare_cameras`` hook so :mod:`isaaclab.sensors.camera` does not need
    to know about PPISP types at all. The returned cfg applies to the whole
    Camera sensor batch; callers pass the first matched camera prim path for
    the camera-bound discovery phase.

    Args:
        isp_cfg: The Camera sensor's :attr:`isp_cfg` value (``None``, ``CameraISPMode``, or :class:`PpispCfg`).
        stage: USD stage used for sentinel discovery and shader-path resolution.
        camera_prim_path: Absolute path of the first matched camera prim in the
            Camera sensor batch (target of the ``camera`` relationship for the
            camera-bound discovery phase).

    Returns:
        A fully-normalised :class:`PpispCfg`, or ``None`` if the batch has no ISP.
    """
    return _resolve_and_normalize_impl(isp_cfg, stage, camera_prim_path, load_controller_weights=True)


def resolve_and_normalize_for_native_spg(isp_cfg: Any, stage: Any, camera_prim_path: str) -> PpispCfg | None:
    """Resolve ``isp_cfg`` for an RTX/OVRTX backend that can run authored SPG directly.

    Controller CUDA weights are not parsed on this path because the renderer's
    SPG runtime consumes the authored source assets itself. Newton still calls
    :func:`resolve_and_normalize`, which loads embedded weights for the Warp
    fallback.
    """
    return _resolve_and_normalize_impl(isp_cfg, stage, camera_prim_path, load_controller_weights=False)


def _resolve_and_normalize_impl(
    isp_cfg: Any,
    stage: Any,
    camera_prim_path: str,
    *,
    load_controller_weights: bool,
) -> PpispCfg | None:
    """Implementation for PPISP cfg resolution."""
    # Local import avoids a top-of-module dep on isaaclab.sensors.
    from isaaclab.sensors.camera.camera_isp import CameraISPMode

    if isp_cfg is None:
        return None
    if isinstance(isp_cfg, CameraISPMode):
        resolved = auto_camera_ppisp_cfg(stage, camera_prim_path, load_controller_weights=load_controller_weights)
        if resolved is None and isp_cfg == CameraISPMode.AUTO_ANY:
            resolved = auto_any_ppisp_cfg(stage, load_controller_weights=load_controller_weights)
        if resolved is None:
            return None
        return normalize_ppisp_cfg(resolved, stage=stage, load_controller_weights=load_controller_weights)
    return normalize_ppisp_cfg(isp_cfg, stage=stage, load_controller_weights=load_controller_weights)


def auto_camera_ppisp_cfg(
    stage: Any, camera_prim_path: str, *, load_controller_weights: bool = True
) -> PpispCfg | None:
    """Find the first PPISP shader on ``stage`` bound to ``camera_prim_path``.

    Walks ``stage`` looking for the first ``RenderProduct`` whose ``camera``
    relationship targets ``camera_prim_path``. If that ``RenderProduct`` has a
    child shader at ``<render_product>/<PPISP_SHADER_NAME>``, parses
    and returns the corresponding :class:`PpispCfg`. Returns ``None`` if
    no matching ``RenderProduct`` is found, or if it has no PPISP shader child.

    Args:
        stage: USD stage to search.
        camera_prim_path: Absolute camera prim path the ``RenderProduct``'s
            ``camera`` relationship must target.

    Returns:
        Parsed :class:`PpispCfg` if a matching shader was found, else ``None``.
    """
    from pxr import Sdf, UsdShade

    from .spg import find_ppisp_shader_prim

    target_path = Sdf.Path(camera_prim_path)
    for prim in stage.Traverse():
        if prim.GetTypeName() != "RenderProduct":
            continue
        camera_rel = prim.GetRelationship("camera")
        if not camera_rel:
            continue
        if target_path not in camera_rel.GetTargets():
            continue
        shader_prim = find_ppisp_shader_prim(stage, prim)
        if shader_prim is not None:
            return ppisp_cfg_from_usd_shader(
                UsdShade.Shader(shader_prim), load_controller_weights=load_controller_weights
            )
    return None


def auto_any_ppisp_cfg(stage: Any, *, load_controller_weights: bool = True) -> PpispCfg | None:
    """Find the first PPISP shader anywhere on ``stage``, regardless of binding.

    Walks ``stage`` looking for the first prim named
    :data:`PPISP_SHADER_NAME` that resolves to a valid ``UsdShade.Shader``.
    Returns ``None`` if no such shader exists.

    Used as a fallback when no ``RenderProduct`` binds a PPISP shader to a
    given camera but the scene still contains a PPISP configuration that
    should apply.

    Args:
        stage: USD stage to search.

    Returns:
        Parsed :class:`PpispCfg` for the first matching shader, else ``None``.
    """
    from pxr import UsdShade

    for prim in stage.Traverse():
        if prim.GetName() not in PPISP_SHADER_NAMES:
            continue
        shader = UsdShade.Shader(prim)
        if shader:
            return ppisp_cfg_from_usd_shader(shader, load_controller_weights=load_controller_weights)
    return None
