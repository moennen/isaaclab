# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPISP controller USD source-asset parsing helpers."""

from __future__ import annotations

import logging
import os
import re
import zipfile
from typing import Any

from .cfg import (
    PPISP_AUTO_SHADER_NAME,
    PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN,
    PPISP_CONTROLLER_PARAMS_INPUT,
    PpispControllerCfg,
    _read_first_authored_value,
)

logger = logging.getLogger(__name__)

PPISP_CONTROLLER_SOURCE_ASSET_ATTR = "info:spg:sourceAsset"
PPISP_CONTROLLER_SOURCE_SUB_IDENTIFIER_ATTR = "info:spg:sourceAsset:subIdentifier"
PPISP_CONTROLLER_SUB_IDENTIFIER = "controllerProcess"
PPISP_CONTROLLER_WEIGHTS_SYMBOL = "kControllerWeights"


def ppisp_controller_cfg_from_auto_shader(shader: Any, *, load_controller_weights: bool) -> PpispControllerCfg | None:
    """Parse the controller stage connected to a ``PPISPAuto`` shader."""
    if shader.GetPrim().GetName() != PPISP_AUTO_SHADER_NAME:
        return None
    controller_shader = _find_controller_shader_for_auto_shader(shader)
    if controller_shader is None:
        return None

    controller_cfg = PpispControllerCfg()
    prior_input = controller_shader.GetInput("priorExposure")
    if prior_input:
        prior = _read_first_authored_value(prior_input.GetAttr())
        if prior is not None:
            controller_cfg.prior_exposure = float(prior)

    source_asset_attr = controller_shader.GetPrim().GetAttribute(PPISP_CONTROLLER_SOURCE_ASSET_ATTR)
    if source_asset_attr and load_controller_weights:
        source_text = _read_shader_source_asset(shader.GetPrim().GetStage(), source_asset_attr.Get())
        controller_cfg.weights = _extract_controller_weights(source_text)
    return controller_cfg


def _find_controller_shader_for_auto_shader(auto_shader: Any) -> Any | None:
    from pxr import UsdShade

    controller_input = auto_shader.GetInput(PPISP_CONTROLLER_PARAMS_INPUT)
    if controller_input:
        for connection in controller_input.GetAttr().GetConnections():
            render_var_prim = auto_shader.GetPrim().GetStage().GetPrimAtPath(connection.GetPrimPath())
            if not render_var_prim or not render_var_prim.IsValid():
                continue
            aov_attr = render_var_prim.GetAttribute("omni:rtx:aov")
            if not aov_attr:
                continue
            for source_connection in aov_attr.GetConnections():
                shader_prim = auto_shader.GetPrim().GetStage().GetPrimAtPath(source_connection.GetPrimPath())
                shader = UsdShade.Shader(shader_prim)
                if shader and _is_controller_shader(shader):
                    return shader

    parent = auto_shader.GetPrim().GetParent()
    if parent:
        for child in parent.GetChildren():
            shader = UsdShade.Shader(child)
            if shader and _is_controller_shader(shader):
                return shader
    return None


def _is_controller_shader(shader: Any) -> bool:
    prim = shader.GetPrim()
    if not prim.GetName().startswith("PPISPController_"):
        return False
    sub_identifier_attr = prim.GetAttribute(PPISP_CONTROLLER_SOURCE_SUB_IDENTIFIER_ATTR)
    return not sub_identifier_attr or sub_identifier_attr.Get() == PPISP_CONTROLLER_SUB_IDENTIFIER


def _asset_path(asset: Any) -> str | None:
    if asset is None:
        return None
    return str(getattr(asset, "path", asset))


def _asset_resolved_path(asset: Any) -> str | None:
    resolved = getattr(asset, "resolvedPath", None)
    return str(resolved) if resolved else None


def _read_shader_source_asset(stage: Any, asset: Any) -> str:
    asset_path = _asset_path(asset)
    if not asset_path:
        raise ValueError("PPISP controller shader is missing info:spg:sourceAsset.")

    resolved_path = _asset_resolved_path(asset)
    candidates = _source_asset_candidates(stage, asset_path, resolved_path)
    for candidate in candidates:
        text = _try_read_source_asset_candidate(candidate)
        if text is not None:
            return text
    raise ValueError(
        f"Unable to read PPISP controller source asset: {asset_path} "
        f"(tried {len(candidates)} candidate path(s); enable debug logging for details)."
    )


def _source_asset_candidates(stage: Any, asset_path: str, resolved_path: str | None) -> list[str]:
    candidates: list[str] = []
    if resolved_path:
        candidates.append(resolved_path)
    candidates.append(asset_path)

    root_layer = stage.GetRootLayer()
    root_identifier = getattr(root_layer, "realPath", None) or getattr(root_layer, "identifier", "")
    if root_identifier:
        package_path = _strip_package_member(root_identifier)
        if package_path.endswith(".usdz"):
            candidates.append(f"{package_path}[{asset_path}]")
        root_dir = os.path.dirname(package_path)
        if root_dir and not os.path.isabs(asset_path) and "://" not in asset_path:
            candidates.append(os.path.join(root_dir, asset_path))

    try:
        from pxr import Ar

        resolver = Ar.GetResolver()
        context = stage.GetPathResolverContext()
        with Ar.ResolverContextBinder(context):
            resolved = resolver.Resolve(asset_path)
        if resolved:
            candidates.insert(0, str(resolved))
    except Exception:
        logger.debug("Unable to resolve PPISP controller source asset through Ar: %s", asset_path, exc_info=True)

    out: list[str] = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            out.append(candidate)
            seen.add(candidate)
    return out


def _try_read_source_asset_candidate(candidate: str) -> str | None:
    if candidate.startswith("file://"):
        candidate = candidate[len("file://") :]
    package_match = re.match(r"^(?P<package>.+\.usdz)\[(?P<member>.+)\]$", candidate)
    if package_match:
        package = package_match.group("package")
        member = package_match.group("member")
        if os.path.exists(package):
            try:
                with zipfile.ZipFile(package) as zf:
                    return zf.read(member).decode("utf-8")
            except (KeyError, OSError, UnicodeDecodeError, zipfile.BadZipFile):
                logger.debug("Unable to read PPISP controller source asset candidate: %s", candidate, exc_info=True)
                return None
    if "://" in candidate:
        try:
            import omni.client

            read_result = omni.client.read_file(candidate)
            if len(read_result) == 2:
                result, data = read_result
            else:
                result, _, data = read_result
            if result == omni.client.Result.OK:
                return bytes(data).decode("utf-8")
            logger.debug(
                "omni.client did not read PPISP controller source asset candidate %s: %s",
                candidate,
                result,
            )
        except Exception:
            logger.debug("Unable to read PPISP controller source asset candidate: %s", candidate, exc_info=True)
            return None
    if os.path.exists(candidate):
        try:
            with open(candidate, encoding="utf-8") as f:
                return f.read()
        except OSError:
            logger.debug("Unable to read PPISP controller source asset candidate: %s", candidate, exc_info=True)
            return None
    return None


def _strip_package_member(identifier: str) -> str:
    package_match = re.match(r"^(.+\.usdz)(?:\[.+\])?$", identifier)
    return package_match.group(1) if package_match else identifier


def _extract_controller_weights(source_text: str) -> tuple[float, ...]:
    pattern = (
        rf"{re.escape(PPISP_CONTROLLER_WEIGHTS_SYMBOL)}\s*\[\s*TOTAL_WEIGHTS\s*\]\s*=\s*"
        r"\{(?P<body>.*?)\};"
    )
    match = re.search(pattern, source_text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Unable to find {PPISP_CONTROLLER_WEIGHTS_SYMBOL} in PPISP controller source.")
    values = tuple(
        float(token[:-1] if token[-1] in "fF" else token)
        for token in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?[fF]?", match.group("body"))
    )
    if len(values) != PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN:
        raise ValueError(
            f"Expected {PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN} PPISP controller weights, got {len(values)}."
        )
    return values
