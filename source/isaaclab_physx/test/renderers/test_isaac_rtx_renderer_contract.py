# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Isaac RTX renderer output contract."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import warp as wp
from isaaclab_ppisp import PpispCfg
from packaging import version

from isaaclab.renderers import RenderBufferKind, RenderBufferSpec

pytestmark = pytest.mark.isaacsim_ci


def _install_omni_stubs(monkeypatch):
    omni_module = sys.modules.get("omni", types.ModuleType("omni"))
    replicator_module = types.ModuleType("omni.replicator")
    replicator_core_module = types.ModuleType("omni.replicator.core")
    syntheticdata_module = types.ModuleType("omni.syntheticdata")
    usd_module = MagicMock()

    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.replicator", replicator_module)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", replicator_core_module)
    monkeypatch.setitem(sys.modules, "omni.syntheticdata", syntheticdata_module)
    monkeypatch.setitem(sys.modules, "omni.usd", usd_module)
    monkeypatch.setattr(omni_module, "replicator", replicator_module, raising=False)
    monkeypatch.setattr(omni_module, "syntheticdata", syntheticdata_module, raising=False)
    monkeypatch.setattr(omni_module, "usd", usd_module, raising=False)
    monkeypatch.setattr(replicator_module, "core", replicator_core_module, raising=False)

    return replicator_core_module, syntheticdata_module


def test_isaac_rtx_supported_output_types_include_rgb_hdr(monkeypatch):
    """Isaac RTX advertises RGB_HDR as a 3-channel float renderer output."""
    _install_omni_stubs(monkeypatch)
    from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderer
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

    renderer = IsaacRtxRenderer.__new__(IsaacRtxRenderer)
    renderer.cfg = IsaacRtxRendererCfg()
    with patch("isaaclab_physx.renderers.isaac_rtx_renderer.get_isaac_sim_version", return_value=version.parse("6.0")):
        specs = renderer.supported_output_types()

    assert specs[RenderBufferKind.RGB_HDR] == RenderBufferSpec(3, wp.float32)


def test_isaac_rtx_native_spg_runtime_enables_extension_and_setting(monkeypatch):
    """Native PPISP SPG requires both the Kit extension and the runtime setting."""
    app_module = types.ModuleType("isaacsim.core.experimental.utils.app")
    extension_calls = []

    def enable_extension(name: str) -> None:
        extension_calls.append(name)

    app_module.enable_extension = enable_extension
    for module_name in (
        "isaacsim",
        "isaacsim.core",
        "isaacsim.core.experimental",
        "isaacsim.core.experimental.utils",
    ):
        monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))
    monkeypatch.setitem(sys.modules, "isaacsim.core.experimental.utils.app", app_module)

    class Settings:
        def __init__(self):
            self.calls = []

        def set_bool(self, path: str, value: bool) -> None:
            self.calls.append((path, value))

    settings = Settings()

    from isaaclab_physx.renderers import isaac_rtx_renderer

    monkeypatch.setattr(isaac_rtx_renderer, "get_settings_manager", lambda: settings)

    isaac_rtx_renderer._enable_native_spg_runtime()

    assert extension_calls == [isaac_rtx_renderer.SPG_EXTENSION_NAME]
    assert settings.calls == [
        (isaac_rtx_renderer.SPG_ENABLED_SETTING, True),
        (isaac_rtx_renderer.SPG_ENABLED_SETTING, True),
    ]


def test_isaac_rtx_prepare_cameras_reloads_non_native_ppisp_for_warp(monkeypatch):
    """Non-native PPISP must be resolved with controller weights for Warp fallback."""
    import isaaclab_ppisp
    from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderer

    requested_cfg = object()
    native_resolved_cfg = PpispCfg()
    full_resolved_cfg = PpispCfg(inputs={"exposureOffset": 1.0})
    calls = []
    exposure_overrides = []

    def resolve_native(isp_cfg, stage, camera_prim_path):
        calls.append(("native", isp_cfg, stage, camera_prim_path))
        return native_resolved_cfg

    def resolve_full(isp_cfg, stage, camera_prim_path):
        calls.append(("full", isp_cfg, stage, camera_prim_path))
        return full_resolved_cfg

    monkeypatch.setattr(isaaclab_ppisp, "resolve_and_normalize_for_native_spg", resolve_native)
    monkeypatch.setattr(isaaclab_ppisp, "resolve_and_normalize", resolve_full)
    monkeypatch.setattr(
        isaaclab_ppisp,
        "apply_rtx_exposure_overrides",
        lambda stage, camera_paths: exposure_overrides.append((stage, camera_paths)),
    )

    stage = object()
    spec = types.SimpleNamespace(
        cfg=types.SimpleNamespace(isp_cfg=requested_cfg),
        camera_prim_paths=("/World/Camera",),
    )
    renderer = IsaacRtxRenderer.__new__(IsaacRtxRenderer)

    renderer.prepare_cameras(stage, spec)

    assert spec.cfg.isp_cfg is full_resolved_cfg
    assert calls == [
        ("native", requested_cfg, stage, "/World/Camera"),
        ("full", requested_cfg, stage, "/World/Camera"),
    ]
    assert exposure_overrides == [(stage, ["/World/Camera"])]
