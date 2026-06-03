# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVRTX renderer output contract."""

import importlib.util
import types

import pytest
import torch
import warp as wp
from isaaclab_ppisp import PpispCfg

from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind, RenderBufferSpec
from isaaclab.sim import PinholeCameraCfg

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderData, OVRTXRenderer  # noqa: E402
else:
    OVRTXRenderData = None
    OVRTXRenderer = None
    OVRTXRendererCfg = None

_SPAWN = PinholeCameraCfg(
    focal_length=24.0,
    focus_distance=400.0,
    horizontal_aperture=20.955,
    clipping_range=(0.1, 1.0e5),
)


def _make_camera_cfg(data_types: list[str]) -> CameraCfg:
    return CameraCfg(
        height=8,
        width=16,
        prim_path="/World/Camera",
        spawn=_SPAWN,
        data_types=data_types,
    )


def _make_ovrtx_render_data() -> OVRTXRenderData:
    rd = OVRTXRenderData.__new__(OVRTXRenderData)
    rd.width = 16
    rd.height = 8
    rd.num_envs = 2
    rd.warp_buffers = {}
    rd.ppisp_pipeline = None
    return rd


def _make_ovrtx_renderer_without_backend() -> OVRTXRenderer:
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    return renderer


def test_ovrtx_renderer_config_enables_spg(monkeypatch):
    """OVRTX must set its SPG config bit so /rtx/spg/enabled is true in the native renderer."""
    from isaaclab_ov.renderers import ovrtx_renderer

    captured_config = {}

    class FakeRendererConfig:
        def __init__(self, **kwargs):
            captured_config.update(kwargs)

    class FakeRenderer:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(ovrtx_renderer, "RendererConfig", FakeRendererConfig)
    monkeypatch.setattr(ovrtx_renderer, "Renderer", FakeRenderer)

    cfg = OVRTXRendererCfg()
    cfg.use_ovrtx_cloning = False
    renderer = OVRTXRenderer(cfg)

    assert renderer._renderer
    assert captured_config["enable_spg"] is True


def test_ovrtx_prepare_cameras_reloads_non_native_ppisp_for_warp(monkeypatch):
    """Non-native PPISP must be resolved with controller weights for Warp fallback."""
    import isaaclab_ppisp

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
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)

    renderer.prepare_cameras(stage, spec)

    assert spec.cfg.isp_cfg is full_resolved_cfg
    assert calls == [
        ("native", requested_cfg, stage, "/World/Camera"),
        ("full", requested_cfg, stage, "/World/Camera"),
    ]
    assert exposure_overrides == [(stage, ["/World/Camera"])]


def test_ovrtx_supported_output_types_key_set():
    """OVRTX publishes the documented key set and per-output spec."""
    renderer = _make_ovrtx_renderer_without_backend()
    specs = renderer.supported_output_types()

    assert set(specs.keys()) == {
        RenderBufferKind.RGB,
        RenderBufferKind.RGBA,
        RenderBufferKind.RGB_HDR,
        RenderBufferKind.ALBEDO,
        RenderBufferKind.SIMPLE_SHADING_CONSTANT_DIFFUSE,
        RenderBufferKind.SIMPLE_SHADING_DIFFUSE_MDL,
        RenderBufferKind.SIMPLE_SHADING_FULL_MDL,
        RenderBufferKind.SEMANTIC_SEGMENTATION,
        RenderBufferKind.DEPTH,
        RenderBufferKind.DISTANCE_TO_IMAGE_PLANE,
        RenderBufferKind.DISTANCE_TO_CAMERA,
    }
    assert specs[RenderBufferKind.RGBA] == RenderBufferSpec(4, wp.uint8)
    assert specs[RenderBufferKind.RGB_HDR] == RenderBufferSpec(3, wp.float32)
    assert specs[RenderBufferKind.DEPTH] == RenderBufferSpec(1, wp.float32)


def test_ovrtx_set_outputs_wraps_caller_torch_zero_copy():
    """OVRTXRenderer.set_outputs publishes warp views over the caller's warp storage."""
    renderer = _make_ovrtx_renderer_without_backend()

    if not torch.cuda.is_available():
        pytest.skip("OVRTX zero-copy wrapping requires a CUDA device")
    device = "cuda"

    cfg = _make_camera_cfg(["rgb", "rgba", "depth"])
    data = CameraData.allocate(
        data_types=cfg.data_types,
        height=8,
        width=16,
        num_views=2,
        device=device,
        supported_specs=renderer.supported_output_types(),
    )
    render_data = _make_ovrtx_render_data()
    renderer.set_outputs(render_data, data.output)

    assert set(render_data.warp_buffers.keys()) >= {"rgba", "depth"}
    assert render_data.warp_buffers["rgba"].ptr == data.output["rgba"].warp.ptr
    assert render_data.warp_buffers["depth"].ptr == data.output["depth"].warp.ptr
    assert "rgb" not in render_data.warp_buffers


def test_ovrtx_set_outputs_wraps_requested_rgb_hdr_output():
    """OVRTXRenderer.set_outputs publishes a zero-copy view for requested RGB_HDR."""
    renderer = _make_ovrtx_renderer_without_backend()

    if not torch.cuda.is_available():
        pytest.skip("OVRTX zero-copy wrapping requires a CUDA device")
    device = "cuda"

    cfg = _make_camera_cfg(["rgb_hdr"])
    data = CameraData.allocate(
        data_types=cfg.data_types,
        height=8,
        width=16,
        num_views=2,
        device=device,
        supported_specs=renderer.supported_output_types(),
    )
    render_data = _make_ovrtx_render_data()
    renderer.set_outputs(render_data, data.output)

    assert render_data.warp_buffers["rgb_hdr"].ptr == data.output["rgb_hdr"].warp.ptr


def test_ovrtx_set_outputs_routes_ppisp_buffers_through_warp_buffers():
    """OVRTXRenderer.set_outputs stores PPISP source/destination in warp_buffers."""
    renderer = _make_ovrtx_renderer_without_backend()

    cfg = _make_camera_cfg(["rgb"])
    data = CameraData.allocate(
        data_types=cfg.data_types,
        height=8,
        width=16,
        num_views=2,
        device="cpu",
        supported_specs=renderer.supported_output_types(),
    )
    render_data = _make_ovrtx_render_data()
    render_data.ppisp_pipeline = object()
    renderer.set_outputs(render_data, data.output)

    assert render_data.warp_buffers["rgba"].ptr == data.output["rgba"].warp.ptr
    assert "rgb_hdr" in render_data.warp_buffers
    assert render_data.warp_buffers["rgb_hdr"].shape == (2, 8, 16, 3)
    assert render_data.warp_buffers["rgb_hdr"].dtype is wp.float32


def test_ovrtx_render_data_uses_native_spg_without_warp_pipeline():
    cfg = _make_camera_cfg(["rgb"])
    cfg.isp_cfg = PpispCfg(
        spg_render_product_prim_path="/Render/Source",
    )
    spec = CameraRenderSpec(
        cfg=cfg,
        device="cpu",
        num_instances=2,
        camera_prim_paths=("/World/envs/env_0/Camera", "/World/envs/env_1/Camera"),
        view_count=2,
        camera_path_relative_to_env_0="Camera",
    )

    render_data = OVRTXRenderData(spec, "cpu")

    assert render_data.ppisp_pipeline is None
    assert render_data.data_types == ["rgb"]


def test_ovrtx_process_frame_skips_ldr_rgba_when_ppisp_is_active():
    """PPISP owns RGBA output, so OVRTX LdrColor should not pre-fill it."""

    class FailingRenderVar:
        def map(self, *args, **kwargs):
            raise AssertionError("PPISP RGBA output must not read OVRTX LdrColor")

    class Frame:
        render_vars = {"LdrColor": FailingRenderVar()}

    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
    render_data.ppisp_pipeline = object()

    renderer._process_render_frame(render_data, Frame(), {"rgba": object()})


def test_ovrtx_ppisp_hdr_source_is_cloned_to_output_device(monkeypatch):
    """PPISP HdrColor source is moved to the HDR output buffer device."""

    class FakeArray:
        device = "cuda:1"

    class OutputArray:
        device = "cuda:0"

    cloned = object()
    clone_calls = []

    def fake_clone(src, *, device):
        clone_calls.append((src, device))
        return cloned

    monkeypatch.setattr(wp, "clone", fake_clone)

    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
    render_data.ppisp_pipeline = object()
    source = FakeArray()

    assert renderer._prepare_ppisp_hdr_source(render_data, source, {"rgb_hdr": OutputArray()}) is cloned
    assert clone_calls == [(source, "cuda:0")]


def test_ovrtx_read_output_is_a_no_op_after_consolidation():
    """OVRTXRenderer.read_output is a no-op once set_outputs wires up zero-copy."""
    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
    camera_data = CameraData()
    camera_data.info = {}
    camera_data._output = {}

    result = renderer.read_output(render_data, camera_data)
    assert result is None
    assert render_data.warp_buffers == {}
    assert camera_data.info == {}
    assert camera_data.output == {}
