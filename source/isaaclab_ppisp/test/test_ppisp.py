# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PPISP USD parsing helpers."""

import pytest
from isaaclab_ppisp import (
    PpispCfg,
    auto_camera_ppisp_cfg,
    copy_ppisp_spg_to_render_product,
    normalize_ppisp_cfg,
    ppisp_cfg_from_usd_shader,
    ppisp_uses_native_spg,
)
from isaaclab_ppisp.cfg import PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN

from pxr import Gf, Sdf, Usd, UsdShade


def _define_auto_ppisp_graph(stage: Usd.Stage, source_asset: str = "ppisp_controller_0.cu") -> str:
    camera = "/World/Camera"
    stage.DefinePrim(camera, "Camera")

    render_product = stage.DefinePrim("/Render/Source", "RenderProduct")
    render_product.CreateRelationship("camera").SetTargets([Sdf.Path(camera)])
    render_product.CreateRelationship("orderedVars").SetTargets(
        [
            Sdf.Path("/Render/Source/HdrColor"),
            Sdf.Path("/Render/Source/ControllerParams"),
            Sdf.Path("/Render/Source/LdrColor"),
        ]
    )

    hdr = stage.DefinePrim("/Render/Source/HdrColor", "RenderVar")
    hdr.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque)
    hdr.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("HdrColor")

    controller = UsdShade.Shader.Define(stage, "/Render/Source/PPISPController_0")
    controller.CreateInput("priorExposure", Sdf.ValueTypeNames.Float).Set(1.25)
    controller.CreateOutput("ControllerParams", Sdf.ValueTypeNames.Opaque)
    controller_prim = controller.GetPrim()
    controller_prim.CreateAttribute("info:spg:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(source_asset))
    controller_prim.CreateAttribute("info:spg:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set(
        "controllerProcess"
    )

    controller_params = stage.DefinePrim("/Render/Source/ControllerParams", "RenderVar")
    controller_params.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("ControllerParams")
    controller_params.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque).SetConnections(
        [Sdf.Path("/Render/Source/PPISPController_0.outputs:ControllerParams")]
    )

    auto_shader = UsdShade.Shader.Define(stage, "/Render/Source/PPISPAuto")
    auto_shader.CreateInput("responsivity", Sdf.ValueTypeNames.Float).Set(2.0)
    auto_shader.CreateInput("ControllerParams", Sdf.ValueTypeNames.Opaque).GetAttr().SetConnections(
        [Sdf.Path("/Render/Source/ControllerParams.omni:rtx:aov")]
    )
    auto_shader.CreateInput("HdrColor", Sdf.ValueTypeNames.Opaque).GetAttr().SetConnections(
        [Sdf.Path("/Render/Source/HdrColor.omni:rtx:aov")]
    )
    auto_shader.CreateOutput("PPISPColor", Sdf.ValueTypeNames.Opaque)

    ldr = stage.DefinePrim("/Render/Source/LdrColor", "RenderVar")
    ldr.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("LdrColor")
    ldr.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque).SetConnections(
        [Sdf.Path("/Render/Source/PPISPAuto.outputs:PPISPColor")]
    )
    return camera


def test_ppisp_shader_import_uses_first_time_sample():
    stage = Usd.Stage.CreateInMemory()
    shader = UsdShade.Shader.Define(stage, "/Render/RenderProduct/PPISP")

    exposure = shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float).GetAttr()
    exposure.Set(1.0)
    exposure.Set(2.0, 10.0)
    exposure.Set(3.0, 20.0)

    color = shader.CreateInput("colorLatentBlue", Sdf.ValueTypeNames.Float2).GetAttr()
    color.Set(Gf.Vec2f(0.0, 0.0))
    color.Set(Gf.Vec2f(0.1, 0.2), 5.0)

    cfg = ppisp_cfg_from_usd_shader(shader)

    assert cfg.inputs["exposureOffset"] == 2.0
    assert cfg.inputs["colorLatentBlue"] == pytest.approx((0.1, 0.2))


def test_normalize_ppisp_cfg_imports_render_product_shader_from_stage():
    stage = Usd.Stage.CreateInMemory()
    shader = UsdShade.Shader.Define(stage, "/Render/RenderProduct/PPISP")
    shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float).Set(1.5)
    shader.CreateInput("colorLatentRed", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.25, -0.5))

    cfg = normalize_ppisp_cfg(PpispCfg(spg_render_product_prim_path="/Render/RenderProduct"), stage=stage)

    assert cfg.inputs["exposureOffset"] == 1.5
    assert cfg.inputs["colorLatentRed"] == pytest.approx((0.25, -0.5))


def test_normalize_ppisp_cfg_applies_explicit_overrides_after_shader_import():
    stage = Usd.Stage.CreateInMemory()
    shader = UsdShade.Shader.Define(stage, "/Render/RenderProduct/PPISP")
    shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float).Set(1.5)
    shader.CreateInput("colorLatentRed", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.25, -0.5))

    cfg = normalize_ppisp_cfg(
        PpispCfg(
            spg_render_product_prim_path="/Render/RenderProduct",
            inputs={"exposureOffset": 2.0},
        ),
        stage=stage,
    )

    assert cfg.inputs["exposureOffset"] == 2.0
    assert cfg.inputs["colorLatentRed"] == pytest.approx((0.25, -0.5))


def test_auto_camera_ppisp_cfg_skips_matching_render_product_without_ppisp():
    """Camera-bound discovery skips generated RenderProducts that lack PPISP.

    Isaac RTX can create a transient RenderProduct that targets the same camera
    as an authored scene RenderProduct. If the transient prim has no ``PPISP``
    child, discovery must keep scanning instead of concluding the camera has no
    camera-bound PPISP shader.
    """
    stage = Usd.Stage.CreateInMemory()
    camera = "/World/Camera"
    stage.DefinePrim(camera, "Camera")

    plain_rp = stage.DefinePrim("/Render/GeneratedRenderProduct", "RenderProduct")
    plain_rp.CreateRelationship("camera").SetTargets([Sdf.Path(camera)])

    authored_rp = stage.DefinePrim("/World/Render/RenderProduct", "RenderProduct")
    authored_rp.CreateRelationship("camera").SetTargets([Sdf.Path(camera)])
    shader = UsdShade.Shader.Define(stage, "/World/Render/RenderProduct/PPISP")
    shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float).Set(1.5)

    cfg = auto_camera_ppisp_cfg(stage, camera)

    assert cfg is not None
    assert cfg.inputs["exposureOffset"] == 1.5


def test_auto_camera_ppisp_cfg_discovers_controller_graph_without_loading_weights():
    stage = Usd.Stage.CreateInMemory()
    camera = _define_auto_ppisp_graph(stage)

    cfg = auto_camera_ppisp_cfg(stage, camera, load_controller_weights=False)

    assert cfg is not None
    assert cfg.spg_render_product_prim_path == "/Render/Source"
    assert cfg.inputs["responsivity"] == 2.0
    assert ppisp_uses_native_spg(cfg)
    assert cfg.controller is not None
    assert cfg.controller.prior_exposure == pytest.approx(1.25)
    assert cfg.controller.weights is None


def test_auto_camera_ppisp_cfg_loads_embedded_controller_weights(tmp_path):
    weights_source = tmp_path / "ppisp_controller_0.cu"
    weights_source.write_text(
        "static __device__ const float kControllerWeights[TOTAL_WEIGHTS] = {"
        + ("0.0f," * PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN)
        + "};",
        encoding="utf-8",
    )
    stage = Usd.Stage.CreateInMemory()
    camera = _define_auto_ppisp_graph(stage, str(weights_source))

    cfg = auto_camera_ppisp_cfg(stage, camera)

    assert cfg is not None
    assert cfg.controller is not None
    assert cfg.controller.weights is not None
    assert len(cfg.controller.weights) == PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN


def test_copy_ppisp_spg_to_render_product_rewrites_connections_and_replaces_duplicate_ldr_var():
    stage = Usd.Stage.CreateInMemory()
    camera = _define_auto_ppisp_graph(stage)
    cfg = auto_camera_ppisp_cfg(stage, camera, load_controller_weights=False)
    assert cfg is not None

    stage.DefinePrim("/Render/Target", "RenderProduct")
    generated_ldr = stage.DefinePrim("/Render/Vars/LdrColor", "RenderVar")
    generated_ldr.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("LdrColor")
    depth = stage.DefinePrim("/Render/Vars/Depth", "RenderVar")
    depth.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("DistanceToImagePlaneSD")
    stage.GetPrimAtPath("/Render/Target").CreateRelationship("orderedVars").SetTargets(
        [Sdf.Path("/Render/Vars/LdrColor"), Sdf.Path("/Render/Vars/Depth")]
    )

    copy_ppisp_spg_to_render_product(stage, cfg.spg_render_product_prim_path, "/Render/Target")

    copied_auto = stage.GetPrimAtPath("/Render/Target/PPISPAuto")
    assert copied_auto.IsValid()
    assert copied_auto.GetAttribute("inputs:HdrColor").GetConnections() == [
        Sdf.Path("/Render/Target/HdrColor.omni:rtx:aov")
    ]
    assert copied_auto.GetAttribute("inputs:ControllerParams").GetConnections() == [
        Sdf.Path("/Render/Target/ControllerParams.omni:rtx:aov")
    ]

    ordered_vars = stage.GetPrimAtPath("/Render/Target").GetRelationship("orderedVars").GetTargets()
    assert Sdf.Path("/Render/Vars/Depth") in ordered_vars
    assert Sdf.Path("/Render/Vars/LdrColor") not in ordered_vars
    assert ordered_vars[-3:] == [
        Sdf.Path("/Render/Target/HdrColor"),
        Sdf.Path("/Render/Target/ControllerParams"),
        Sdf.Path("/Render/Target/LdrColor"),
    ]


def test_copy_ppisp_spg_to_render_product_follows_authored_graph_connections():
    stage = Usd.Stage.CreateInMemory()
    camera = _define_auto_ppisp_graph(stage)

    extra_shader = UsdShade.Shader.Define(stage, "/Render/Source/CustomPreprocess")
    extra_shader.CreateInput("Input", Sdf.ValueTypeNames.Opaque).GetAttr().SetConnections(
        [Sdf.Path("/Render/Source/HdrColor.omni:rtx:aov")]
    )
    extra_shader.CreateOutput("Output", Sdf.ValueTypeNames.Opaque)
    stage.GetPrimAtPath("/Render/Source/PPISPAuto").GetAttribute("inputs:HdrColor").SetConnections(
        [Sdf.Path("/Render/Source/CustomPreprocess.outputs:Output")]
    )
    cfg = auto_camera_ppisp_cfg(stage, camera, load_controller_weights=False)
    assert cfg is not None
    assert cfg.spg_render_product_prim_path == "/Render/Source"
    stage.DefinePrim("/Render/Target", "RenderProduct")

    copy_ppisp_spg_to_render_product(stage, cfg.spg_render_product_prim_path, "/Render/Target")

    copied_extra = stage.GetPrimAtPath("/Render/Target/CustomPreprocess")
    assert copied_extra.IsValid()
    assert copied_extra.GetAttribute("inputs:Input").GetConnections() == [
        Sdf.Path("/Render/Target/HdrColor.omni:rtx:aov")
    ]
    assert stage.GetPrimAtPath("/Render/Target/PPISPAuto").GetAttribute("inputs:HdrColor").GetConnections() == [
        Sdf.Path("/Render/Target/CustomPreprocess.outputs:Output")
    ]


def test_copy_ppisp_spg_to_render_product_follows_relative_sibling_connections():
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Camera", "Camera")

    source = stage.DefinePrim("/Render/Source", "RenderProduct")
    source.CreateRelationship("camera").SetTargets([Sdf.Path("/World/Camera")])
    source.CreateRelationship("orderedVars").SetTargets(
        [Sdf.Path("/Render/Source/HdrColor"), Sdf.Path("/Render/Source/LdrColor")]
    )

    hdr = stage.DefinePrim("/Render/Source/HdrColor", "RenderVar")
    hdr.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("HdrColor")
    hdr.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False)

    shader = UsdShade.Shader.Define(stage, "/Render/Source/PPISP")
    shader.CreateInput("HdrColor", Sdf.ValueTypeNames.Opaque).GetAttr().SetConnections(
        [Sdf.Path("../HdrColor.omni:rtx:aov")]
    )
    shader.CreateOutput("PPISPColor", Sdf.ValueTypeNames.Opaque)

    ldr = stage.DefinePrim("/Render/Source/LdrColor", "RenderVar")
    ldr.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("LdrColor")
    ldr.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False).SetConnections(
        [Sdf.Path("/Render/Source/PPISP.outputs:PPISPColor")]
    )

    target = stage.DefinePrim("/Render/Target", "RenderProduct")
    target.CreateRelationship("orderedVars")

    cfg = ppisp_cfg_from_usd_shader(shader, load_controller_weights=False)
    assert cfg.spg_render_product_prim_path == "/Render/Source"

    copy_ppisp_spg_to_render_product(stage, cfg.spg_render_product_prim_path, "/Render/Target")

    assert stage.GetPrimAtPath("/Render/Target/HdrColor").IsValid()
    assert stage.GetPrimAtPath("/Render/Target/LdrColor").IsValid()
    assert stage.GetPrimAtPath("/Render/Target/PPISP").GetAttribute("inputs:HdrColor").GetConnections() == [
        Sdf.Path("../HdrColor.omni:rtx:aov")
    ]
    assert stage.GetPrimAtPath("/Render/Target/LdrColor").GetAttribute("omni:rtx:aov").GetConnections() == [
        Sdf.Path("/Render/Target/PPISP.outputs:PPISPColor")
    ]
