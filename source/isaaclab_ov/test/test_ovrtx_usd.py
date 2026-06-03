# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVRTX USD render product authoring."""

import importlib.util

import pytest

_REQUIRED_MODULES = ("isaaclab_ov", "pxr")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers.ovrtx_usd import (  # noqa: E402
        build_render_product_on_stage,
        get_render_var_config,
        get_render_var_configs,
        stage_from_string,
    )

    from pxr import Sdf, Usd  # noqa: E402
else:
    Sdf = None
    Usd = None
    build_render_product_on_stage = None
    get_render_var_config = None
    get_render_var_configs = None
    stage_from_string = None


def test_ovrtx_rgb_hdr_uses_hdr_color_render_var():
    """Requesting RGB_HDR from OVRTX selects the HdrColor render variable."""
    assert get_render_var_config(["rgb_hdr"]) == ("/Render/Vars/HdrColor", "HdrColor", "HdrColor")


def test_ovrtx_rgb_and_rgb_hdr_author_both_render_vars():
    """Requesting LDR RGB and RGB_HDR keeps both OVRTX render variables."""
    render_var_configs = get_render_var_configs(["rgb", "rgb_hdr"])

    assert render_var_configs == [
        ("/Render/Vars/LdrColor", "LdrColor", "LdrColor"),
        ("/Render/Vars/HdrColor", "HdrColor", "HdrColor"),
    ]

    stage = Usd.Stage.CreateInMemory()
    render_product_path = build_render_product_on_stage(
        stage=stage,
        width=16,
        height=8,
        num_envs=1,
        data_types=["rgb", "rgb_hdr"],
        camera_rel_path="Camera",
    )

    assert stage.GetPrimAtPath(render_product_path).GetRelationship("orderedVars").GetTargets() == [
        Sdf.Path("/Render/Vars/LdrColor"),
        Sdf.Path("/Render/Vars/HdrColor"),
    ]
    assert stage.GetPrimAtPath("/Render/Vars/LdrColor").IsValid()
    assert stage.GetPrimAtPath("/Render/Vars/HdrColor").IsValid()


def test_ovrtx_build_render_product_on_stage_authors_equivalent_render_vars():
    """The stage-authoring path creates the expected OVRTX RenderProduct prims."""
    stage = Usd.Stage.CreateInMemory()

    render_product_path = build_render_product_on_stage(
        stage=stage,
        width=16,
        height=8,
        num_envs=1,
        data_types=["rgb", "rgb_hdr"],
        camera_rel_path="Camera",
        render_product_name="IsaacLabRenderProduct",
    )

    render_product = stage.GetPrimAtPath(render_product_path)
    assert render_product.IsValid()
    assert render_product.GetRelationship("camera").GetTargets() == [Sdf.Path("/World/envs/env_0/Camera")]
    assert render_product.GetRelationship("orderedVars").GetTargets() == [
        Sdf.Path("/Render/Vars/LdrColor"),
        Sdf.Path("/Render/Vars/HdrColor"),
    ]
    assert stage.GetPrimAtPath("/Render/Vars/LdrColor").GetAttribute("sourceName").Get() == "LdrColor"
    assert stage.GetPrimAtPath("/Render/Vars/HdrColor").GetAttribute("sourceName").Get() == "HdrColor"


def test_ovrtx_build_render_product_on_temp_stage_keeps_source_stage_unchanged():
    """Native-SPG render product injection happens on export scratch USD, not the live stage."""
    source_stage = Usd.Stage.CreateInMemory()
    source_stage.DefinePrim("/World/envs/env_0/Camera", "Camera")
    temp_stage = stage_from_string(source_stage.ExportToString())

    build_render_product_on_stage(
        stage=temp_stage,
        width=16,
        height=8,
        num_envs=1,
        data_types=["rgb"],
        camera_rel_path="Camera",
        render_product_name="IsaacLabRenderProduct",
    )

    assert not source_stage.GetPrimAtPath("/Render/IsaacLabRenderProduct").IsValid()
    assert temp_stage.GetPrimAtPath("/Render/IsaacLabRenderProduct").IsValid()
