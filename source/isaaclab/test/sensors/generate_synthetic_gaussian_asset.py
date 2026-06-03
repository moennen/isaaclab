# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a tiny synthetic Gaussian-Splat USD asset for camera PPISP tests.

Avoids dependencies on heavyweight Nucleus assets by authoring a few large
opaque gaussians of known colors, bound to ``ParticleFieldEmissive.mdl`` with
``apply_inverse_tonemap=0`` and ``apply_srgb_linear=0`` so the wrapper PPISP is
the sole ISP authority. Tests assert *semantic invariants* of the PPISP
behavior (non-degenerate LDR output from renderer HDR, vignetting darkens
corners, the CRF keeps values bounded, etc.) instead of doing a
fidelity-against-baked comparison — which sidesteps cross-renderer
HDR-magnitude calibration drift entirely.
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from isaaclab_ppisp import PpispCfg, normalize_ppisp_cfg

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.camera.camera_isp import CameraISPMode
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from collections.abc import Iterator
    from os import PathLike

    from isaaclab.renderers.renderer_cfg import RendererCfg
    from isaaclab.sim import SimulationCfg, SimulationContext


# SH degree-0 evaluation constant ``Y_0 = 1 / (2 * sqrt(pi))``. The standard
# 3DGS convention encodes a particle's base color as
# ``color = 0.5 + Y_0 * dc`` so inverting gives ``dc = (color - 0.5) / Y_0``.
_SH_Y0 = 1.0 / (2.0 * math.sqrt(math.pi))


@dataclass
class SyntheticGaussian:
    """One opaque gaussian in the synthetic scene."""

    position: tuple[float, float, float]
    """World-space position (x, y, z) in metres."""

    color: tuple[float, float, float]
    """Target final color in [0, 1] linear scene-referred space. Encoded into SH so
    the rendered (pre-PPISP) HDR pixels at the gaussian center approximate this color."""

    scale: float = 0.3
    """Isotropic scale (radius) of the gaussian ellipsoid in metres."""

    opacity: float = 1.0
    """Opacity in [0, 1]. Use 1.0 for fully opaque coverage."""


@dataclass
class SyntheticGaussianScene:
    """Scene description consumed by :func:`make_synthetic_gaussian_usd`.

    Defaults arrange four large fully-opaque gaussians (R, G, B, W) in a 2x2
    grid in the X-Y plane at Z=0, with a camera placed on +Z looking at the
    grid origin.
    """

    gaussians: list[SyntheticGaussian] = field(
        default_factory=lambda: [
            SyntheticGaussian(position=(-0.6, +0.6, 0.0), color=(0.9, 0.1, 0.1)),  # red
            SyntheticGaussian(position=(+0.6, +0.6, 0.0), color=(0.1, 0.9, 0.1)),  # green
            SyntheticGaussian(position=(-0.6, -0.6, 0.0), color=(0.1, 0.1, 0.9)),  # blue
            SyntheticGaussian(position=(+0.6, -0.6, 0.0), color=(0.9, 0.9, 0.9)),  # white
        ]
    )
    """Gaussians in the scene. Default forms a 2x2 RGBW grid."""

    camera_position: tuple[float, float, float] = (0.0, 0.0, 3.0)
    """Camera position. Default looks at the grid origin from +Z."""

    focal_length: float = 24.0
    """Pinhole camera focal length in mm."""

    horizontal_aperture: float = 20.955
    """Camera horizontal aperture in mm."""


def make_synthetic_gaussian_usd(path: str, scene: SyntheticGaussianScene | None = None) -> str:
    """Author a tiny gaussian-splat USD at ``path`` and return that path.

    The asset references ``ParticleFieldEmissive.mdl`` with ``apply_inverse_tonemap=0``
    and ``apply_srgb_linear=0`` so the wrapper PPISP is the sole ISP authority.
    The default prim is ``World``; cameras live at ``/World/Cameras/test_cam``
    and the gaussians at ``/World/Scene/gaussians/Gaussians/gaussians``.
    """
    if scene is None:
        scene = SyntheticGaussianScene()

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    stage = Usd.Stage.CreateNew(path)
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Default prim ``/World`` so this asset can be referenced under any parent.
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # Camera under ``/World/Cameras/test_cam``. Authored without time samples so
    # ``UsdGeom.XformCache`` resolves the world transform at the default time.
    UsdGeom.Xform.Define(stage, "/World/Cameras")
    cam = UsdGeom.Camera.Define(stage, "/World/Cameras/test_cam")
    cam.GetFocalLengthAttr().Set(scene.focal_length)
    cam.GetHorizontalApertureAttr().Set(scene.horizontal_aperture)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
    cam.GetFStopAttr().Set(1.0)
    # Look from camera_position toward origin (-Z view direction).
    cx, cy, cz = scene.camera_position
    cam.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))

    # Gaussian particle field. Use a deeply nested path matching the typical
    # 3DGS export layout so user-side scene wiring matches real assets.
    UsdGeom.Xform.Define(stage, "/World/Scene")
    UsdGeom.Xform.Define(stage, "/World/Scene/gaussians")
    UsdGeom.Xform.Define(stage, "/World/Scene/gaussians/Gaussians")
    gauss_prim_path = "/World/Scene/gaussians/Gaussians/gaussians"
    gauss_prim = stage.DefinePrim(gauss_prim_path, "ParticleField3DGaussianSplat")

    def _attr(name: str, type_name: Sdf.ValueTypeName, value):
        attr = gauss_prim.CreateAttribute(name, type_name)
        attr.Set(value)
        return attr

    _attr("positions", Sdf.ValueTypeNames.Point3fArray, [Gf.Vec3f(*g.position) for g in scene.gaussians])
    _attr(
        "orientations",
        Sdf.ValueTypeNames.QuatfArray,
        # Identity quaternion (w, x, y, z) - Gf.Quatf takes (real, imaginary).
        [Gf.Quatf(1.0, 0.0, 0.0, 0.0) for _ in scene.gaussians],
    )
    _attr(
        "scales",
        Sdf.ValueTypeNames.Float3Array,
        [Gf.Vec3f(g.scale, g.scale, g.scale) for g in scene.gaussians],
    )
    _attr("opacities", Sdf.ValueTypeNames.FloatArray, [float(g.opacity) for g in scene.gaussians])

    # Encode the desired final color in SH degree-0 coefficients. With
    # apply_inverse_tonemap=0 and apply_srgb_linear=0, the MDL evaluates the
    # gaussian color as ``0.5 + Y_0 * dc * emission_intensity``. Solving for dc
    # gives the inverse encoding used here.
    sh_coeffs = [
        Gf.Vec3f(
            (g.color[0] - 0.5) / _SH_Y0,
            (g.color[1] - 0.5) / _SH_Y0,
            (g.color[2] - 0.5) / _SH_Y0,
        )
        for g in scene.gaussians
    ]
    _attr("radiance:sphericalHarmonicsCoefficients", Sdf.ValueTypeNames.Float3Array, sh_coeffs)
    sh_degree_attr = gauss_prim.CreateAttribute(
        "radiance:sphericalHarmonicsDegree", Sdf.ValueTypeNames.Int, custom=False
    )
    sh_degree_attr.Set(0)

    # Conservative extent — bounding box of all gaussian centers expanded by
    # their largest scale.
    if scene.gaussians:
        max_scale = max(g.scale for g in scene.gaussians)
        positions = [g.position for g in scene.gaussians]
        lo = (
            min(p[0] for p in positions) - max_scale,
            min(p[1] for p in positions) - max_scale,
            min(p[2] for p in positions) - max_scale,
        )
        hi = (
            max(p[0] for p in positions) + max_scale,
            max(p[1] for p in positions) + max_scale,
            max(p[2] for p in positions) + max_scale,
        )
        gauss_prim.CreateAttribute("extent", Sdf.ValueTypeNames.Float3Array).Set([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])

    # Material binding: ``ParticleFieldEmissive.mdl`` with the two boolean
    # ``apply_*`` inputs set to false so the wrapper PPISP is the sole ISP
    # authority and the gaussian color comes out of the renderer as linear
    # scene-referred radiance.
    UsdGeom.Xform.Define(stage, "/World/Scene/gaussians/Looks")
    material = stage.DefinePrim("/World/Scene/gaussians/Looks/ParticleFieldEmissive", "Material")
    shader = stage.DefinePrim("/World/Scene/gaussians/Looks/ParticleFieldEmissive/Shader", "Shader")
    shader.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set("sourceAsset")
    shader.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set("ParticleFieldEmissive.mdl")
    shader.CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set("ParticleFieldEmissive")
    shader.CreateAttribute("inputs:apply_inverse_tonemap", Sdf.ValueTypeNames.Bool, custom=True).Set(False)
    shader.CreateAttribute("inputs:apply_srgb_linear", Sdf.ValueTypeNames.Bool, custom=True).Set(False)
    shader.CreateAttribute("outputs:out", Sdf.ValueTypeNames.Token, custom=True)
    for output in ("mdl:displacement", "mdl:surface", "mdl:volume"):
        material.CreateAttribute(f"outputs:{output}", Sdf.ValueTypeNames.Token).AddConnection(
            shader.GetPath().AppendProperty("outputs:out")
        )
    gauss_prim.CreateRelationship("material:binding").SetTargets([material.GetPath()])

    stage.GetRootLayer().Save()
    return path


# PPISP cfg helpers ----------------------------------------------------------


# Strong negative radial coefficient — the warp kernel uses
# ``factor = clamp(1 + alpha1 * r^2 + alpha2 * r^4 + alpha3 * r^6, 0, 1)`` where
# ``r`` is normalised by ``max(W, H)``. With this value the corner of a square
# frame (``r^2 = 0.5``) attenuates by ``factor = 1 + (-1.5)(0.5) = 0.25``, i.e.
# corners drop to ~25% of center intensity. Visible but not fully black.
_AGGRESSIVE_VIGNETTING_ALPHA1 = -1.8

# Negative exposure offset (input × 2^-5 = ÷32) tuned so the aggressive cfg
# safely brings the RTX-bearing backends' gaussian HDR magnitudes (~10
# single-tile, ~17 multi-tile observed on OVRTX) below the CRF's [0,1] clamp
# before tonemapping. Newton's much lower native HDR scale is normalised
# separately via :func:`make_aggressive_ppisp_cfg`'s ``responsivity`` kwarg.
_AGGRESSIVE_EXPOSURE_OFFSET = -5.0

_PPISP_SPG_SOURCE_RENDER_PRODUCT_PATH = "/Render/PPISPSource"

_PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN = 241_961
_PPISP_CONTROLLER_EMBEDDED_WEIGHTS_MARKER = "// __PPISP_CONTROLLER_EMBEDDED_WEIGHTS__"
_PPISP_CONTROLLER_OFF_CONV1_W = 0
_PPISP_CONTROLLER_OFF_CONV1_B = _PPISP_CONTROLLER_OFF_CONV1_W + 16 * 3
_PPISP_CONTROLLER_OFF_CONV2_W = _PPISP_CONTROLLER_OFF_CONV1_B + 16
_PPISP_CONTROLLER_OFF_CONV2_B = _PPISP_CONTROLLER_OFF_CONV2_W + 32 * 16
_PPISP_CONTROLLER_OFF_CONV3_W = _PPISP_CONTROLLER_OFF_CONV2_B + 32
_PPISP_CONTROLLER_OFF_CONV3_B = _PPISP_CONTROLLER_OFF_CONV3_W + 64 * 32
_PPISP_CONTROLLER_OFF_TRUNK0_W = _PPISP_CONTROLLER_OFF_CONV3_B + 64
_PPISP_CONTROLLER_OFF_TRUNK0_B = _PPISP_CONTROLLER_OFF_TRUNK0_W + 128 * 1601
_PPISP_CONTROLLER_OFF_TRUNK1_W = _PPISP_CONTROLLER_OFF_TRUNK0_B + 128
_PPISP_CONTROLLER_OFF_TRUNK1_B = _PPISP_CONTROLLER_OFF_TRUNK1_W + 128 * 128
_PPISP_CONTROLLER_OFF_TRUNK2_W = _PPISP_CONTROLLER_OFF_TRUNK1_B + 128
_PPISP_CONTROLLER_OFF_TRUNK2_B = _PPISP_CONTROLLER_OFF_TRUNK2_W + 128 * 128
_PPISP_CONTROLLER_OFF_EXP_W = _PPISP_CONTROLLER_OFF_TRUNK2_B + 128
_PPISP_CONTROLLER_OFF_EXP_B = _PPISP_CONTROLLER_OFF_EXP_W + 128
_PPISP_CONTROLLER_OFF_COL_W = _PPISP_CONTROLLER_OFF_EXP_B + 1
_PPISP_CONTROLLER_OFF_COL_B = _PPISP_CONTROLLER_OFF_COL_W + 8 * 128

_PPISP_CONTROLLER_TOTAL_WEIGHTS = _PPISP_CONTROLLER_OFF_COL_B + 8
if _PPISP_CONTROLLER_TOTAL_WEIGHTS != _PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN:
    raise RuntimeError(
        "Synthetic PPISP controller fixture offsets are inconsistent: "
        f"{_PPISP_CONTROLLER_TOTAL_WEIGHTS} != {_PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN}."
    )

_SYNTHETIC_STATIC_PPISP_CUDA = """
static __device__ __forceinline__ float clamp01(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

static __device__ __forceinline__ unsigned char toU8(float value) {
    return static_cast<unsigned char>(clamp01(value) * 255.0f);
}

static __device__ __forceinline__ float applyVignetting(
    float value,
    float u,
    float v,
    float centerX,
    float centerY,
    float alpha1,
    float alpha2,
    float alpha3) {
    const float dx = u - centerX;
    const float dy = v - centerY;
    const float r2 = dx * dx + dy * dy;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;
    return value * clamp01(1.0f + alpha1 * r2 + alpha2 * r4 + alpha3 * r6);
}

extern "C" __global__ void ppispProcess(
    int width,
    int height,
    cudaTextureObject_t inHdrColor,
    const float* __restrict__ params,
    cudaSurfaceObject_t outPPISPColor) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float4 pixel = tex2D<float4>(inHdrColor, x, y);
    const float exposureScale = params[0] * exp2f(params[1]);
    float r = pixel.x * exposureScale;
    float g = pixel.y * exposureScale;
    float b = pixel.z * exposureScale;

    const float maxRes = fmaxf(float(width), float(height));
    const float u = (float(x) + 0.5f - float(width) * 0.5f) / maxRes;
    const float v = (float(y) + 0.5f - float(height) * 0.5f) / maxRes;
    r = applyVignetting(r, u, v, params[2], params[3], params[4], params[5], params[6]);
    g = applyVignetting(g, u, v, params[7], params[8], params[9], params[10], params[11]);
    b = applyVignetting(b, u, v, params[12], params[13], params[14], params[15], params[16]);

    const uchar4 out = {toU8(r), toU8(g), toU8(b), 255};
    surf2Dwrite<uchar4>(out, outPPISPColor, x * int(sizeof(uchar4)), y);
}
""".lstrip()

_SYNTHETIC_STATIC_PPISP_LUA = """
function ppispProcess(inputs, outputs)
    local in_hdr = inputs["HdrColor"]
    assert(in_hdr and in_hdr.rank == 2, "HdrColor input must be a 2D texture")

    local height = in_hdr.shape[1]
    local width = in_hdr.shape[2]
    outputs["PPISPColor"] = cuda.image(width, height, cuda.uchar4)

    local function getFloat(name, default)
        return cuda.float(inputs[name] or default).value
    end

    local function getFloat2(name)
        local value = inputs[name]
        local packed = value and cuda.float2(value) or cuda.float2(0.0, 0.0)
        return packed.value
    end

    local vignettingCenterR = getFloat2("vignettingCenterR")
    local vignettingCenterG = getFloat2("vignettingCenterG")
    local vignettingCenterB = getFloat2("vignettingCenterB")
    local params = {
        getFloat("responsivity", 1.0),
        getFloat("exposureOffset", 0.0),
        vignettingCenterR[1],
        vignettingCenterR[2],
        getFloat("vignettingAlpha1R", 0.0),
        getFloat("vignettingAlpha2R", 0.0),
        getFloat("vignettingAlpha3R", 0.0),
        vignettingCenterG[1],
        vignettingCenterG[2],
        getFloat("vignettingAlpha1G", 0.0),
        getFloat("vignettingAlpha2G", 0.0),
        getFloat("vignettingAlpha3G", 0.0),
        vignettingCenterB[1],
        vignettingCenterB[2],
        getFloat("vignettingAlpha1B", 0.0),
        getFloat("vignettingAlpha2B", 0.0),
        getFloat("vignettingAlpha3B", 0.0),
    }

    return cuda.kernel({
        args = {
            cuda.int(width),
            cuda.int(height),
            cuda.TextureObject(in_hdr),
            cuda.array(params, cuda.float),
            cuda.SurfaceObject(outputs["PPISPColor"]),
        },
        block = { 16, 16, 1 },
        grid = { math.ceil(width / 16), math.ceil(height / 16), 1 },
    })
end
""".lstrip()

_SYNTHETIC_STATIC_PPISP_USDA = """
#usda 1.0
(
    defaultPrim = "PPISP"
)

def Shader "PPISP"
{
    uniform token info:implementationSource = "sourceAsset"
    uniform asset info:spg:sourceAsset = @./ppisp_usd_spg.cu@
    uniform token info:spg:sourceAsset:subIdentifier = "ppispProcess"

    float inputs:responsivity = 1.0
    float inputs:exposureOffset = 0.0

    float2 inputs:vignettingCenterR = (0.0, 0.0)
    float inputs:vignettingAlpha1R = 0.0
    float inputs:vignettingAlpha2R = 0.0
    float inputs:vignettingAlpha3R = 0.0

    float2 inputs:vignettingCenterG = (0.0, 0.0)
    float inputs:vignettingAlpha1G = 0.0
    float inputs:vignettingAlpha2G = 0.0
    float inputs:vignettingAlpha3G = 0.0

    float2 inputs:vignettingCenterB = (0.0, 0.0)
    float inputs:vignettingAlpha1B = 0.0
    float inputs:vignettingAlpha2B = 0.0
    float inputs:vignettingAlpha3B = 0.0

    float2 inputs:colorLatentBlue = (0.0, 0.0)
    float2 inputs:colorLatentRed = (0.0, 0.0)
    float2 inputs:colorLatentGreen = (0.0, 0.0)
    float2 inputs:colorLatentNeutral = (0.0, 0.0)

    float inputs:crfToeR = 0.013659
    float inputs:crfShoulderR = 0.013659
    float inputs:crfGammaR = 0.378165
    float inputs:crfCenterR = 0.0

    float inputs:crfToeG = 0.013659
    float inputs:crfShoulderG = 0.013659
    float inputs:crfGammaG = 0.378165
    float inputs:crfCenterG = 0.0

    float inputs:crfToeB = 0.013659
    float inputs:crfShoulderB = 0.013659
    float inputs:crfGammaB = 0.378165
    float inputs:crfCenterB = 0.0

    opaque inputs:HdrColor
    opaque outputs:PPISPColor
}
""".lstrip()

_SYNTHETIC_CONTROLLER_CUDA_TEMPLATE = f"""
static const int POOL_FEATURE_LEN = 1600;
static const int OFF_CONV1_W = {_PPISP_CONTROLLER_OFF_CONV1_W};
static const int OFF_CONV1_B = {_PPISP_CONTROLLER_OFF_CONV1_B};
static const int OFF_CONV2_W = {_PPISP_CONTROLLER_OFF_CONV2_W};
static const int OFF_CONV2_B = {_PPISP_CONTROLLER_OFF_CONV2_B};
static const int OFF_CONV3_W = {_PPISP_CONTROLLER_OFF_CONV3_W};
static const int OFF_CONV3_B = {_PPISP_CONTROLLER_OFF_CONV3_B};
static const int OFF_TRUNK0_W = {_PPISP_CONTROLLER_OFF_TRUNK0_W};
static const int OFF_TRUNK0_B = {_PPISP_CONTROLLER_OFF_TRUNK0_B};
static const int OFF_TRUNK1_W = {_PPISP_CONTROLLER_OFF_TRUNK1_W};
static const int OFF_TRUNK1_B = {_PPISP_CONTROLLER_OFF_TRUNK1_B};
static const int OFF_TRUNK2_W = {_PPISP_CONTROLLER_OFF_TRUNK2_W};
static const int OFF_TRUNK2_B = {_PPISP_CONTROLLER_OFF_TRUNK2_B};
static const int OFF_EXP_W = {_PPISP_CONTROLLER_OFF_EXP_W};
static const int OFF_EXP_B = {_PPISP_CONTROLLER_OFF_EXP_B};
static const int OFF_COL_W = {_PPISP_CONTROLLER_OFF_COL_W};
static const int OFF_COL_B = {_PPISP_CONTROLLER_OFF_COL_B};
static const int TOTAL_WEIGHTS = {_PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN};
{_PPISP_CONTROLLER_EMBEDDED_WEIGHTS_MARKER}

extern "C" __global__ void controllerPoolProcess(
    int inputWidth,
    int inputHeight,
    cudaTextureObject_t inHdrColor,
    float* __restrict__ outControllerFeatures) {{
    const int globalThread = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = int(gridDim.x * blockDim.x);
    for (int i = globalThread; i < POOL_FEATURE_LEN; i += stride) {{
        outControllerFeatures[i] = 0.0f;
    }}
}}

extern "C" __global__ void controllerProcess(
    const float* __restrict__ controllerFeatures,
    float priorExposure,
    float* __restrict__ outControllerParams) {{
    const int tid = int(threadIdx.x);
    if (tid == 0) {{
        outControllerParams[0] = kControllerWeights[OFF_EXP_B];
    }}
    if (tid < 8) {{
        outControllerParams[1 + tid] = kControllerWeights[OFF_COL_B + tid];
    }}
}}
""".lstrip()

_SYNTHETIC_CONTROLLER_LUA = """
function controllerPoolProcess(inputs, outputs)
    local in_hdr = inputs["HdrColor"]
    assert(in_hdr ~= nil, "controllerPoolProcess: HdrColor input is missing")
    assert(in_hdr.rank == 2, "controllerPoolProcess: HdrColor must be a 2D image")

    outputs["ControllerFeatures"] = cuda.empty({ 1, 1600 }, cuda.float)

    return cuda.kernel({
        args = {
            cuda.int(in_hdr.shape[2]),
            cuda.int(in_hdr.shape[1]),
            cuda.TextureObject(in_hdr),
            cuda.array(outputs["ControllerFeatures"], cuda.float),
        },
        block = { 256, 1, 1 },
        grid = { 25, 1, 1 },
    })
end

function controllerProcess(inputs, outputs)
    local features = inputs["ControllerFeatures"]
    assert(features ~= nil, "controllerProcess: ControllerFeatures input is missing")

    outputs["ControllerParams"] = cuda.empty({ 1, 9 }, cuda.float)

    return cuda.kernel({
        args = {
            cuda.array(features, cuda.float),
            cuda.float(inputs["priorExposure"] or 0.0),
            cuda.array(outputs["ControllerParams"], cuda.float),
        },
        block = { 128, 1, 1 },
        grid = { 1, 1, 1 },
    })
end
""".lstrip()

_SYNTHETIC_AUTO_PPISP_CUDA = """
static __device__ __forceinline__ float clamp01(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

static __device__ __forceinline__ unsigned char toU8(float value) {
    return static_cast<unsigned char>(clamp01(value) * 255.0f);
}

static __device__ __forceinline__ float applyVignetting(
    float value,
    float u,
    float v,
    float centerX,
    float centerY,
    float alpha1,
    float alpha2,
    float alpha3) {
    const float dx = u - centerX;
    const float dy = v - centerY;
    const float r2 = dx * dx + dy * dy;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;
    return value * clamp01(1.0f + alpha1 * r2 + alpha2 * r4 + alpha3 * r6);
}

extern "C" __global__ void ppispProcessAuto(
    int width,
    int height,
    cudaTextureObject_t inHdrColor,
    const float* __restrict__ controllerParams,
    const float* __restrict__ params,
    cudaSurfaceObject_t outPPISPColor) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float4 pixel = tex2D<float4>(inHdrColor, x, y);
    const float exposureScale = params[0] * exp2f(controllerParams[0]);
    float r = pixel.x * exposureScale;
    float g = pixel.y * exposureScale;
    float b = pixel.z * exposureScale;

    const float maxRes = fmaxf(float(width), float(height));
    const float u = (float(x) + 0.5f - float(width) * 0.5f) / maxRes;
    const float v = (float(y) + 0.5f - float(height) * 0.5f) / maxRes;
    r = applyVignetting(r, u, v, params[1], params[2], params[3], params[4], params[5]);
    g = applyVignetting(g, u, v, params[6], params[7], params[8], params[9], params[10]);
    b = applyVignetting(b, u, v, params[11], params[12], params[13], params[14], params[15]);

    const uchar4 out = {toU8(r), toU8(g), toU8(b), 255};
    surf2Dwrite<uchar4>(out, outPPISPColor, x * int(sizeof(uchar4)), y);
}
""".lstrip()

_SYNTHETIC_AUTO_PPISP_LUA = """
function ppispProcessAuto(inputs, outputs)
    local in_hdr = inputs["HdrColor"]
    assert(in_hdr and in_hdr.rank == 2, "HdrColor input must be a 2D texture")

    local controller = inputs["ControllerParams"]
    assert(controller, "ppispProcessAuto needs a ControllerParams input")

    local height = in_hdr.shape[1]
    local width = in_hdr.shape[2]
    outputs["PPISPColor"] = cuda.image(width, height, cuda.uchar4)

    local function getFloat(name, default)
        return cuda.float(inputs[name] or default).value
    end

    local function getFloat2(name)
        local value = inputs[name]
        local packed = value and cuda.float2(value) or cuda.float2(0.0, 0.0)
        return packed.value
    end

    local vignettingCenterR = getFloat2("vignettingCenterR")
    local vignettingCenterG = getFloat2("vignettingCenterG")
    local vignettingCenterB = getFloat2("vignettingCenterB")
    local params = {
        getFloat("responsivity", 1.0),
        vignettingCenterR[1],
        vignettingCenterR[2],
        getFloat("vignettingAlpha1R", 0.0),
        getFloat("vignettingAlpha2R", 0.0),
        getFloat("vignettingAlpha3R", 0.0),
        vignettingCenterG[1],
        vignettingCenterG[2],
        getFloat("vignettingAlpha1G", 0.0),
        getFloat("vignettingAlpha2G", 0.0),
        getFloat("vignettingAlpha3G", 0.0),
        vignettingCenterB[1],
        vignettingCenterB[2],
        getFloat("vignettingAlpha1B", 0.0),
        getFloat("vignettingAlpha2B", 0.0),
        getFloat("vignettingAlpha3B", 0.0),
    }

    return cuda.kernel({
        args = {
            cuda.int(width),
            cuda.int(height),
            cuda.TextureObject(in_hdr),
            cuda.array(controller, cuda.float),
            cuda.array(params, cuda.float),
            cuda.SurfaceObject(outputs["PPISPColor"]),
        },
        block = { 16, 16, 1 },
        grid = { math.ceil(width / 16), math.ceil(height / 16), 1 },
    })
end
""".lstrip()

_SYNTHETIC_AUTO_PPISP_USDA = """
#usda 1.0
(
    defaultPrim = "PPISPAuto"
)

def Shader "PPISPAuto"
{
    uniform token info:implementationSource = "sourceAsset"
    uniform asset info:spg:sourceAsset = @./ppisp_usd_spg_auto.cu@
    uniform token info:spg:sourceAsset:subIdentifier = "ppispProcessAuto"

    float inputs:responsivity = 1.0

    float2 inputs:vignettingCenterR = (0.0, 0.0)
    float inputs:vignettingAlpha1R = 0.0
    float inputs:vignettingAlpha2R = 0.0
    float inputs:vignettingAlpha3R = 0.0

    float2 inputs:vignettingCenterG = (0.0, 0.0)
    float inputs:vignettingAlpha1G = 0.0
    float inputs:vignettingAlpha2G = 0.0
    float inputs:vignettingAlpha3G = 0.0

    float2 inputs:vignettingCenterB = (0.0, 0.0)
    float inputs:vignettingAlpha1B = 0.0
    float inputs:vignettingAlpha2B = 0.0
    float inputs:vignettingAlpha3B = 0.0

    float inputs:crfToeR = 0.013659
    float inputs:crfShoulderR = 0.013659
    float inputs:crfGammaR = 0.378165
    float inputs:crfCenterR = 0.0

    float inputs:crfToeG = 0.013659
    float inputs:crfShoulderG = 0.013659
    float inputs:crfGammaG = 0.378165
    float inputs:crfCenterG = 0.0

    float inputs:crfToeB = 0.013659
    float inputs:crfShoulderB = 0.013659
    float inputs:crfGammaB = 0.378165
    float inputs:crfCenterB = 0.0

    opaque inputs:HdrColor
    opaque inputs:ControllerParams
    opaque outputs:PPISPColor
}
""".lstrip()


def make_aggressive_ppisp_cfg(*, responsivity: float = 1.0) -> PpispCfg:
    """Return a :class:`~isaaclab_ppisp.PpispCfg` with every PPISP feature engaged enough
    to be assertable in a downstream test.

    Each input is dialed past the "subtle correction" defaults so an integration
    test can check semantic invariants of the wrapper PPISP pipeline:

    * **Exposure**: ``exposureOffset = -5`` (input × 2^-5 = ÷32) — tuned so that
      a near-typical RTX-style gaussian HDR magnitude (≈10–17) lands below the
      CRF's [0,1] clamp before tonemapping, then CRF compresses to upper LDR.
    * **Vignetting**: per-channel ``alpha1 = -1.8`` — corners drop to ~0% of
      center intensity for a square frame. Slight per-channel imbalance
      (R < G < B in alpha2) produces a non-uniform corner colour cast.
    * **Color homography**: ``red_latent`` pulls the red anchor outward and
      ``green_latent`` pulls the green anchor down — input white pixels acquire
      a visible warm hue shift.
    * **CRF**: per-channel toe/shoulder/gamma/center values that meaningfully
      compress highlights (no overflow above 1.0 ⇒ max LDR uint8 stays at 255
      only when the wrapper actually clamps; under-engaged CRF would let the
      explicit ``clamp(.., 0, 1)`` in the kernel do all the work).

    Args:
        responsivity: PPISP achromatic ``responsivity`` factor applied **before**
            exposure. Defaults to ``1.0`` (calibrated for RTX-bearing backends'
            HDR magnitude). The Newton backend produces a much lower-magnitude
            HDR for the same scene and tests pass a value > 1 to bring its
            effective signal in line with the RTX backends.
    """
    inputs: dict[str, float | tuple[float, float]] = {
        "responsivity": responsivity,
        "exposureOffset": _AGGRESSIVE_EXPOSURE_OFFSET,
        # Vignetting: identical optical center for all channels (image center),
        # with a slight per-channel falloff offset so a vignetted region has
        # a faint chromatic gradient — verifies the per-channel paths are wired.
        "vignettingCenterR": (0.0, 0.0),
        "vignettingAlpha1R": _AGGRESSIVE_VIGNETTING_ALPHA1,
        "vignettingAlpha2R": -0.4,
        "vignettingAlpha3R": 0.0,
        "vignettingCenterG": (0.0, 0.0),
        "vignettingAlpha1G": _AGGRESSIVE_VIGNETTING_ALPHA1,
        "vignettingAlpha2G": -0.2,
        "vignettingAlpha3G": 0.0,
        "vignettingCenterB": (0.0, 0.0),
        "vignettingAlpha1B": _AGGRESSIVE_VIGNETTING_ALPHA1,
        "vignettingAlpha2B": 0.0,
        "vignettingAlpha3B": 0.0,
        # Color homography: shift the red and green anchors so the output picks
        # up a clear hue rotation. Blue and neutral remain near identity.
        "colorLatentRed": (0.4, 0.0),
        "colorLatentGreen": (0.0, -0.4),
        "colorLatentBlue": (0.0, 0.0),
        "colorLatentNeutral": (0.0, 0.0),
        # CRF: stronger shoulder than the default highlight knee so a saturated
        # input is compressed rather than clipped. Per-channel gammas are split
        # to produce a subtle warm cast.
        "crfToeR": 0.05,
        "crfShoulderR": 0.20,
        "crfGammaR": 0.50,
        "crfCenterR": 0.0,
        "crfToeG": 0.05,
        "crfShoulderG": 0.20,
        "crfGammaG": 0.45,
        "crfCenterG": 0.0,
        "crfToeB": 0.05,
        "crfShoulderB": 0.20,
        "crfGammaB": 0.40,
        "crfCenterB": 0.0,
    }
    return normalize_ppisp_cfg(PpispCfg(inputs=inputs))


def make_neutral_ppisp_cfg(*, responsivity: float = 1.0) -> PpispCfg:
    """Return a mild static PPISP cfg used as the native-SPG negative control."""
    return normalize_ppisp_cfg(PpispCfg(inputs={"responsivity": responsivity, "exposureOffset": 0.0}))


def missing_ppisp_spg_sidecars() -> list[str]:
    """Return missing generated PPISP SPG sidecars.

    The sidecars are synthesized into each test's temporary directory, so
    there is no external Nucleus/local fixture dependency.
    """
    return []


def prepare_ppisp_spg_sidecars(
    directory: str | PathLike[str],
    *,
    controller_output_cfg: PpispCfg,
) -> str:
    """Generate PPISP-compatible SPG sidecars for native-renderer tests.

    The generated kernels intentionally keep the runtime fixture small: the
    PPISP node applies the assertable exposure/vignetting parts of the PPISP
    contract, and the controller graph writes deterministic exposure/color
    latents through the same ``ControllerParams`` path as exported graphs.
    The full PPISP math remains covered by the Warp kernel unit tests and the
    wrapper integration tests.
    """
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "ppisp_usd_spg.cu": _SYNTHETIC_STATIC_PPISP_CUDA,
        "ppisp_usd_spg.cu.lua": _SYNTHETIC_STATIC_PPISP_LUA,
        "ppisp_usd_spg.usda": _SYNTHETIC_STATIC_PPISP_USDA,
        "ppisp_controller.cu.lua": _SYNTHETIC_CONTROLLER_LUA,
        "ppisp_usd_spg_auto.cu": _SYNTHETIC_AUTO_PPISP_CUDA,
        "ppisp_usd_spg_auto.cu.lua": _SYNTHETIC_AUTO_PPISP_LUA,
        "ppisp_usd_spg_auto.usda": _SYNTHETIC_AUTO_PPISP_USDA,
    }
    for filename, contents in files.items():
        (output_dir / filename).write_text(contents, encoding="utf-8")

    controller_source = _render_deterministic_controller_source(
        _SYNTHETIC_CONTROLLER_CUDA_TEMPLATE, controller_output_cfg
    )
    (output_dir / "ppisp_controller.cu").write_text(controller_source, encoding="utf-8")
    (output_dir / "ppisp_controller_0.cu").write_text(controller_source, encoding="utf-8")
    (output_dir / "ppisp_controller_0.cu.lua").write_bytes((output_dir / "ppisp_controller.cu.lua").read_bytes())
    return str(output_dir)


def assert_images_meaningfully_different(
    reference_rgb: torch.Tensor,
    candidate_rgb: torch.Tensor,
    *,
    min_mean_abs_diff: float = 3.0,
    label: str = "",
) -> None:
    """Assert two LDR RGB tiles differ enough to prove a shader path changed output."""
    prefix = f"[{label}] " if label else ""
    diff = (reference_rgb[..., :3].float() - candidate_rgb[..., :3].float()).abs()
    mean_abs_diff = diff.mean().item()
    assert mean_abs_diff > min_mean_abs_diff, (
        f"{prefix}image difference too small: mean_abs_diff={mean_abs_diff:.3f}, "
        f"expected > {min_mean_abs_diff}. The SPG graph may not be applied."
    )


def assert_ppisp_controller_matches_static(
    static_rgb: torch.Tensor,
    controller_rgb: torch.Tensor,
    *,
    max_mean_abs_diff: float = 8.0,
    label: str = "",
) -> None:
    """Assert deterministic controller output matches the equivalent static PPISP graph."""
    prefix = f"[{label}] " if label else ""
    diff = (static_rgb[..., :3].float() - controller_rgb[..., :3].float()).abs()
    mean_abs_diff = diff.mean().item()
    assert mean_abs_diff < max_mean_abs_diff, (
        f"{prefix}controller PPISP differs from static reference: mean_abs_diff={mean_abs_diff:.3f}, "
        f"expected < {max_mean_abs_diff}."
    )


def _render_deterministic_controller_source(controller_template: str, ppisp_cfg: PpispCfg) -> str:
    if _PPISP_CONTROLLER_EMBEDDED_WEIGHTS_MARKER not in controller_template:
        raise ValueError("PPISP controller template is missing the embedded-weights marker.")

    inputs = ppisp_cfg.inputs
    weights = ["0.0f"] * _PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN
    weights[_PPISP_CONTROLLER_OFF_EXP_B] = _cuda_float_literal(float(inputs["exposureOffset"]))
    color_values = (
        *_float2(inputs["colorLatentBlue"]),
        *_float2(inputs["colorLatentRed"]),
        *_float2(inputs["colorLatentGreen"]),
        *_float2(inputs["colorLatentNeutral"]),
    )
    for i, value in enumerate(color_values):
        weights[_PPISP_CONTROLLER_OFF_COL_B + i] = _cuda_float_literal(value)

    lines = []
    for start in range(0, len(weights), 8):
        lines.append("    " + ", ".join(weights[start : start + 8]))
    weight_array = (
        f"static_assert(TOTAL_WEIGHTS == {_PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN}, "
        '"embedded PPISP controller weight count mismatch");\n'
        "static __device__ const float kControllerWeights[TOTAL_WEIGHTS] = {\n" + ",\n".join(lines) + "\n};"
    )
    return controller_template.replace(_PPISP_CONTROLLER_EMBEDDED_WEIGHTS_MARKER, weight_array)


def _cuda_float_literal(value: float) -> str:
    return f"{float(value):.9e}f"


def _float2(value: float | tuple[float, float]) -> tuple[float, float]:
    assert not isinstance(value, float)
    return (float(value[0]), float(value[1]))


def _camera_path_for_env(env_id: int = 0) -> str:
    return f"/World/envs/env_{env_id}/{SYNTHETIC_GAUSSIAN_SCENE_REL_PATH}/Cameras/{SYNTHETIC_GAUSSIAN_CAMERA_NAME}"


def _define_ppisp_source_render_product(
    stage: Usd.Stage,
    *,
    width: int,
    height: int,
) -> Usd.Prim:
    stage.DefinePrim("/Render", "Scope")
    render_product = stage.DefinePrim(_PPISP_SPG_SOURCE_RENDER_PRODUCT_PATH, "RenderProduct")
    render_product.CreateRelationship("camera").SetTargets([Sdf.Path(_camera_path_for_env(0))])
    render_product.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(width, height))

    hdr = stage.DefinePrim(f"{_PPISP_SPG_SOURCE_RENDER_PRODUCT_PATH}/HdrColor", "RenderVar")
    hdr.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("HdrColor")
    hdr.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False)
    render_product.CreateRelationship("orderedVars").SetTargets(
        [Sdf.Path(f"{_PPISP_SPG_SOURCE_RENDER_PRODUCT_PATH}/HdrColor")]
    )
    return render_product


def _author_source_asset(shader: UsdShade.Shader, source_path: Path, sub_identifier: str) -> None:
    prim = shader.GetPrim()
    prim.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token, custom=False).Set("sourceAsset")
    prim.CreateAttribute("info:spg:sourceAsset", Sdf.ValueTypeNames.Asset, custom=False).Set(
        Sdf.AssetPath(str(source_path))
    )
    prim.CreateAttribute("info:spg:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token, custom=False).Set(
        sub_identifier
    )


def _set_ppisp_inputs(shader: UsdShade.Shader, inputs: dict[str, float | tuple[float, float]]) -> None:
    for name, value in inputs.items():
        if isinstance(value, tuple):
            shader.CreateInput(name, Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(float(value[0]), float(value[1])))
        else:
            shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(float(value))


def _append_ordered_var(render_product: Usd.Prim, path: str) -> None:
    rel = render_product.CreateRelationship("orderedVars")
    targets = list(rel.GetTargets())
    sdf_path = Sdf.Path(path)
    if sdf_path not in targets:
        targets.append(sdf_path)
        rel.SetTargets(targets)


def _define_connected_render_var(
    stage: Usd.Stage,
    render_product: Usd.Prim,
    name: str,
    source: Sdf.Path,
) -> Usd.Prim:
    path = f"{_PPISP_SPG_SOURCE_RENDER_PRODUCT_PATH}/{name}"
    render_var = stage.DefinePrim(path, "RenderVar")
    render_var.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set(name)
    render_var.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False).SetConnections([source])
    _append_ordered_var(render_product, path)
    return render_var


def author_static_ppisp_spg(
    stage: Usd.Stage,
    *,
    sidecar_dir: str,
    ppisp_cfg: PpispCfg,
    width: int,
    height: int,
) -> str:
    """Author a source RenderProduct with the exported static PPISP SPG graph."""
    render_product = _define_ppisp_source_render_product(stage, width=width, height=height)
    sidecars = Path(sidecar_dir)

    shader = UsdShade.Shader.Define(stage, f"{_PPISP_SPG_SOURCE_RENDER_PRODUCT_PATH}/PPISP")
    shader.GetPrim().GetReferences().AddReference(str(sidecars / "ppisp_usd_spg.usda"))
    _author_source_asset(shader, sidecars / "ppisp_usd_spg.cu", "ppispProcess")
    shader.CreateInput("HdrColor", Sdf.ValueTypeNames.Opaque).GetAttr().SetConnections(
        [Sdf.Path("../HdrColor.omni:rtx:aov")]
    )
    shader.CreateOutput("PPISPColor", Sdf.ValueTypeNames.Opaque)
    _set_ppisp_inputs(shader, ppisp_cfg.inputs)
    _define_connected_render_var(
        stage,
        render_product,
        "LdrColor",
        shader.GetPath().AppendProperty("outputs:PPISPColor"),
    )
    return _PPISP_SPG_SOURCE_RENDER_PRODUCT_PATH


def author_controller_ppisp_spg(
    stage: Usd.Stage,
    *,
    sidecar_dir: str,
    ppisp_cfg: PpispCfg,
    width: int,
    height: int,
    prior_exposure: float = 0.0,
) -> str:
    """Author a source RenderProduct with controller + PPISPAuto SPG graph."""
    render_product = _define_ppisp_source_render_product(stage, width=width, height=height)
    sidecars = Path(sidecar_dir)

    pool_shader = UsdShade.Shader.Define(stage, f"{_PPISP_SPG_SOURCE_RENDER_PRODUCT_PATH}/PPISPControllerPool_0")
    _author_source_asset(pool_shader, sidecars / "ppisp_controller_0.cu", "controllerPoolProcess")
    pool_shader.CreateInput("HdrColor", Sdf.ValueTypeNames.Opaque).GetAttr().SetConnections(
        [Sdf.Path("../HdrColor.omni:rtx:aov")]
    )
    pool_shader.CreateOutput("ControllerFeatures", Sdf.ValueTypeNames.Opaque)
    _define_connected_render_var(
        stage,
        render_product,
        "ControllerFeatures",
        pool_shader.GetPath().AppendProperty("outputs:ControllerFeatures"),
    )

    controller_shader = UsdShade.Shader.Define(stage, f"{_PPISP_SPG_SOURCE_RENDER_PRODUCT_PATH}/PPISPController_0")
    _author_source_asset(controller_shader, sidecars / "ppisp_controller_0.cu", "controllerProcess")
    controller_shader.CreateInput("ControllerFeatures", Sdf.ValueTypeNames.Opaque).GetAttr().SetConnections(
        [Sdf.Path("../ControllerFeatures.omni:rtx:aov")]
    )
    controller_shader.CreateInput("priorExposure", Sdf.ValueTypeNames.Float).Set(float(prior_exposure))
    controller_shader.CreateOutput("ControllerParams", Sdf.ValueTypeNames.Opaque)
    _define_connected_render_var(
        stage,
        render_product,
        "ControllerParams",
        controller_shader.GetPath().AppendProperty("outputs:ControllerParams"),
    )

    auto_shader = UsdShade.Shader.Define(stage, f"{_PPISP_SPG_SOURCE_RENDER_PRODUCT_PATH}/PPISPAuto")
    auto_shader.GetPrim().GetReferences().AddReference(str(sidecars / "ppisp_usd_spg_auto.usda"))
    _author_source_asset(auto_shader, sidecars / "ppisp_usd_spg_auto.cu", "ppispProcessAuto")
    auto_shader.CreateInput("HdrColor", Sdf.ValueTypeNames.Opaque).GetAttr().SetConnections(
        [Sdf.Path("../HdrColor.omni:rtx:aov")]
    )
    auto_shader.CreateInput("ControllerParams", Sdf.ValueTypeNames.Opaque).GetAttr().SetConnections(
        [Sdf.Path("../ControllerParams.omni:rtx:aov")]
    )
    auto_shader.CreateOutput("PPISPColor", Sdf.ValueTypeNames.Opaque)
    _set_ppisp_inputs(
        auto_shader,
        {
            name: value
            for name, value in ppisp_cfg.inputs.items()
            if name != "exposureOffset" and not name.startswith("colorLatent")
        },
    )
    _define_connected_render_var(
        stage,
        render_product,
        "LdrColor",
        auto_shader.GetPath().AppendProperty("outputs:PPISPColor"),
    )
    return _PPISP_SPG_SOURCE_RENDER_PRODUCT_PATH


def assert_ppisp_invariants(
    rgb_tile: torch.Tensor,
    *,
    patch: int = 16,
    vignetting_corner_ratio_max: float = 0.5,
    label: str = "",
) -> None:
    """Assert the four PPISP signatures expected from :func:`make_aggressive_ppisp_cfg`
    on a single ``[H, W, C>=3]`` rgb tile (uint8-range floats).

    1. Non-degenerate frame: ``5 < mean < 250``.
    2. Vignetting: each of the 4 corner patches is below
       ``vignetting_corner_ratio_max`` times the center patch (``alpha1=-1.8``
       drives the pre-CRF corner factor to ~0; after CRF compression the
       per-renderer corner/center ratio sits well below 0.5).
    3. Exposure: center patch mean > 50 (the aggressive cfg's
       ``responsivity * 2^exposureOffset`` is tuned so the per-renderer HDR
       magnitude lands solidly into mid-to-upper LDR after CRF).
    4. CRF clamping: output stays in ``[0, 255]`` (also catches NaNs implicitly).

    ``label`` is included in every assertion message so the caller can identify
    which renderer / tile failed.
    """
    prefix = f"[{label}] " if label else ""
    h, w = rgb_tile.shape[:2]

    mean = rgb_tile.mean().item()
    assert 5.0 < mean < 250.0, f"{prefix}render is degenerate (mean={mean:.1f})"

    cy, cx = h // 2 - patch // 2, w // 2 - patch // 2
    center_mean = rgb_tile[cy : cy + patch, cx : cx + patch, :3].mean().item()
    assert center_mean > 1.0, f"{prefix}center patch is degenerate (mean={center_mean:.1f})"

    for corner_name, y0, x0 in (
        ("top-left", 0, 0),
        ("top-right", 0, w - patch),
        ("bottom-left", h - patch, 0),
        ("bottom-right", h - patch, w - patch),
    ):
        corner_mean = rgb_tile[y0 : y0 + patch, x0 : x0 + patch, :3].mean().item()
        ratio = corner_mean / center_mean
        assert ratio < vignetting_corner_ratio_max, (
            f"{prefix}vignetting too weak at {corner_name}: corner/center = {ratio:.3f} "
            f"(expected < {vignetting_corner_ratio_max}). "
            f"corner_mean={corner_mean:.1f}, center_mean={center_mean:.1f}"
        )

    assert center_mean > 50.0, (
        f"{prefix}aggressive PPISP cfg should land the center patch above 50 (mid-LDR); "
        f"got {center_mean:.1f}. Check responsivity/exposureOffset and that the renderer is producing HDR > 0."
    )

    assert rgb_tile.max().item() <= 255.0, f"{prefix}output overflow: max={rgb_tile.max().item():.1f}"
    assert rgb_tile.min().item() >= 0.0, f"{prefix}output underflow: min={rgb_tile.min().item():.1f}"


def assert_ppisp_lifts_exposure(
    hdr_tile: torch.Tensor,
    rgb_tile: torch.Tensor,
    *,
    patch: int = 16,
    hdr_center_min: float = 1.0e-2,
    ldr_center_norm_range: tuple[float, float] = (0.1, 0.95),
    label: str = "",
) -> None:
    """Assert PPISP normalises the renderer's HDR into a useful LDR range.

    Different renderer backends produce wildly different HDR magnitudes for the
    same synthetic gaussian scene (Newton's emissive scale is ~10× lower than
    the RTX backends'). The aggressive cfg's ``responsivity`` knob is tuned per
    backend to bring the effective signal in line; this assertion then only
    enforces that:

    * the renderer is producing an HDR AOV (lower bound on the HDR center)
    * PPISP delivers a non-degenerate LDR center (not black, not fully
      saturated)

    Localises failures:

    * **Renderer not producing HDR** — caught by the HDR lower bound.
    * **PPISP saturated or black** — caught by the LDR range bound; suggests
      either ``responsivity``/``exposureOffset`` mis-tuned or the pipeline
      not running.

    Args:
        hdr_tile: ``[H, W, 3]`` float HDR tile (from ``output["rgb_hdr"]``).
        rgb_tile: ``[H, W, C>=3]`` uint8-range float LDR tile (from ``output["rgb"]``).
        patch: Center-patch window size in pixels.
        hdr_center_min: Lower bound on the raw HDR center mean.
        ldr_center_norm_range: ``(min, max)`` for the LDR center mean / 255.
        label: Included in every assertion message so the caller can identify
            which renderer / tile failed.
    """
    prefix = f"[{label}] " if label else ""
    h, w = rgb_tile.shape[:2]
    cy, cx = h // 2 - patch // 2, w // 2 - patch // 2

    hdr_center = hdr_tile[cy : cy + patch, cx : cx + patch, :3].float().mean().item()
    ldr_center_norm = rgb_tile[cy : cy + patch, cx : cx + patch, :3].float().mean().item() / 255.0

    assert hdr_center > hdr_center_min, (
        f"{prefix}HDR center too dark (mean={hdr_center:.4f}) — renderer not producing the HDR AOV?"
    )
    ldr_lo, ldr_hi = ldr_center_norm_range
    assert ldr_lo < ldr_center_norm < ldr_hi, (
        f"{prefix}PPISP LDR center out of range: ldr_norm={ldr_center_norm:.4f} "
        f"(expected {ldr_lo} < x < {ldr_hi}; hdr_center={hdr_center:.4f}). "
        f"Likely saturated or black — check responsivity/exposureOffset tuning."
    )


# ──────────────────────────────────────────────────────────────────────────────
# InteractiveScene helpers shared by the Isaac RTX-, Newton-, and OVRTX-backed
# gaussian tests.
# ──────────────────────────────────────────────────────────────────────────────

SYNTHETIC_GAUSSIAN_SCENE_REL_PATH = "Scene"
"""Asset prim path under each environment in :class:`SyntheticGaussianSceneCfg`."""

SYNTHETIC_GAUSSIAN_CAMERA_NAME = "test_cam"
"""Camera prim name authored inside the synthesised asset USD."""

SYNTHETIC_GAUSSIAN_CAMERA_REGEX = (
    f"/World/envs/env_.*/{SYNTHETIC_GAUSSIAN_SCENE_REL_PATH}/Cameras/{SYNTHETIC_GAUSSIAN_CAMERA_NAME}"
)
"""Regex camera prim path that resolves to one camera per env (single or tiled)."""


@configclass
class SyntheticGaussianSceneCfg(InteractiveSceneCfg):
    """Minimal :class:`~isaaclab.scene.InteractiveScene` cfg wrapping the synthesised gaussian asset.

    The :attr:`anchor` rigid body exists solely to give Newton-backed physics
    a non-empty body table — it is invisible at the camera viewpoint and far
    enough below the scene to never appear in the render.

    The ``gaussian`` asset URL is filled in at runtime by
    :func:`fresh_synthetic_gaussian_interactive_scene`.
    """

    env_spacing: float = 2.0

    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

    gaussian = AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{SYNTHETIC_GAUSSIAN_SCENE_REL_PATH}",
        spawn=sim_utils.UsdFileCfg(usd_path=""),  # filled in at runtime
    )

    anchor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Anchor",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
    )


@contextlib.contextmanager
def fresh_synthetic_gaussian_interactive_scene(
    usd_path: str,
    sim_cfg: SimulationCfg,
    *,
    num_envs: int = 1,
) -> Iterator[SimulationContext]:
    """Yield a fresh :class:`~isaaclab.sim.SimulationContext` with the synthesised
    gaussian asset referenced under each env via :class:`SyntheticGaussianSceneCfg`.

    The InteractiveScene is held alive for the lifetime of the context — its
    callbacks register *weak* refs to the parent's bound methods; if the scene
    is dropped, the next ``dispatch_event`` raises ``ReferenceError`` from a
    dead weakref.

    Args:
        usd_path: Path to the synthesised gaussian USD asset (typically produced
            by :func:`make_synthetic_gaussian_usd`).
        sim_cfg: The simulation cfg (caller-provided, since the physics backend
            and timestep are renderer-specific).
        num_envs: Number of tiled envs to spawn.

    Yields:
        The constructed :class:`SimulationContext`.
    """
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = SyntheticGaussianSceneCfg(num_envs=num_envs)
    scene_cfg.gaussian.spawn = sim_utils.UsdFileCfg(usd_path=usd_path)
    scene = InteractiveScene(scene_cfg)  # noqa: F841 — kept alive intentionally
    try:
        yield sim
    finally:
        with contextlib.suppress(Exception):
            sim.stop()
        with contextlib.suppress(Exception):
            sim.clear_instance()


def render_synthetic_gaussian_scene(
    usd_path: str,
    *,
    sim_cfg: SimulationCfg,
    renderer_cfg: RendererCfg,
    data_types: list[str],
    num_envs: int = 1,
    height: int = 128,
    width: int = 128,
    sim_dt: float = 1.0 / 60.0,
    stabilisation_steps: int = 5,
    responsivity: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Render the synthesised gaussian asset with the aggressive wrapper PPISP.

    Builds an :class:`~isaaclab.scene.InteractiveScene` via
    :func:`fresh_synthetic_gaussian_interactive_scene`, instantiates a
    :class:`~isaaclab.sensors.camera.Camera` whose prim path is
    :data:`SYNTHETIC_GAUSSIAN_CAMERA_REGEX` (one camera per env), drives the
    sim for ``stabilisation_steps`` ticks, and returns every requested output.

    Args:
        usd_path: Path to the synthesised gaussian USD asset.
        sim_cfg: Caller-provided simulation cfg (carries the physics backend).
        renderer_cfg: Renderer cfg (typically ``IsaacRtxRendererCfg``,
            ``NewtonWarpRendererCfg``, or ``OVRTXRendererCfg``).
        data_types: List passed through to :attr:`~isaaclab.sensors.camera.CameraCfg.data_types`.
            Include ``"rgb_hdr"`` here when callers need access to the renderer's HDR AOV
            (e.g. for :func:`assert_ppisp_lifts_exposure`).
        num_envs: Number of tiled envs.
        height: Render height [pixels].
        width: Render width [pixels].
        sim_dt: Simulation timestep [s] used for ``camera.update``.
        stabilisation_steps: Sim steps to run before reading the final frame.

    Returns:
        A dict mapping every key present in ``camera.data.output`` to a
        ``[num_envs, height, width, channels]`` float32 CPU tensor (uint8 LDR
        buffers are cast to float for downstream arithmetic).
    """
    isp_cfg = make_aggressive_ppisp_cfg(responsivity=responsivity)
    with fresh_synthetic_gaussian_interactive_scene(usd_path, sim_cfg, num_envs=num_envs) as sim:
        cfg = CameraCfg(
            prim_path=SYNTHETIC_GAUSSIAN_CAMERA_REGEX,
            update_period=0.0,
            height=height,
            width=width,
            data_types=data_types,
            spawn=None,
            isp_cfg=isp_cfg,
            renderer_cfg=renderer_cfg,
        )
        camera = Camera(cfg)
        sim.reset()
        for _ in range(stabilisation_steps):
            sim.step()
        camera.update(sim_dt)
        outputs = {name: tensor.clone().detach().cpu().to(torch.float32) for name, tensor in camera.data.output.items()}
        del camera
        return outputs


def render_synthetic_gaussian_scene_with_static_ppisp_spg(
    usd_path: str,
    *,
    sim_cfg: SimulationCfg,
    renderer_cfg: RendererCfg,
    sidecar_dir: str,
    ppisp_cfg: PpispCfg,
    data_types: list[str],
    num_envs: int = 1,
    height: int = 128,
    width: int = 128,
    sim_dt: float = 1.0 / 60.0,
    stabilisation_steps: int = 5,
) -> dict[str, torch.Tensor]:
    """Render the synthesised gaussian asset through an authored static PPISP SPG graph.

    The camera uses :class:`CameraISPMode.AUTO_CAMERA`; renderer backends must
    discover the authored source RenderProduct and either copy/execute it
    natively (RTX/OVRTX) or parse it into the Warp fallback (Newton).
    """
    with fresh_synthetic_gaussian_interactive_scene(usd_path, sim_cfg, num_envs=num_envs) as sim:
        author_static_ppisp_spg(
            sim.stage,
            sidecar_dir=sidecar_dir,
            ppisp_cfg=ppisp_cfg,
            width=width,
            height=height,
        )
        return _render_synthetic_gaussian_camera(
            renderer_cfg=renderer_cfg,
            data_types=data_types,
            height=height,
            width=width,
            sim_dt=sim_dt,
            stabilisation_steps=stabilisation_steps,
            isp_cfg=CameraISPMode.AUTO_CAMERA,
            sim=sim,
        )


def render_synthetic_gaussian_scene_with_controller_ppisp_spg(
    usd_path: str,
    *,
    sim_cfg: SimulationCfg,
    renderer_cfg: RendererCfg,
    sidecar_dir: str,
    ppisp_cfg: PpispCfg,
    data_types: list[str],
    num_envs: int = 1,
    height: int = 128,
    width: int = 128,
    sim_dt: float = 1.0 / 60.0,
    stabilisation_steps: int = 5,
) -> dict[str, torch.Tensor]:
    """Render the synthesised gaussian asset through controller + PPISPAuto SPG."""
    with fresh_synthetic_gaussian_interactive_scene(usd_path, sim_cfg, num_envs=num_envs) as sim:
        author_controller_ppisp_spg(
            sim.stage,
            sidecar_dir=sidecar_dir,
            ppisp_cfg=ppisp_cfg,
            width=width,
            height=height,
        )
        return _render_synthetic_gaussian_camera(
            renderer_cfg=renderer_cfg,
            data_types=data_types,
            height=height,
            width=width,
            sim_dt=sim_dt,
            stabilisation_steps=stabilisation_steps,
            isp_cfg=CameraISPMode.AUTO_CAMERA,
            sim=sim,
        )


def _render_synthetic_gaussian_camera(
    *,
    renderer_cfg: RendererCfg,
    data_types: list[str],
    height: int,
    width: int,
    sim_dt: float,
    stabilisation_steps: int,
    isp_cfg: PpispCfg | CameraISPMode | None,
    sim: SimulationContext,
) -> dict[str, torch.Tensor]:
    cfg = CameraCfg(
        prim_path=SYNTHETIC_GAUSSIAN_CAMERA_REGEX,
        update_period=0.0,
        height=height,
        width=width,
        data_types=data_types,
        spawn=None,
        isp_cfg=isp_cfg,
        renderer_cfg=renderer_cfg,
    )
    camera = Camera(cfg)
    sim.reset()
    for _ in range(stabilisation_steps):
        sim.step()
    camera.update(sim_dt)
    outputs = {name: tensor.clone().detach().cpu().to(torch.float32) for name, tensor in camera.data.output.items()}
    del camera
    return outputs
