# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for PPISP post-processing."""

from __future__ import annotations

import warp as wp

from .cfg import PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN, PpispCfg

wp.init()

PPISP_CONTROLLER_INPUT_DOWNSAMPLING = 3
PPISP_CONTROLLER_POOL_GRID_H = 5
PPISP_CONTROLLER_POOL_GRID_W = 5
PPISP_CONTROLLER_POOL_CELL_COUNT = PPISP_CONTROLLER_POOL_GRID_H * PPISP_CONTROLLER_POOL_GRID_W
PPISP_CONTROLLER_FEATURE_LEN = 64 * PPISP_CONTROLLER_POOL_CELL_COUNT
PPISP_CONTROLLER_HIDDEN_DIM = 128
PPISP_CONTROLLER_PARAM_COUNT = 9

# This layout mirrors the generated PPISP controller CUDA source exported by
# NRE's ``ppisp_export/controller/weights.py``. Linear/conv matrices are
# flattened as row-major ``[out_channel, in_channel]`` slices.
PPISP_CONTROLLER_OFF_CONV1_W = 0
PPISP_CONTROLLER_OFF_CONV1_B = PPISP_CONTROLLER_OFF_CONV1_W + 16 * 3
PPISP_CONTROLLER_OFF_CONV2_W = PPISP_CONTROLLER_OFF_CONV1_B + 16
PPISP_CONTROLLER_OFF_CONV2_B = PPISP_CONTROLLER_OFF_CONV2_W + 32 * 16
PPISP_CONTROLLER_OFF_CONV3_W = PPISP_CONTROLLER_OFF_CONV2_B + 32
PPISP_CONTROLLER_OFF_CONV3_B = PPISP_CONTROLLER_OFF_CONV3_W + 64 * 32
PPISP_CONTROLLER_OFF_TRUNK0_W = PPISP_CONTROLLER_OFF_CONV3_B + 64
PPISP_CONTROLLER_OFF_TRUNK0_B = PPISP_CONTROLLER_OFF_TRUNK0_W + 128 * 1601
PPISP_CONTROLLER_OFF_TRUNK1_W = PPISP_CONTROLLER_OFF_TRUNK0_B + 128
PPISP_CONTROLLER_OFF_TRUNK1_B = PPISP_CONTROLLER_OFF_TRUNK1_W + 128 * 128
PPISP_CONTROLLER_OFF_TRUNK2_W = PPISP_CONTROLLER_OFF_TRUNK1_B + 128
PPISP_CONTROLLER_OFF_TRUNK2_B = PPISP_CONTROLLER_OFF_TRUNK2_W + 128 * 128
PPISP_CONTROLLER_OFF_EXP_W = PPISP_CONTROLLER_OFF_TRUNK2_B + 128
PPISP_CONTROLLER_OFF_EXP_B = PPISP_CONTROLLER_OFF_EXP_W + 128
PPISP_CONTROLLER_OFF_COL_W = PPISP_CONTROLLER_OFF_EXP_B + 1
PPISP_CONTROLLER_OFF_COL_B = PPISP_CONTROLLER_OFF_COL_W + 8 * 128
PPISP_CONTROLLER_TOTAL_WEIGHTS = PPISP_CONTROLLER_OFF_COL_B + 8
if PPISP_CONTROLLER_TOTAL_WEIGHTS != PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN:
    raise RuntimeError(
        "PPISP controller weight offsets do not match the expected exported weight count: "
        f"{PPISP_CONTROLLER_TOTAL_WEIGHTS} != {PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN}."
    )


@wp.func
def _bounded_softplus(raw: wp.float32, min_value: wp.float32):
    """Map an unconstrained parameter to a positive value with a lower bound.

    The PPISP CRF stores toe/shoulder/gamma as raw optimization parameters. This
    helper applies ``min + log(1 + exp(raw))`` so the resulting shape parameters
    stay positive and numerically away from degenerate zero values.
    """
    return min_value + wp.log(1.0 + wp.exp(raw))


@wp.func
def _sigmoid(raw: wp.float32):
    return 1.0 / (1.0 + wp.exp(0.0 - raw))


@wp.func
def _apply_vignetting(
    value: wp.float32,
    uv: wp.vec2f,
    optical_center: wp.vec2f,
    alpha1: wp.float32,
    alpha2: wp.float32,
    alpha3: wp.float32,
):
    """Apply per-channel radial vignetting in local normalized image coordinates.

    The pixel coordinate ``uv`` is centered on the current de-tiled camera image
    and normalized by ``max(width, height)``. The falloff is the clamped radial
    polynomial ``1 + a1 r^2 + a2 r^4 + a3 r^6``, where
    ``r^2 = dot(uv - optical_center, uv - optical_center)``.
    """
    delta = uv - optical_center
    radius_squared = wp.dot(delta, delta)
    radius_power = radius_squared
    falloff = wp.float32(1.0) + alpha1 * radius_power
    radius_power = radius_power * radius_squared
    falloff = falloff + alpha2 * radius_power
    radius_power = radius_power * radius_squared
    falloff = falloff + alpha3 * radius_power
    return value * wp.clamp(falloff, 0.0, 1.0)


@wp.func
def _apply_crf(
    value: wp.float32,
    toe_raw: wp.float32,
    shoulder_raw: wp.float32,
    gamma_raw: wp.float32,
    center_raw: wp.float32,
):
    """Apply one channel of the PPISP camera response function.

    The CRF is a piecewise power curve split around ``center = sigmoid(raw)``.
    Toe, shoulder, and gamma are bounded-softplus parameters. Values below the
    center use the toe exponent; values above it use the shoulder exponent; the
    result is then raised to ``gamma``. The center is clamped away from 0 and 1
    because it appears in the denominator of both curve segments.
    """
    x = wp.clamp(value, 0.0, 1.0)
    toe = _bounded_softplus(toe_raw, 0.3)
    shoulder = _bounded_softplus(shoulder_raw, 0.3)
    gamma = _bounded_softplus(gamma_raw, 0.1)
    center = wp.clamp(_sigmoid(center_raw), 1.0e-6, 1.0 - 1.0e-6)

    lerp_value = (shoulder - toe) * center + toe
    a = (shoulder * center) / lerp_value
    b = 1.0 - a

    y = wp.float32(0.0)
    if x <= center:
        y = a * wp.pow(x / center, toe)
    else:
        y = 1.0 - b * wp.pow((1.0 - x) / (1.0 - center), shoulder)

    return wp.pow(wp.max(0.0, y), gamma)


@wp.func
def _compute_homography_mul(
    rgb: wp.vec3f,
    blue_latent: wp.vec2f,
    red_latent: wp.vec2f,
    green_latent: wp.vec2f,
    neutral_latent: wp.vec2f,
):
    """Apply the PPISP color correction homography.

    The four 2D latent controls perturb the blue, red, green, and neutral
    chromaticity anchors. A projective transform is solved from those anchors
    and applied in ``r,g,intensity`` space, preserving the input pixel intensity
    after the chromaticity remap.
    """
    blue_delta = wp.vec2f(
        0.0480542 * blue_latent[0] - 0.0043631 * blue_latent[1],
        -0.0043631 * blue_latent[0] + 0.0481283 * blue_latent[1],
    )
    red_delta = wp.vec2f(
        0.0580570 * red_latent[0] - 0.0179872 * red_latent[1],
        -0.0179872 * red_latent[0] + 0.0431061 * red_latent[1],
    )
    green_delta = wp.vec2f(
        0.0433336 * green_latent[0] - 0.0180537 * green_latent[1],
        -0.0180537 * green_latent[0] + 0.0580500 * green_latent[1],
    )
    neutral_delta = wp.vec2f(
        0.0128369 * neutral_latent[0] - 0.0034654 * neutral_latent[1],
        -0.0034654 * neutral_latent[0] + 0.0128158 * neutral_latent[1],
    )

    target_blue = wp.vec3f(blue_delta[0], blue_delta[1], 1.0)
    target_red = wp.vec3f(1.0 + red_delta[0], red_delta[1], 1.0)
    target_green = wp.vec3f(green_delta[0], 1.0 + green_delta[1], 1.0)
    target_gray = wp.vec3f(1.0 / 3.0 + neutral_delta[0], 1.0 / 3.0 + neutral_delta[1], 1.0)

    row0 = wp.vec3f(target_gray[1] - target_blue[1], target_gray[1] - target_red[1], target_gray[1] - target_green[1])
    row1 = wp.vec3f(target_blue[0] - target_gray[0], target_red[0] - target_gray[0], target_green[0] - target_gray[0])
    row2 = wp.vec3f(
        -target_gray[1] * target_blue[0] + target_gray[0] * target_blue[1],
        -target_gray[1] * target_red[0] + target_gray[0] * target_red[1],
        -target_gray[1] * target_green[0] + target_gray[0] * target_green[1],
    )

    lam = wp.cross(row0, row1)
    if wp.dot(lam, lam) < 1.0e-20:
        lam = wp.cross(row0, row2)
        if wp.dot(lam, lam) < 1.0e-20:
            lam = wp.cross(row1, row2)

    col0 = -target_blue * lam[0] + target_red * lam[1]
    col1 = -target_blue * lam[0] + target_green * lam[2]
    col2 = target_blue * lam[0]

    h22 = col0[2] + col1[2] + col2[2]
    if wp.abs(h22) > 1.0e-20:
        inv_h22 = 1.0 / h22
        col0 = col0 * inv_h22
        col1 = col1 * inv_h22
        col2 = col2 * inv_h22

    intensity = rgb[0] + rgb[1] + rgb[2]
    rgi = col0 * rgb[0] + col1 * rgb[1] + col2 * intensity
    rgi = rgi * (intensity / (rgi[2] + 1.0e-5))
    return wp.vec3f(rgi[0], rgi[1], rgi[2] - rgi[0] - rgi[1])


@wp.kernel(enable_backward=False)
def _zero_ppisp_controller_features_kernel(features: wp.array2d(dtype=wp.float32)):
    camera_id, feature_id = wp.tid()
    features[camera_id, feature_id] = 0.0


@wp.kernel(enable_backward=False)
def _ppisp_controller_pool1_kernel(
    hdr_color: wp.array4d(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    pool1: wp.array4d(dtype=wp.float32),
    image_width: wp.int32,
    image_height: wp.int32,
):
    """Run Conv1x1(3->16), 3x3/stride-3 max-pool, and ReLU."""
    camera_id, dy, packed = wp.tid()
    dx = packed // 16
    out_channel = packed - dx * 16

    x0 = dx * PPISP_CONTROLLER_INPUT_DOWNSAMPLING
    y0 = dy * PPISP_CONTROLLER_INPUT_DOWNSAMPLING
    x1 = x0 + PPISP_CONTROLLER_INPUT_DOWNSAMPLING
    y1 = y0 + PPISP_CONTROLLER_INPUT_DOWNSAMPLING
    if x1 > image_width:
        x1 = image_width
    if y1 > image_height:
        y1 = image_height

    pooled = wp.float32(-3.4028234663852886e38)
    yy = y0
    while yy < y1:
        xx = x0
        while xx < x1:
            v = weights[PPISP_CONTROLLER_OFF_CONV1_B + out_channel]
            v = v + hdr_color[camera_id, yy, xx, 0] * weights[PPISP_CONTROLLER_OFF_CONV1_W + out_channel * 3]
            v = v + hdr_color[camera_id, yy, xx, 1] * weights[PPISP_CONTROLLER_OFF_CONV1_W + out_channel * 3 + 1]
            v = v + hdr_color[camera_id, yy, xx, 2] * weights[PPISP_CONTROLLER_OFF_CONV1_W + out_channel * 3 + 2]
            pooled = wp.max(pooled, v)
            xx = xx + 1
        yy = yy + 1
    pool1[camera_id, dy, dx, out_channel] = wp.max(0.0, pooled)


@wp.kernel(enable_backward=False)
def _ppisp_controller_conv2_kernel(
    pool1: wp.array4d(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    conv2: wp.array4d(dtype=wp.float32),
):
    """Run Conv1x1(16->32) and ReLU on downsampled controller features."""
    camera_id, dy, packed = wp.tid()
    dx = packed // 32
    out_channel = packed - dx * 32

    v = weights[PPISP_CONTROLLER_OFF_CONV2_B + out_channel]
    for in_channel in range(16):
        v = (
            v
            + pool1[camera_id, dy, dx, in_channel]
            * weights[PPISP_CONTROLLER_OFF_CONV2_W + out_channel * 16 + in_channel]
        )
    conv2[camera_id, dy, dx, out_channel] = wp.max(0.0, v)


@wp.kernel(enable_backward=False)
def _ppisp_controller_pool_features_kernel(
    conv2: wp.array4d(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    features: wp.array2d(dtype=wp.float32),
    ds_width: wp.int32,
    ds_height: wp.int32,
):
    """Run Conv1x1(32->64) and accumulate AdaptiveAvgPool2d((5, 5))."""
    camera_id, dy, packed = wp.tid()
    dx = packed // 64
    out_channel = packed - dx * 64

    v = weights[PPISP_CONTROLLER_OFF_CONV3_B + out_channel]
    for in_channel in range(32):
        v = (
            v
            + conv2[camera_id, dy, dx, in_channel]
            * weights[PPISP_CONTROLLER_OFF_CONV3_W + out_channel * 32 + in_channel]
        )

    gy = wp.int32(0)
    while gy < PPISP_CONTROLLER_POOL_GRID_H:
        h_start = (gy * ds_height) // PPISP_CONTROLLER_POOL_GRID_H
        h_end = ((gy + 1) * ds_height + PPISP_CONTROLLER_POOL_GRID_H - 1) // PPISP_CONTROLLER_POOL_GRID_H
        if h_end > ds_height:
            h_end = ds_height
        if dy >= h_start and dy < h_end:
            gx = wp.int32(0)
            while gx < PPISP_CONTROLLER_POOL_GRID_W:
                w_start = (gx * ds_width) // PPISP_CONTROLLER_POOL_GRID_W
                w_end = ((gx + 1) * ds_width + PPISP_CONTROLLER_POOL_GRID_W - 1) // PPISP_CONTROLLER_POOL_GRID_W
                if w_end > ds_width:
                    w_end = ds_width
                if dx >= w_start and dx < w_end:
                    count = (h_end - h_start) * (w_end - w_start)
                    cell = gy * PPISP_CONTROLLER_POOL_GRID_W + gx
                    # The exporter-side [N, C, H, W] flatten is channel-major;
                    # trunk0 was trained against that exact order.
                    feature_id = out_channel * PPISP_CONTROLLER_POOL_CELL_COUNT + cell
                    wp.atomic_add(features, camera_id, feature_id, v / wp.float32(count))
                gx = gx + 1
        gy = gy + 1


@wp.kernel(enable_backward=False)
def _ppisp_controller_trunk0_kernel(
    features: wp.array2d(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    hidden_out: wp.array2d(dtype=wp.float32),
    prior_exposure: wp.float32,
):
    """Run the controller first trunk layer: Linear(1601->128) + ReLU."""
    camera_id, out_channel = wp.tid()
    v = weights[PPISP_CONTROLLER_OFF_TRUNK0_B + out_channel]
    for in_channel in range(PPISP_CONTROLLER_FEATURE_LEN):
        v = (
            v
            + features[camera_id, in_channel] * weights[PPISP_CONTROLLER_OFF_TRUNK0_W + out_channel * 1601 + in_channel]
        )
    v = v + prior_exposure * weights[PPISP_CONTROLLER_OFF_TRUNK0_W + out_channel * 1601 + 1600]
    hidden_out[camera_id, out_channel] = wp.max(0.0, v)


@wp.kernel(enable_backward=False)
def _ppisp_controller_hidden_kernel(
    hidden_in: wp.array2d(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    hidden_out: wp.array2d(dtype=wp.float32),
    weight_offset: wp.int32,
    bias_offset: wp.int32,
):
    """Run one controller hidden layer: Linear(128->128) + ReLU."""
    camera_id, out_channel = wp.tid()
    v = weights[bias_offset + out_channel]
    for in_channel in range(PPISP_CONTROLLER_HIDDEN_DIM):
        v = (
            v
            + hidden_in[camera_id, in_channel]
            * weights[weight_offset + out_channel * PPISP_CONTROLLER_HIDDEN_DIM + in_channel]
        )
    hidden_out[camera_id, out_channel] = wp.max(0.0, v)


@wp.kernel(enable_backward=False)
def _ppisp_controller_heads_kernel(
    hidden: wp.array2d(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    controller_params: wp.array2d(dtype=wp.float32),
):
    """Run exposure and color heads, writing [exposureOffset, colorLatents...]."""
    camera_id, param_id = wp.tid()
    if param_id == 0:
        v = weights[PPISP_CONTROLLER_OFF_EXP_B]
        for in_channel in range(PPISP_CONTROLLER_HIDDEN_DIM):
            v = v + hidden[camera_id, in_channel] * weights[PPISP_CONTROLLER_OFF_EXP_W + in_channel]
        controller_params[camera_id, 0] = v
    else:
        out_channel = param_id - 1
        v = weights[PPISP_CONTROLLER_OFF_COL_B + out_channel]
        for in_channel in range(PPISP_CONTROLLER_HIDDEN_DIM):
            v = (
                v
                + hidden[camera_id, in_channel]
                * weights[PPISP_CONTROLLER_OFF_COL_W + out_channel * PPISP_CONTROLLER_HIDDEN_DIM + in_channel]
            )
        controller_params[camera_id, param_id] = v


@wp.kernel(enable_backward=False)
def _apply_ppisp_kernel(
    hdr_color: wp.array4d(dtype=wp.float32),
    out_rgba: wp.array4d(dtype=wp.uint8),
    image_width: wp.int32,
    image_height: wp.int32,
    responsivity: wp.float32,
    exposure_offset: wp.float32,
    vignetting_center_r: wp.vec2f,
    vignetting_alpha1_r: wp.float32,
    vignetting_alpha2_r: wp.float32,
    vignetting_alpha3_r: wp.float32,
    vignetting_center_g: wp.vec2f,
    vignetting_alpha1_g: wp.float32,
    vignetting_alpha2_g: wp.float32,
    vignetting_alpha3_g: wp.float32,
    vignetting_center_b: wp.vec2f,
    vignetting_alpha1_b: wp.float32,
    vignetting_alpha2_b: wp.float32,
    vignetting_alpha3_b: wp.float32,
    color_latent_blue: wp.vec2f,
    color_latent_red: wp.vec2f,
    color_latent_green: wp.vec2f,
    color_latent_neutral: wp.vec2f,
    crf_toe_r: wp.float32,
    crf_shoulder_r: wp.float32,
    crf_gamma_r: wp.float32,
    crf_center_r: wp.float32,
    crf_toe_g: wp.float32,
    crf_shoulder_g: wp.float32,
    crf_gamma_g: wp.float32,
    crf_center_g: wp.float32,
    crf_toe_b: wp.float32,
    crf_shoulder_b: wp.float32,
    crf_gamma_b: wp.float32,
    crf_center_b: wp.float32,
):
    """Apply the camera PPISP pipeline to one de-tiled HDR color tensor.

    For each pixel, the model applies HDR responsivity scaling, exposure
    scaling, per-channel vignetting, color homography correction, CRF tone
    mapping, and uint8 RGBA packing. The first tensor dimension is the
    camera/environment index, so image-space effects use local ``height_id``
    and ``width_id`` coordinates per camera.
    """
    camera_id, height_id, width_id = wp.tid()
    rgb = wp.vec3f(
        hdr_color[camera_id, height_id, width_id, 0],
        hdr_color[camera_id, height_id, width_id, 1],
        hdr_color[camera_id, height_id, width_id, 2],
    )
    max_resolution = wp.float32(image_width)
    if image_height > image_width:
        max_resolution = wp.float32(image_height)
    uv = wp.vec2f(
        (wp.float32(width_id) + 0.5 - wp.float32(image_width) * 0.5) / max_resolution,
        (wp.float32(height_id) + 0.5 - wp.float32(image_height) * 0.5) / max_resolution,
    )

    # 0. HDR responsivity (achromatic, applied pre-PPISP) — orthogonal to the
    # asset-baked radiance scaling on Gaussian SH coefficients. Default 1.0 = no-op.
    out_rgb = rgb * responsivity
    # 1. Exposure
    out_rgb = out_rgb * wp.pow(2.0, exposure_offset)
    out_rgb[0] = _apply_vignetting(
        out_rgb[0], uv, vignetting_center_r, vignetting_alpha1_r, vignetting_alpha2_r, vignetting_alpha3_r
    )
    out_rgb[1] = _apply_vignetting(
        out_rgb[1], uv, vignetting_center_g, vignetting_alpha1_g, vignetting_alpha2_g, vignetting_alpha3_g
    )
    out_rgb[2] = _apply_vignetting(
        out_rgb[2], uv, vignetting_center_b, vignetting_alpha1_b, vignetting_alpha2_b, vignetting_alpha3_b
    )
    out_rgb = _compute_homography_mul(
        out_rgb, color_latent_blue, color_latent_red, color_latent_green, color_latent_neutral
    )
    out_rgb[0] = _apply_crf(out_rgb[0], crf_toe_r, crf_shoulder_r, crf_gamma_r, crf_center_r)
    out_rgb[1] = _apply_crf(out_rgb[1], crf_toe_g, crf_shoulder_g, crf_gamma_g, crf_center_g)
    out_rgb[2] = _apply_crf(out_rgb[2], crf_toe_b, crf_shoulder_b, crf_gamma_b, crf_center_b)

    out_rgba[camera_id, height_id, width_id, 0] = wp.uint8(wp.clamp(out_rgb[0], 0.0, 1.0) * 255.0)
    out_rgba[camera_id, height_id, width_id, 1] = wp.uint8(wp.clamp(out_rgb[1], 0.0, 1.0) * 255.0)
    out_rgba[camera_id, height_id, width_id, 2] = wp.uint8(wp.clamp(out_rgb[2], 0.0, 1.0) * 255.0)
    out_rgba[camera_id, height_id, width_id, 3] = wp.uint8(255)


@wp.kernel(enable_backward=False)
def _apply_ppisp_controller_kernel(
    hdr_color: wp.array4d(dtype=wp.float32),
    out_rgba: wp.array4d(dtype=wp.uint8),
    controller_params: wp.array2d(dtype=wp.float32),
    image_width: wp.int32,
    image_height: wp.int32,
    responsivity: wp.float32,
    vignetting_center_r: wp.vec2f,
    vignetting_alpha1_r: wp.float32,
    vignetting_alpha2_r: wp.float32,
    vignetting_alpha3_r: wp.float32,
    vignetting_center_g: wp.vec2f,
    vignetting_alpha1_g: wp.float32,
    vignetting_alpha2_g: wp.float32,
    vignetting_alpha3_g: wp.float32,
    vignetting_center_b: wp.vec2f,
    vignetting_alpha1_b: wp.float32,
    vignetting_alpha2_b: wp.float32,
    vignetting_alpha3_b: wp.float32,
    crf_toe_r: wp.float32,
    crf_shoulder_r: wp.float32,
    crf_gamma_r: wp.float32,
    crf_center_r: wp.float32,
    crf_toe_g: wp.float32,
    crf_shoulder_g: wp.float32,
    crf_gamma_g: wp.float32,
    crf_center_g: wp.float32,
    crf_toe_b: wp.float32,
    crf_shoulder_b: wp.float32,
    crf_gamma_b: wp.float32,
    crf_center_b: wp.float32,
):
    """Apply PPISP using per-camera exposure/color parameters predicted by the controller."""
    camera_id, height_id, width_id = wp.tid()
    rgb = wp.vec3f(
        hdr_color[camera_id, height_id, width_id, 0],
        hdr_color[camera_id, height_id, width_id, 1],
        hdr_color[camera_id, height_id, width_id, 2],
    )
    max_resolution = wp.float32(image_width)
    if image_height > image_width:
        max_resolution = wp.float32(image_height)
    uv = wp.vec2f(
        (wp.float32(width_id) + 0.5 - wp.float32(image_width) * 0.5) / max_resolution,
        (wp.float32(height_id) + 0.5 - wp.float32(image_height) * 0.5) / max_resolution,
    )

    # Exported controller params are:
    # [exposure, blue.x, blue.y, red.x, red.y, green.x, green.y, neutral.x, neutral.y].
    color_latent_blue = wp.vec2f(controller_params[camera_id, 1], controller_params[camera_id, 2])
    color_latent_red = wp.vec2f(controller_params[camera_id, 3], controller_params[camera_id, 4])
    color_latent_green = wp.vec2f(controller_params[camera_id, 5], controller_params[camera_id, 6])
    color_latent_neutral = wp.vec2f(controller_params[camera_id, 7], controller_params[camera_id, 8])

    out_rgb = rgb * responsivity
    out_rgb = out_rgb * wp.pow(2.0, controller_params[camera_id, 0])
    out_rgb[0] = _apply_vignetting(
        out_rgb[0], uv, vignetting_center_r, vignetting_alpha1_r, vignetting_alpha2_r, vignetting_alpha3_r
    )
    out_rgb[1] = _apply_vignetting(
        out_rgb[1], uv, vignetting_center_g, vignetting_alpha1_g, vignetting_alpha2_g, vignetting_alpha3_g
    )
    out_rgb[2] = _apply_vignetting(
        out_rgb[2], uv, vignetting_center_b, vignetting_alpha1_b, vignetting_alpha2_b, vignetting_alpha3_b
    )
    out_rgb = _compute_homography_mul(
        out_rgb, color_latent_blue, color_latent_red, color_latent_green, color_latent_neutral
    )
    out_rgb[0] = _apply_crf(out_rgb[0], crf_toe_r, crf_shoulder_r, crf_gamma_r, crf_center_r)
    out_rgb[1] = _apply_crf(out_rgb[1], crf_toe_g, crf_shoulder_g, crf_gamma_g, crf_center_g)
    out_rgb[2] = _apply_crf(out_rgb[2], crf_toe_b, crf_shoulder_b, crf_gamma_b, crf_center_b)

    out_rgba[camera_id, height_id, width_id, 0] = wp.uint8(wp.clamp(out_rgb[0], 0.0, 1.0) * 255.0)
    out_rgba[camera_id, height_id, width_id, 1] = wp.uint8(wp.clamp(out_rgb[1], 0.0, 1.0) * 255.0)
    out_rgba[camera_id, height_id, width_id, 2] = wp.uint8(wp.clamp(out_rgb[2], 0.0, 1.0) * 255.0)
    out_rgba[camera_id, height_id, width_id, 3] = wp.uint8(255)


def apply_ppisp_to_rgba(hdr_color: wp.array, out_rgba: wp.array, cfg: PpispCfg) -> None:
    """Apply PPISP to ``hdr_color`` and write LDR RGBA into ``out_rgba``.

    Args:
        hdr_color: HDR scene-linear input. ``wp.array4d(dtype=wp.float32)`` of
            shape ``(N, H, W, 3)``.
        out_rgba: LDR RGBA output. ``wp.array4d(dtype=wp.uint8)`` of shape
            ``(N, H, W, 4)``. Written in place.
        cfg: PPISP configuration.
    """
    if hdr_color.dtype is not wp.float32:
        raise ValueError(f"Camera PPISP HDR input must be wp.float32, got {hdr_color.dtype}.")
    if out_rgba.dtype is not wp.uint8:
        raise ValueError(f"Camera PPISP RGBA output must be wp.uint8, got {out_rgba.dtype}.")

    inputs = cfg.inputs
    wp.launch(
        _apply_ppisp_kernel,
        dim=out_rgba.shape[:3],
        inputs=[
            hdr_color,
            out_rgba,
            int(out_rgba.shape[2]),
            int(out_rgba.shape[1]),
            float(inputs.get("responsivity", 1.0)),
            float(inputs["exposureOffset"]),
            wp.vec2f(*inputs["vignettingCenterR"]),
            float(inputs["vignettingAlpha1R"]),
            float(inputs["vignettingAlpha2R"]),
            float(inputs["vignettingAlpha3R"]),
            wp.vec2f(*inputs["vignettingCenterG"]),
            float(inputs["vignettingAlpha1G"]),
            float(inputs["vignettingAlpha2G"]),
            float(inputs["vignettingAlpha3G"]),
            wp.vec2f(*inputs["vignettingCenterB"]),
            float(inputs["vignettingAlpha1B"]),
            float(inputs["vignettingAlpha2B"]),
            float(inputs["vignettingAlpha3B"]),
            wp.vec2f(*inputs["colorLatentBlue"]),
            wp.vec2f(*inputs["colorLatentRed"]),
            wp.vec2f(*inputs["colorLatentGreen"]),
            wp.vec2f(*inputs["colorLatentNeutral"]),
            float(inputs["crfToeR"]),
            float(inputs["crfShoulderR"]),
            float(inputs["crfGammaR"]),
            float(inputs["crfCenterR"]),
            float(inputs["crfToeG"]),
            float(inputs["crfShoulderG"]),
            float(inputs["crfGammaG"]),
            float(inputs["crfCenterG"]),
            float(inputs["crfToeB"]),
            float(inputs["crfShoulderB"]),
            float(inputs["crfGammaB"]),
            float(inputs["crfCenterB"]),
        ],
        device=str(out_rgba.device),
    )


def compute_ppisp_controller_params(
    hdr_color: wp.array,
    controller_weights: wp.array,
    pool1: wp.array,
    conv2: wp.array,
    features: wp.array,
    hidden_a: wp.array,
    hidden_b: wp.array,
    controller_params: wp.array,
    prior_exposure: float,
) -> None:
    """Run the PPISP controller network using only Warp buffers and kernels.

    Args:
        hdr_color: HDR scene-linear input, shape ``(N, H, W, 3)``.
        controller_weights: Flattened embedded controller weights.
        pool1: Scratch buffer, shape ``(N, H//3, W//3, 16)`` with minimum
            downsampled size one.
        conv2: Scratch buffer, shape ``(N, H//3, W//3, 32)`` with minimum
            downsampled size one.
        features: Scratch buffer, shape ``(N, 1600)``.
        hidden_a: Scratch buffer, shape ``(N, 128)``.
        hidden_b: Scratch buffer, shape ``(N, 128)``.
        controller_params: Output buffer, shape ``(N, 9)``.
        prior_exposure: Scalar prior exposure authored on the controller shader.
    """
    if hdr_color.dtype is not wp.float32:
        raise ValueError(f"Camera PPISP controller HDR input must be wp.float32, got {hdr_color.dtype}.")
    if controller_weights.dtype is not wp.float32:
        raise ValueError(f"Camera PPISP controller weights must be wp.float32, got {controller_weights.dtype}.")
    if controller_weights.shape[0] != PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN:
        raise ValueError(
            "Camera PPISP controller weights must have shape "
            f"({PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN},), got {controller_weights.shape}."
        )
    if features.shape != (hdr_color.shape[0], PPISP_CONTROLLER_FEATURE_LEN):
        raise ValueError(
            "Camera PPISP controller features must have shape "
            f"({hdr_color.shape[0]}, {PPISP_CONTROLLER_FEATURE_LEN}), got {features.shape}."
        )
    if hidden_a.shape != (hdr_color.shape[0], PPISP_CONTROLLER_HIDDEN_DIM):
        raise ValueError(
            "Camera PPISP controller hidden buffer must have shape "
            f"({hdr_color.shape[0]}, {PPISP_CONTROLLER_HIDDEN_DIM}), got {hidden_a.shape}."
        )
    if hidden_b.shape != hidden_a.shape:
        raise ValueError(f"Camera PPISP controller hidden buffers must match, got {hidden_b.shape}.")
    if controller_params.shape != (hdr_color.shape[0], PPISP_CONTROLLER_PARAM_COUNT):
        raise ValueError(
            "Camera PPISP controller params must have shape "
            f"({hdr_color.shape[0]}, {PPISP_CONTROLLER_PARAM_COUNT}), got {controller_params.shape}."
        )

    image_height = int(hdr_color.shape[1])
    image_width = int(hdr_color.shape[2])
    ds_width = max(1, image_width // PPISP_CONTROLLER_INPUT_DOWNSAMPLING)
    ds_height = max(1, image_height // PPISP_CONTROLLER_INPUT_DOWNSAMPLING)
    expected_pool1_shape = (hdr_color.shape[0], ds_height, ds_width, 16)
    expected_conv2_shape = (hdr_color.shape[0], ds_height, ds_width, 32)
    if pool1.shape != expected_pool1_shape:
        raise ValueError(
            f"Camera PPISP controller pool1 buffer must have shape {expected_pool1_shape}, got {pool1.shape}."
        )
    if conv2.shape != expected_conv2_shape:
        raise ValueError(
            f"Camera PPISP controller conv2 buffer must have shape {expected_conv2_shape}, got {conv2.shape}."
        )

    device = str(hdr_color.device)
    wp.launch(
        _zero_ppisp_controller_features_kernel,
        dim=features.shape,
        inputs=[features],
        device=device,
    )
    wp.launch(
        _ppisp_controller_pool1_kernel,
        dim=(int(hdr_color.shape[0]), ds_height, ds_width * 16),
        inputs=[hdr_color, controller_weights, pool1, image_width, image_height],
        device=device,
    )
    wp.launch(
        _ppisp_controller_conv2_kernel,
        dim=(int(hdr_color.shape[0]), ds_height, ds_width * 32),
        inputs=[pool1, controller_weights, conv2],
        device=device,
    )
    wp.launch(
        _ppisp_controller_pool_features_kernel,
        dim=(int(hdr_color.shape[0]), ds_height, ds_width * 64),
        inputs=[conv2, controller_weights, features, ds_width, ds_height],
        device=device,
    )
    wp.launch(
        _ppisp_controller_trunk0_kernel,
        dim=(int(hdr_color.shape[0]), PPISP_CONTROLLER_HIDDEN_DIM),
        inputs=[features, controller_weights, hidden_a, float(prior_exposure)],
        device=device,
    )
    wp.launch(
        _ppisp_controller_hidden_kernel,
        dim=(int(hdr_color.shape[0]), PPISP_CONTROLLER_HIDDEN_DIM),
        inputs=[
            hidden_a,
            controller_weights,
            hidden_b,
            PPISP_CONTROLLER_OFF_TRUNK1_W,
            PPISP_CONTROLLER_OFF_TRUNK1_B,
        ],
        device=device,
    )
    wp.launch(
        _ppisp_controller_hidden_kernel,
        dim=(int(hdr_color.shape[0]), PPISP_CONTROLLER_HIDDEN_DIM),
        inputs=[
            hidden_b,
            controller_weights,
            hidden_a,
            PPISP_CONTROLLER_OFF_TRUNK2_W,
            PPISP_CONTROLLER_OFF_TRUNK2_B,
        ],
        device=device,
    )
    wp.launch(
        _ppisp_controller_heads_kernel,
        dim=(int(hdr_color.shape[0]), PPISP_CONTROLLER_PARAM_COUNT),
        inputs=[hidden_a, controller_weights, controller_params],
        device=device,
    )


def apply_ppisp_to_rgba_with_controller_params(
    hdr_color: wp.array, out_rgba: wp.array, cfg: PpispCfg, controller_params: wp.array
) -> None:
    """Apply PPISP with per-camera controller parameters.

    Args:
        hdr_color: HDR scene-linear input, shape ``(N, H, W, 3)``.
        out_rgba: LDR RGBA output, shape ``(N, H, W, 4)``.
        cfg: PPISP configuration. Static inputs provide responsivity,
            vignetting and CRF.
        controller_params: ``wp.array2d(float32)`` of shape ``(N, 9)`` holding
            ``[exposureOffset, colorLatentBlue.xy, colorLatentRed.xy,
            colorLatentGreen.xy, colorLatentNeutral.xy]``.
    """
    if hdr_color.dtype is not wp.float32:
        raise ValueError(f"Camera PPISP HDR input must be wp.float32, got {hdr_color.dtype}.")
    if out_rgba.dtype is not wp.uint8:
        raise ValueError(f"Camera PPISP RGBA output must be wp.uint8, got {out_rgba.dtype}.")
    if controller_params.dtype is not wp.float32:
        raise ValueError(f"Camera PPISP controller params must be wp.float32, got {controller_params.dtype}.")
    if controller_params.shape[0] != out_rgba.shape[0] or controller_params.shape[1] != 9:
        raise ValueError(
            f"Camera PPISP controller params must have shape ({out_rgba.shape[0]}, 9), got {controller_params.shape}."
        )

    inputs = cfg.inputs
    wp.launch(
        _apply_ppisp_controller_kernel,
        dim=out_rgba.shape[:3],
        inputs=[
            hdr_color,
            out_rgba,
            controller_params,
            int(out_rgba.shape[2]),
            int(out_rgba.shape[1]),
            float(inputs.get("responsivity", 1.0)),
            wp.vec2f(*inputs["vignettingCenterR"]),
            float(inputs["vignettingAlpha1R"]),
            float(inputs["vignettingAlpha2R"]),
            float(inputs["vignettingAlpha3R"]),
            wp.vec2f(*inputs["vignettingCenterG"]),
            float(inputs["vignettingAlpha1G"]),
            float(inputs["vignettingAlpha2G"]),
            float(inputs["vignettingAlpha3G"]),
            wp.vec2f(*inputs["vignettingCenterB"]),
            float(inputs["vignettingAlpha1B"]),
            float(inputs["vignettingAlpha2B"]),
            float(inputs["vignettingAlpha3B"]),
            float(inputs["crfToeR"]),
            float(inputs["crfShoulderR"]),
            float(inputs["crfGammaR"]),
            float(inputs["crfCenterR"]),
            float(inputs["crfToeG"]),
            float(inputs["crfShoulderG"]),
            float(inputs["crfGammaG"]),
            float(inputs["crfCenterG"]),
            float(inputs["crfToeB"]),
            float(inputs["crfShoulderB"]),
            float(inputs["crfGammaB"]),
            float(inputs["crfCenterB"]),
        ],
        device=str(out_rgba.device),
    )
