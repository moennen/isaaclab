#!/usr/bin/env python3
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert a 3D Gaussian Splatting (3DGS) PLY file to a USD asset with Simplicits physics
for use with Isaac Lab's gaussian_lifting task.

Usage (with Isaac Lab Newton env):
  micromamba run -n isaaclab-newton python scripts/gaussian_ply_to_usd.py input.ply output.usdc --scale 0.1 --object-name my_object

The script:
  1. Loads the 3DGS PLY (positions, rotations, scales, opacities, SH coefficients).
  2. Applies rescaling (positions and scales multiplied by --scale).
  3. Builds a Simplicits representation (interior sampling, training, cubature + render weights).
  4. Exports USD with ParticleField3DGaussianSplat + GeomSubset + KaolinSimplicitsSetup
     so Isaac Lab can load it via spawn_gaussian_from_usd().

If "Sampling points in Gaussian volume" exits with no traceback:
  - Likely OOM or segfault in the densifier (C++/CUDA). To diagnose:
    1. OOM: check ``dmesg | tail -30`` for "Out of memory" / "Killed process".
    2. Segfault: run with ``python -X faulthandler scripts/gaussian_ply_to_usd.py ...``
       to get a stack trace when the process crashes.
    3. GPU errors: run with ``CUDA_LAUNCH_BLOCKING=1 python scripts/...`` so GPU
       errors surface as Python exceptions.
    4. Reduce memory: use ``--octree-level 6`` (default 7; tutorials use 8 but it OOMs on large clouds)
       and/or ``--max-gaussians 15000`` and/or ``--use-gaussian-centers`` to skip volume sampling.

Tutorial-style values (examples/tutorial/physics/simplicits_inria_splatting*.ipynb, simulatable_3dgrut.ipynb):
  num_handles=40, num_qp=2048, model_layers=10, training_num_steps=10000 (or 25000 for 40 handles).
  Octree default in notebooks is 8; script defaults to 7 to avoid OOM (use 8 if you have enough RAM).
"""

import argparse
import logging
import os
import sys
import traceback

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _trace(msg: str) -> None:
    """Print a checkpoint and flush so we see it even if the process is killed right after."""
    print(f"[TRACE] {msg}", flush=True)
    log.info("%s", msg)


def _load_ply_3dgs(path: str, device: str = "cuda"):
    """Load a 3D Gaussian Splatting PLY file.

    Expects standard 3DGS PLY format: x, y, z, nx, ny, nz,
    f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2,
    rot_0, rot_1, rot_2, rot_3, then f_rest_* (45 coeffs for SH degree 3).

    Returns:
        dict with keys: positions (N,3), rotations (N,4) quat wxyz,
        scales (N,3), opacities (N,), sh_coeff (N,16,3).
    """
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header.append(line)
            if line == "end_header":
                break
        # Parse header for vertex count and properties
        vertex_count = 0
        props = []
        for line in header:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                # property <type> <name>
                props.append((parts[2], parts[1]))

        dtype_map = {"float": np.float32, "double": np.float64}
        dtype = np.dtype([(name, dtype_map.get(typ, np.float32)) for name, typ in props])
        data = np.fromfile(f, dtype=dtype, count=vertex_count)

    # Required 3DGS properties
    x = np.asarray(data["x"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)
    z = np.asarray(data["z"], dtype=np.float32)
    positions = np.stack([x, y, z], axis=1)

    scale_0 = np.asarray(data["scale_0"], dtype=np.float32)
    scale_1 = np.asarray(data["scale_1"], dtype=np.float32)
    scale_2 = np.asarray(data["scale_2"], dtype=np.float32)
    scales = np.stack([scale_0, scale_1, scale_2], axis=1)

    rot_0 = np.asarray(data["rot_0"], dtype=np.float32)
    rot_1 = np.asarray(data["rot_1"], dtype=np.float32)
    rot_2 = np.asarray(data["rot_2"], dtype=np.float32)
    rot_3 = np.asarray(data["rot_3"], dtype=np.float32)
    # 3DGS stores quaternion as (w,x,y,z)
    rotations = np.stack([rot_0, rot_1, rot_2, rot_3], axis=1)

    opacity = np.asarray(data["opacity"], dtype=np.float32)

    f_dc_0 = np.asarray(data["f_dc_0"], dtype=np.float32)
    f_dc_1 = np.asarray(data["f_dc_1"], dtype=np.float32)
    f_dc_2 = np.asarray(data["f_dc_2"], dtype=np.float32)
    sh_dc = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=1)  # (N, 3)
    sh_rest = []
    for i in range(45):
        key = f"f_rest_{i}"
        if key in data.dtype.names:
            sh_rest.append(np.asarray(data[key], dtype=np.float32))
        else:
            sh_rest.append(np.zeros(vertex_count, dtype=np.float32))
    sh_rest = np.stack(sh_rest, axis=1)  # (N, 45)
    # SH coeff layout for Kaolin USD: (N, 16, 3) — 16 bands, 3 channels
    sh_coeff = np.zeros((vertex_count, 16, 3), dtype=np.float32)
    sh_coeff[:, 0, :] = sh_dc
    for i in range(min(15, sh_rest.shape[1] // 3)):
        for c in range(3):
            sh_coeff[:, 1 + i, c] = sh_rest[:, i * 3 + c]

    out = {
        "positions": torch.from_numpy(positions).float().to(device),
        "rotations": torch.from_numpy(rotations).float().to(device),
        "scales": torch.from_numpy(scales).float().to(device),
        "opacities": torch.from_numpy(opacity).float().to(device),
        "sh_coeff": torch.from_numpy(sh_coeff).float().to(device),
    }
    log.info("Loaded %d Gaussians from %s", positions.shape[0], path)
    return out


def apply_activations_3dgs(gaussian_data: dict) -> None:
    """Apply 3DGS activations in-place so scales and opacities are post-activation.

    PLY stores scale as log(scale) and opacity as raw; densifier and simplicits expect
    post-activation (exp(scale), sigmoid(opacity)), matching GaussianModel.get_scaling / get_opacity.
    """
    gaussian_data["scales"] = torch.exp(gaussian_data["scales"])
    gaussian_data["opacities"] = torch.sigmoid(gaussian_data["opacities"])


def apply_scaling(gaussian_data: dict, scale: float) -> None:
    """Rescale positions and scales in-place. Rotations (quaternions) and SH/opacity are unchanged.

    Must be called after apply_activations_3dgs so that scales are post-activation (exp);
    then positions and scales are multiplied by scale for world-space units.
    """
    if scale != 1.0:
        log.info("Applying scale=%.4f to positions and scales only (rotations unchanged).", scale)
    gaussian_data["positions"] = gaussian_data["positions"] * scale
    gaussian_data["scales"] = gaussian_data["scales"] * scale


def subsample_gaussians(gaussian_data: dict, max_gaussians: int, seed: int = 0) -> None:
    """Keep at most max_gaussians by random subsampling. Modifies gaussian_data in-place."""
    n = gaussian_data["positions"].shape[0]
    if n <= max_gaussians:
        return
    gen = torch.Generator(device=gaussian_data["positions"].device).manual_seed(seed)
    idx = torch.randperm(n, generator=gen, device=gaussian_data["positions"].device)[:max_gaussians]
    for key in gaussian_data:
        gaussian_data[key] = gaussian_data[key][idx]
    log.info("Subsampled to %d Gaussians (from %d)", max_gaussians, n)


def build_simplicits_and_export_usd(
    gaussian_data: dict,
    output_path: str,
    object_name: str,
    gaussian_prim_path: str = "/World/Gaussians/gaussians_0",
    num_handles: int = 40,
    num_qp: int = 2048,
    num_training_steps: int = 5000,
    youngs_modulus: float = 1e6,
    poisson_ratio: float = 0.45,
    density: float = 100.0,
    friction_coeff: float = 0.5,
    up_axis: str = "Y",
    use_gaussian_centers: bool = False,
    octree_level: int = 7,
    opacity_threshold: float = 0.0,
    model_layers: int = 10,
    max_physics_points: int = 20_000,
) -> None:
    """Build Simplicits from Gaussian data and export to USD for Isaac Lab."""

    import kaolin as kal
    from kaolin.io.usd import (
        add_binded_simplicits_setup,
        add_gaussiancloud,
        add_subset,
        create_stage,
    )

    device = gaussian_data["positions"].device
    positions = gaussian_data["positions"]
    rotations = gaussian_data["rotations"]
    scales = gaussian_data["scales"]
    opacities = gaussian_data["opacities"]
    sh_coeff = gaussian_data["sh_coeff"]

    _trace("build_simplicits: start")
    num_gaussians = positions.shape[0]
    # Approximate volume from bbox (used for Simplicits)
    bb_min = positions.min(dim=0).values
    bb_max = positions.max(dim=0).values
    bbox_extent = (bb_max - bb_min).tolist()
    appx_vol = (bb_max - bb_min).prod().item()
    appx_vol = max(appx_vol, 1e-6)
    log.info(
        "Problem size: num_gaussians=%d, bbox_extent=[%.4f, %.4f, %.4f], appx_vol=%.6f",
        num_gaussians, bbox_extent[0], bbox_extent[1], bbox_extent[2], appx_vol,
    )

    if use_gaussian_centers:
        # Skip volume sampling: use Gaussian centers as physics points (faster, no densifier OOM)
        _trace("build_simplicits: using Gaussian centers (skip volume sampling)")
        pts_volume = positions
    else:
        # Sample points inside the Gaussian volume for physics
        _trace(f"build_simplicits: sample_points_in_volume START (octree_level={octree_level}, n_pts={positions.shape[0]})")
        pts_volume = kal.ops.gaussian.sample_points_in_volume(
            xyz=positions,
            scale=scales,
            rotation=rotations,
            opacity=opacities,
            clip_samples_to_input_bbox=False,
            octree_level=octree_level,
            opacity_threshold=opacity_threshold
        )
        _trace(f"build_simplicits: sample_points_in_volume DONE -> {pts_volume.shape[0]} volume points")
        if pts_volume.shape[0] == 0:
            log.warning(
                "Volume sampling returned 0 points (e.g. all voxels culled by opacity_threshold). "
                "Falling back to Gaussian centers for physics."
            )
            pts_volume = positions
        elif pts_volume.shape[0] < num_qp:
            log.warning(
                "Volume sampling returned %d points (fewer than num_qp=%d). "
                "Falling back to Gaussian centers for physics so cubature has enough points.",
                pts_volume.shape[0], num_qp,
            )
            pts_volume = positions

    num_volume_pts = pts_volume.shape[0]
    log.info(
        "Physics setup: num_volume_pts=%d, num_handles=%d, num_qp=%d, model_layers=%d, training_steps=%d, youngs_modulus=%.2g, density=%.1f",
        num_volume_pts, num_handles, num_qp, model_layers, num_training_steps, youngs_modulus, density,
    )

    # Train Simplicits (skin weights)
    _trace(f"build_simplicits: SimplicitsObject.create_trained START (num_handles={num_handles}, steps={num_training_steps})")
    sim_obj = kal.physics.simplicits.SimplicitsObject.create_trained(
        pts_volume,
        youngs_modulus,
        poisson_ratio,
        density,
        appx_vol,
        num_handles=num_handles,
        num_samples=2048,
        model_layers=model_layers,
        training_num_steps=num_training_steps,
        training_log_every=max(1, num_training_steps // 5),
    )
    _trace("build_simplicits: create_trained DONE")

    # Bake for simulation (cubature points) and rendering (Gaussian weights)
    _trace(f"build_simplicits: bake_for_simulation START (num_qp={num_qp})")
    baked_obj, qp_indices = sim_obj.bake_for_simulation(num_qp=num_qp)
    _trace("build_simplicits: bake_for_simulation DONE")
    _trace("build_simplicits: bake_for_rendering START")
    render_obj = sim_obj.bake_for_rendering(points=positions)
    _trace("build_simplicits: bake_for_rendering DONE")

    # Ensure dFdz dense is computed for USD export
    _trace("build_simplicits: computing dFdz_dense")
    dfdz_dense = baked_obj.dFdz_dense
    _trace("build_simplicits: dFdz_dense DONE")

    # Create USD stage and export Gaussian cloud
    _trace("build_simplicits: create_stage + add_gaussiancloud START")
    stage = create_stage(output_path, up_axis=up_axis)
    try:
        add_gaussiancloud(
            stage,
            gaussian_prim_path,
            positions=positions,
            orientations=rotations,
            scales=scales,
            opacities=opacities,
            sh_coeff=sh_coeff,
            up_axis=up_axis,
            overwrite=True,
        )
        _trace("build_simplicits: add_gaussiancloud DONE")

        # Subset: all Gaussians belong to this object (single object in file)
        num_gaussians = positions.shape[0]
        indices = torch.arange(num_gaussians, dtype=torch.long, device="cpu")
        _trace("build_simplicits: add_subset START")
        subset_prim = add_subset(
            stage,
            gaussian_prim_path,
            name=object_name,
            indices=indices,
            override=True,
        )
        _trace("build_simplicits: add_subset DONE")
        subset_path = f"{gaussian_prim_path}/{object_name}"

        # Bind Simplicits setup to the subset (physics + render weights)
        setup_path = f"/World/SimplicitsSetup/{object_name}"
        _trace("build_simplicits: add_binded_simplicits_setup START")
        add_binded_simplicits_setup(
            stage,
            subset_path,
            setup_path,
            pts=baked_obj.pts,
            yms=baked_obj.yms,
            prs=baked_obj.prs,
            rhos=baked_obj.rhos,
            appx_vol=baked_obj.appx_vol,
            lbs_weights=baked_obj.skinning_weights,
            elem_weights=render_obj.skinning_weights,
            dfdz=dfdz_dense,
            friction_coeff=friction_coeff,
            override=True,
        )
        _trace("build_simplicits: add_binded_simplicits_setup DONE")
        stage.Save()
        _trace("build_simplicits: stage.Save DONE")
    finally:
        del stage

    _trace(f"Saved USD to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3DGS PLY to USD with Simplicits for Isaac Lab gaussian_lifting."
    )
    parser.add_argument("ply_path", type=str, help="Path to 3D Gaussian Splatting .ply file")
    parser.add_argument("output_usd", type=str, help="Output USD path (.usdc or .usda)")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Rescaling factor for positions and scales (default: 1.0)",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default="object",
        help="Name of the object (GeomSubset + SimplicitsSetup); use in Isaac Lab spawn cfg (default: object)",
    )
    parser.add_argument(
        "--gaussian-prim-path",
        type=str,
        default="/World/Gaussians/gaussians_0",
        help="USD path for ParticleField3DGaussianSplat (default: /World/Gaussians/gaussians_0)",
    )
    parser.add_argument("--num-handles", type=int, default=40, help="Simplicits num handles (default: 40, tutorials)")
    parser.add_argument("--num-qp", type=int, default=2048, help="Cubature points (default: 2048, tutorials)")
    parser.add_argument(
        "--training-steps",
        type=int,
        default=10000,
        help="Simplicits training steps (default: 10000; use 25000 for 40 handles, simulatable_3dgrut)",
    )
    parser.add_argument(
        "--model-layers",
        type=int,
        default=10,
        help="Simplicits MLP layers (default: 10, tutorials)",
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=None,
        metavar="N",
        help="Subsample to at most N Gaussians (reduces memory and runtime for large PLYs; default: use all)",
    )
    parser.add_argument(
        "--use-gaussian-centers",
        action="store_true",
        help="Use Gaussian centers as physics points instead of volume sampling (avoids densifier; use if volume sampling crashes or OOMs)",
    )
    parser.add_argument(
        "--octree-level",
        type=int,
        default=7,
        choices=(6, 7, 8, 9, 10),
        metavar="L",
        help="Densifier octree level 6–10; default 7 (tutorials use 8; 8 can OOM on large clouds)",
    )
    parser.add_argument(
        "--opacity-threshold",
        type=float,
        default=0.0,
        metavar="T",
        help="Densifier opacity culling; 0 = keep all voxels (default: 0.0); 0.35 can cull all at low octree_level",
    )
    parser.add_argument(
        "--youngs-modulus",
        type=float,
        default=1e6,
        metavar="E",
        help="Young's modulus [Pa] for Simplicits; higher = stiffer (default: 1e6; tutorials: soft 21e3, medium 1e6, stiff 1e7; doll 1e5)",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=100.0,
        metavar="RHO",
        help="Density [kg/m³] for Simplicits (default: 100; tutorials often 100–500)",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    if not os.path.isfile(args.ply_path):
        log.error("PLY file not found: %s", args.ply_path)
        sys.exit(1)

    try:
        log.info("Step 1/4: Loading PLY...")
        gaussian_data = _load_ply_3dgs(args.ply_path, device=args.device)
        if args.max_gaussians is not None:
            subsample_gaussians(gaussian_data, args.max_gaussians)
        apply_activations_3dgs(gaussian_data)
        apply_scaling(gaussian_data, args.scale)

        log.info("Step 2/4: Building Simplicits and exporting USD...")
        build_simplicits_and_export_usd(
            gaussian_data,
            args.output_usd,
            object_name=args.object_name,
            gaussian_prim_path=args.gaussian_prim_path,
            num_handles=args.num_handles,
            num_qp=args.num_qp,
            num_training_steps=args.training_steps,
            model_layers=args.model_layers,
            use_gaussian_centers=args.use_gaussian_centers,
            octree_level=args.octree_level,
            opacity_threshold=args.opacity_threshold,
            youngs_modulus=args.youngs_modulus,
            density=args.density,
        )
        log.info("Done. Output: %s", args.output_usd)
    except Exception as e:
        log.exception("Failed: %s", e)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
