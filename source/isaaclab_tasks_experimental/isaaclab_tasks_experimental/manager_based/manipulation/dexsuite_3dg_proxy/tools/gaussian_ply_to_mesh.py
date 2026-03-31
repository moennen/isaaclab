#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Convert a 3D Gaussian Splatting (3DGS) PLY file to a surface mesh and tetrahedral volume mesh.

Pipeline:
  1. Load the 3DGS PLY (positions, rotations, scales, opacities).
  2. Sample a dense surface point cloud (default: Gaussian mixture sampling).
  3. Voxelize the point cloud into a 3D density grid and smooth it.
  4. Interpolate the density field onto a FlexiCubes voxel grid and extract a surface mesh (.obj).
  5. Tetrahedralize the surface mesh with fTetWild via wildmeshing (.msh, Gmsh format).

Point cloud sampling options (mutually exclusive, default = Gaussian mixture):
  (default)               Sample points from the Gaussian mixture: x = mu + R @ (s * z), z~N(0,I).
  --use-kaolin-sampler    Use kaolin's sample_points_in_volume densifier instead.
  --use-gaussian-centers  Use opacity-filtered Gaussian centers only (fast, lower quality).

The .msh output is in Gmsh format and can be loaded directly by Newton's XPBD solver via meshio:
    import meshio
    mesh = meshio.read("object.msh")
    nodes = mesh.points                          # (V, 3)
    tets  = mesh.cells_dict["tetra"]             # (T, 4)

Requirements:
    pip install scikit-image scipy meshio wildmeshing

Usage:
    python gaussian_ply_to_mesh.py input.ply output_dir/
    python gaussian_ply_to_mesh.py input.ply output_dir/ --scale 0.1 --num-samples 300000
    python gaussian_ply_to_mesh.py input.ply output_dir/ --use-kaolin-sampler --octree-level 7
    python gaussian_ply_to_mesh.py input.ply output_dir/ --use-gaussian-centers --no-tet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _trace(msg: str) -> None:
    print(f"[TRACE] {msg}", flush=True)


# ============================================================================
# PLY LOADING
# ============================================================================

def load_ply_3dgs(path: str, device: str = "cuda") -> dict:
    """Load a standard 3DGS PLY file.

    Returns:
        dict with keys: positions (N,3), rotations (N,4) wxyz, scales (N,3),
        opacities (N,).
    """
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header.append(line)
            if line == "end_header":
                break
        vertex_count = 0
        props = []
        for line in header:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                props.append((parts[2], parts[1]))

        dtype_map = {"float": np.float32, "double": np.float64}
        dtype = np.dtype([(name, dtype_map.get(typ, np.float32)) for name, typ in props])
        data = np.fromfile(f, dtype=dtype, count=vertex_count)

    positions = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    scales = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1).astype(np.float32)
    rotations = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1).astype(np.float32)
    opacities = np.asarray(data["opacity"], dtype=np.float32)

    log.info("Loaded %d Gaussians from %s", vertex_count, path)
    return {
        "positions": torch.from_numpy(positions).to(device),
        "rotations": torch.from_numpy(rotations).to(device),
        "scales": torch.from_numpy(scales).to(device),
        "opacities": torch.from_numpy(opacities).to(device),
    }


def apply_activations(data: dict) -> None:
    """Apply 3DGS activations in-place: exp(scale), sigmoid(opacity)."""
    data["scales"] = torch.exp(data["scales"])
    data["opacities"] = torch.sigmoid(data["opacities"])


def apply_scaling(data: dict, scale: float) -> None:
    """Rescale positions and scales in-place."""
    if scale != 1.0:
        log.info("Applying scale=%.4f", scale)
    data["positions"] = data["positions"] * scale
    data["scales"] = data["scales"] * scale


# ============================================================================
# POINT CLOUD SAMPLING
# ============================================================================

def _build_rotation_matrices(rotations: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions (N,4) to rotation matrices (N,3,3)."""
    w, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    norm = torch.sqrt(w*w + x*x + y*y + z*z).clamp(min=1e-8)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    R = torch.stack([
        1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y),
          2*(x*y+w*z),  1-2*(x*x+z*z),  2*(y*z-w*x),
          2*(x*z-w*y),  2*(y*z+w*x),    1-2*(x*x+y*y),
    ], dim=1).reshape(-1, 3, 3)
    return R


def sample_gaussian_mixture(data: dict, num_samples: int, opacity_threshold: float, seed: int = 42) -> np.ndarray:
    """Sample 3D points from the Gaussian mixture defined by the splat.

    Each point: select component i ∝ opacity_i, then x = mu_i + R_i @ (s_i * z), z~N(0,I).

    Args:
        data: dict with positions (N,3), rotations (N,4) wxyz, scales (N,3), opacities (N,).
        num_samples: number of points to draw.
        opacity_threshold: ignore Gaussians below this opacity.
        seed: RNG seed.

    Returns:
        (num_samples, 3) float32 numpy array.
    """
    torch.manual_seed(seed)
    device = data["positions"].device

    mask = data["opacities"] > opacity_threshold
    positions = data["positions"][mask]
    rotations = data["rotations"][mask]
    scales    = data["scales"][mask]
    opacities = data["opacities"][mask]

    log.info("Gaussian mixture sampling: %d / %d Gaussians (opacity > %.2f)",
             mask.sum().item(), data["opacities"].shape[0], opacity_threshold)

    R = _build_rotation_matrices(rotations)  # (K,3,3)

    # sample component indices proportional to opacity
    component_ids = torch.multinomial(opacities, num_samples, replacement=True)  # (S,)

    mu = positions[component_ids]      # (S,3)
    s  = scales[component_ids]         # (S,3)
    Ri = R[component_ids]              # (S,3,3)

    z = torch.randn(num_samples, 3, 1, device=device)          # (S,3,1)
    pts = mu + (Ri @ (s.unsqueeze(-1) * z)).squeeze(-1)        # (S,3)

    _trace(f"Gaussian mixture sampling -> {pts.shape[0]} points")
    return pts.cpu().numpy().astype(np.float32)


def sample_kaolin(data: dict, octree_level: int, opacity_threshold: float) -> np.ndarray:
    """Sample a dense interior point cloud via the kaolin Gaussian densifier.

    Falls back to opacity-filtered Gaussian centers on failure.

    Returns:
        (M, 3) float32 numpy array.
    """
    import kaolin as kal

    _trace(f"sample_points_in_volume (octree_level={octree_level})")
    try:
        pts = kal.ops.gaussian.sample_points_in_volume(
            xyz=data["positions"],
            scale=data["scales"],
            rotation=data["rotations"],
            opacity=data["opacities"],
            clip_samples_to_input_bbox=False,
            octree_level=octree_level,
            opacity_threshold=opacity_threshold,
        )
        pts_np = pts.cpu().numpy().astype(np.float32)
        _trace(f"sample_points_in_volume -> {pts_np.shape[0]} points")
        if pts_np.shape[0] > 0:
            return pts_np
        log.warning("Densifier returned 0 points — falling back to Gaussian centers.")
    except Exception as exc:
        log.warning("Densifier failed (%s) — falling back to Gaussian centers.", exc)

    return _gaussian_centers_fallback(data, opacity_threshold)


def _gaussian_centers_fallback(data: dict, opacity_threshold: float) -> np.ndarray:
    mask = data["opacities"] > opacity_threshold
    pts = data["positions"][mask].cpu().numpy().astype(np.float32)
    log.info("Gaussian centers: %d (opacity > %.2f)", pts.shape[0], opacity_threshold)
    return pts


# ============================================================================
# DENSITY FIELD: POINT CLOUD → VOXEL GRID → SMOOTH
# ============================================================================

def build_density_grid(
    pts: np.ndarray,
    resolution: int,
    smooth_sigma: float,
    padding: int = 4,
    fill_interior: bool = True,
    closing_radius: float = 2.0,
):
    """Voxelize a point cloud and smooth it to produce a density field.

    Args:
        pts: (M, 3) interior point cloud.
        resolution: grid resolution along the longest axis.
        smooth_sigma: Gaussian blur sigma in voxels (higher = smoother).
        padding: empty voxel margin around the bounding box.
        fill_interior: apply morphological closing then flood-fill to make the
            occupancy grid solid. Fixes flat/shell meshes from single-sided captures.
        closing_radius: structuring element radius in voxels for morphological closing.

    Returns:
        grid: (Rx, Ry, Rz) float32 density array, values in [0, 1].
        origin: (3,) world-space origin of the grid.
        voxel_size: scalar, world units per voxel.
    """
    try:
        from scipy.ndimage import gaussian_filter, binary_closing, binary_fill_holes
        from scipy.ndimage import generate_binary_structure, iterate_structure
    except ImportError:
        log.error("scipy is required: pip install scipy")
        sys.exit(1)

    bb_min = pts.min(axis=0)
    bb_max = pts.max(axis=0)
    extents = bb_max - bb_min
    voxel_size = float(extents.max() / (resolution - 2 * padding))

    dims = (np.ceil(extents / voxel_size).astype(int) + 2 * padding + 1).tolist()
    origin = bb_min - padding * voxel_size
    log.info("Density grid: %s, voxel_size=%.5f", dims, voxel_size)

    indices = np.floor((pts - origin) / voxel_size).astype(int)
    indices = np.clip(indices, 0, np.array(dims) - 1)

    grid = np.zeros(dims, dtype=np.float32)
    np.add.at(grid, (indices[:, 0], indices[:, 1], indices[:, 2]), 1.0)

    if grid.max() > 0:
        grid /= grid.max()

    if fill_interior:
        # morphological closing bridges gaps between front/back shell points,
        # then flood-fill turns the closed shell into a solid binary volume
        _trace(f"Morphological closing (radius={closing_radius:.1f} voxels) + flood-fill")
        occ = grid > 0.0
        struct = generate_binary_structure(3, 1)
        r = max(1, int(closing_radius))
        struct = iterate_structure(struct, r)
        occ = binary_closing(occ, structure=struct)
        # fill interior slice-by-slice along each axis, take union
        filled = (
            binary_fill_holes(occ, structure=generate_binary_structure(3, 1))
        )
        grid = filled.astype(np.float32)

    if smooth_sigma > 0:
        grid = gaussian_filter(grid.astype(np.float64), sigma=smooth_sigma).astype(np.float32)
        if grid.max() > 0:
            grid /= grid.max()

    return grid, origin, voxel_size


# ============================================================================
# FLEXICUBES SURFACE EXTRACTION
# ============================================================================

def extract_surface_flexicubes(
    density_grid: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    fc_resolution: int,
    iso_level: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a surface mesh with FlexiCubes from a smooth density field.

    Interpolates the pre-built density grid onto a FlexiCubes voxel grid, converts
    it to a signed scalar field (negative inside), then calls FlexiCubes.

    Args:
        density_grid: (Rx, Ry, Rz) float32 density in [0,1].
        origin: world-space origin of density_grid.
        voxel_size: world units per density voxel.
        fc_resolution: FlexiCubes grid resolution.
        iso_level: density threshold for the surface (0–1).
        device: torch device string.

    Returns:
        verts: (V, 3) float32 world-space vertices.
        faces: (F, 3) int32 face indices.
    """
    from kaolin.ops.conversions import FlexiCubes

    _trace(f"Building FlexiCubes grid (resolution={fc_resolution})")
    fc = FlexiCubes(device=device)
    # verts_unit in [-0.5, 0.5], cube_idx shape (num_cubes, 8)
    verts_unit, cube_idx = fc.construct_voxel_grid(fc_resolution)

    # Map unit verts to the bounding box of the density grid
    grid_dims = np.array(density_grid.shape, dtype=np.float32)
    bb_min = torch.tensor(origin, dtype=torch.float32, device=device)
    bb_max = torch.tensor(
        origin + (grid_dims - 1) * voxel_size, dtype=torch.float32, device=device
    )
    # Rescale from [-0.5, 0.5] to [bb_min, bb_max]
    verts_world = verts_unit * (bb_max - bb_min) + (bb_min + bb_max) * 0.5

    # Interpolate density grid at FlexiCubes vertex positions using grid_sample.
    # grid_sample expects input in [-1, 1] per axis, shape (1, 1, D, H, W).
    density_t = torch.from_numpy(density_grid).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,Rx,Ry,Rz)

    # Normalize verts_world to [-1, 1] within the density grid extent
    verts_norm = (verts_world - bb_min) / (bb_max - bb_min) * 2.0 - 1.0  # (N, 3)
    # grid_sample with input (B,C,D,H,W) maps sample coord (gx,gy,gz) → (W,H,D).
    # Our grid is (Rx,Ry,Rz)=(D,H,W), so world x→D, y→H, z→W.
    # We must pass coords as (gz=x, gy=y, gx=z) = reversed xyz.
    sample_coords = verts_norm[:, [2, 1, 0]].reshape(1, 1, 1, -1, 3)  # (1, 1, 1, N, 3)
    density_at_verts = F.grid_sample(
        density_t, sample_coords, mode="bilinear", padding_mode="zeros", align_corners=True
    ).reshape(-1)  # (N,)

    # Convert density to scalar field: negative = inside (density > iso_level)
    scalar_field = iso_level - density_at_verts  # negative inside

    _trace("Running FlexiCubes mesh extraction")
    with torch.no_grad():
        verts_fc, faces_fc, _ = fc(
            verts_world, scalar_field, cube_idx, fc_resolution, training=False
        )

    verts_out = verts_fc.cpu().numpy().astype(np.float32)
    faces_out = faces_fc.cpu().numpy().astype(np.int32)
    log.info("FlexiCubes surface: %d verts, %d faces", verts_out.shape[0], faces_out.shape[0])
    return verts_out, faces_out


# ============================================================================
# MESH I/O
# ============================================================================

def save_ply_pointcloud(path: str, pts: np.ndarray) -> None:
    """Write a point cloud to a binary PLY file."""
    n = pts.shape[0]
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(pts.astype(np.float32).tobytes())
    log.info("Saved point cloud: %s (%d pts)", path, n)


def save_obj(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write a triangle mesh to a Wavefront OBJ file."""
    with open(path, "w") as f:
        f.write(f"# {verts.shape[0]} vertices, {faces.shape[0]} faces\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    log.info("Saved surface mesh: %s", path)


# ============================================================================
# TETRAHEDRALIZATION (fTetWild via wildmeshing)
# ============================================================================

def tetrahedralize(surface_obj_path: str, msh_path: str, stop_quality: float) -> None:
    """Tetrahedralize a surface mesh using fTetWild (via wildmeshing) and save as .msh.

    The .msh can be loaded by Newton's XPBD solver:
        import meshio
        m = meshio.read("object.msh")
        nodes, tets = m.points, m.cells_dict["tetra"]

    Args:
        surface_obj_path: input OBJ surface mesh.
        msh_path: output .msh tet mesh path.
        stop_quality: target tet quality; lower is higher quality but slower (range 5–20).
    """
    try:
        import wildmeshing
    except ImportError:
        log.error("wildmeshing (fTetWild) is required: pip install wildmeshing")
        sys.exit(1)

    _trace(f"Tetrahedralizing with fTetWild (stop_quality={stop_quality})")
    tetra = wildmeshing.Tetrahedralizer(stop_quality=stop_quality)
    tetra.load_mesh(surface_obj_path)
    tetra.tetrahedralize()
    tetra.save(msh_path)
    log.info("Saved tet mesh: %s", msh_path)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert 3DGS PLY to surface mesh (.obj, via FlexiCubes) "
            "and tet volume mesh (.msh, via fTetWild)."
        )
    )
    parser.add_argument("ply_path", help="Input 3DGS PLY file")
    parser.add_argument("output_dir", help="Output directory (created if missing)")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform rescaling applied to positions and scales (default: 1.0)",
    )
    parser.add_argument(
        "--density-resolution",
        type=int,
        default=128,
        metavar="R",
        help="Resolution of the intermediate density voxel grid along the longest axis "
             "(default: 128; increase for finer density capture)",
    )
    parser.add_argument(
        "--fc-resolution",
        type=int,
        default=64,
        metavar="R",
        help="FlexiCubes grid resolution (default: 64; higher = more detail but slower)",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.5,
        metavar="S",
        help="Gaussian blur sigma in voxels applied to the density grid (default: 1.5)",
    )
    parser.add_argument(
        "--iso-level",
        type=float,
        default=0.1,
        metavar="L",
        help="Density iso-level for the surface in [0,1] (default: 0.1; higher = tighter)",
    )
    parser.add_argument(
        "--opacity-threshold",
        type=float,
        default=0.05,
        metavar="T",
        help="Opacity threshold for kaolin densifier / Gaussian center fallback (default: 0.05)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200_000,
        metavar="N",
        help="Number of points to draw from Gaussian mixture (default: 200000)",
    )
    parser.add_argument(
        "--use-kaolin-sampler",
        action="store_true",
        help="Use kaolin sample_points_in_volume instead of Gaussian mixture sampling",
    )
    parser.add_argument(
        "--octree-level",
        type=int,
        default=7,
        choices=(6, 7, 8, 9, 10),
        metavar="L",
        help="Kaolin densifier octree level, only used with --use-kaolin-sampler (default: 7)",
    )
    parser.add_argument(
        "--use-gaussian-centers",
        action="store_true",
        help="Use opacity-filtered Gaussian centers directly (fast, lower quality)",
    )
    parser.add_argument(
        "--no-fill",
        action="store_true",
        help="Disable morphological closing + flood-fill (use if mesh is already solid)",
    )
    parser.add_argument(
        "--closing-radius",
        type=float,
        default=2.0,
        metavar="R",
        help="Structuring element radius in voxels for morphological closing (default: 2.0)",
    )
    parser.add_argument(
        "--no-tet",
        action="store_true",
        help="Skip tetrahedralization — only produce the surface OBJ",
    )
    parser.add_argument(
        "--tet-quality",
        type=float,
        default=10.0,
        metavar="Q",
        help="fTetWild stop_quality: lower = higher quality, slower (default: 10)",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default="object",
        help="Base name for output files (default: object)",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    if not os.path.isfile(args.ply_path):
        log.error("PLY not found: %s", args.ply_path)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    obj_path = os.path.join(args.output_dir, f"{args.object_name}.obj")
    msh_path = os.path.join(args.output_dir, f"{args.object_name}.msh")

    # --- Load ---
    _trace("Loading PLY")
    data = load_ply_3dgs(args.ply_path, device=args.device)
    apply_activations(data)
    apply_scaling(data, args.scale)

    # --- Interior point cloud ---
    if args.use_gaussian_centers:
        pts = _gaussian_centers_fallback(data, args.opacity_threshold)
    elif args.use_kaolin_sampler:
        pts = sample_kaolin(data, args.octree_level, args.opacity_threshold)
    else:
        pts = sample_gaussian_mixture(data, args.num_samples, args.opacity_threshold)

    if pts.shape[0] == 0:
        log.error("No points sampled — cannot extract mesh.")
        sys.exit(1)

    # --- Export sample point cloud ---
    pts_ply_path = os.path.join(args.output_dir, f"{args.object_name}_samples.ply")
    save_ply_pointcloud(pts_ply_path, pts)

    # --- Density grid ---
    _trace("Building density grid from point cloud")
    density_grid, origin, voxel_size = build_density_grid(
        pts,
        resolution=args.density_resolution,
        smooth_sigma=args.smooth_sigma,
        fill_interior=not args.no_fill,
        closing_radius=args.closing_radius,
    )

    # --- FlexiCubes surface ---
    _trace("Extracting surface mesh with FlexiCubes")
    verts, faces = extract_surface_flexicubes(
        density_grid,
        origin,
        voxel_size,
        fc_resolution=args.fc_resolution,
        iso_level=args.iso_level,
        device=args.device,
    )
    save_obj(obj_path, verts, faces)

    # --- fTetWild tet mesh ---
    if not args.no_tet:
        tetrahedralize(obj_path, msh_path, stop_quality=args.tet_quality)
    else:
        log.info("Skipping tetrahedralization (--no-tet).")

    _trace("Done.")
    print("\nOutputs:")
    print(f"  Sample cloud : {pts_ply_path}")
    print(f"  Surface mesh : {obj_path}")
    if not args.no_tet:
        print(f"  Tet mesh     : {msh_path}")
    print()
    print("Load tet mesh in Newton XPBD:")
    print("    import meshio")
    print(f"    m = meshio.read('{msh_path}')")
    print("    nodes, tets = m.points, m.cells_dict['tetra']")


if __name__ == "__main__":
    main()
