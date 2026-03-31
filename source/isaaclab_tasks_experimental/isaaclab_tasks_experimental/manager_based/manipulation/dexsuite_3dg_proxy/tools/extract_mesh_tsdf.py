#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Extract a surface mesh from a 3DGS PLY file using Poisson reconstruction.

Since we cannot render depth maps from Gaussians directly, this script samples
3D points from the Gaussian mixture distribution and reconstructs a surface
from the resulting oriented point cloud using Poisson reconstruction (which
computes a watertight implicit SDF, analogous to TSDF fusion).

Pipeline:
  1. Load the 3DGS PLY (positions, rotations, scales, opacities).
  2. Sample 3D points from the Gaussian mixture (each Gaussian ~ N(mu, Sigma)).
  3. Compute per-point normals from each Gaussian's smallest-scale axis.
  4. Run open3d Poisson surface reconstruction on the oriented point cloud.
  5. Post-process: remove small disconnected components.
  6. (Optional) Tetrahedralize with fTetWild via wildmeshing (.msh).

Requirements:
    pip install open3d wildmeshing meshio tqdm

Usage:
    python extract_mesh_tsdf.py input.ply output_dir/
    python extract_mesh_tsdf.py input.ply output_dir/ --num-samples 500000 --poisson-depth 10
    python extract_mesh_tsdf.py input.ply output_dir/ --no-tet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

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
    scales    = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1).astype(np.float32)
    rotations = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1).astype(np.float32)
    opacities = np.asarray(data["opacity"], dtype=np.float32)

    log.info("Loaded %d Gaussians from %s", vertex_count, path)
    return {
        "positions": torch.from_numpy(positions).to(device),
        "rotations": torch.from_numpy(rotations).to(device),
        "scales":    torch.from_numpy(scales).to(device),
        "opacities": torch.from_numpy(opacities).to(device),
    }


def apply_activations(data: dict) -> None:
    """Apply 3DGS activations in-place: exp(scale), sigmoid(opacity)."""
    data["scales"]    = torch.exp(data["scales"])
    data["opacities"] = torch.sigmoid(data["opacities"])


def apply_scaling(data: dict, scale: float) -> None:
    """Rescale positions and scales in-place."""
    if scale != 1.0:
        log.info("Applying scale=%.4f", scale)
    data["positions"] = data["positions"] * scale
    data["scales"]    = data["scales"] * scale


# ============================================================================
# GAUSSIAN MIXTURE SAMPLING
# ============================================================================

def build_rotation_matrices(rotations: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions to 3x3 rotation matrices.

    Args:
        rotations: (N, 4) quaternion tensor in wxyz convention.

    Returns:
        R: (N, 3, 3) rotation matrices.
    """
    w, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def sample_from_gaussian_mixture(
    positions: torch.Tensor,
    R: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample 3D points from the Gaussian mixture distribution.

    Each Gaussian i defines N(mu_i, Sigma_i) where Sigma_i = R_i @ diag(s_i^2) @ R_i^T.
    Components are selected with probability proportional to opacity.

    Args:
        positions: (N, 3) Gaussian centers.
        R: (N, 3, 3) rotation matrices.
        scales: (N, 3) post-activation scales (standard deviations per axis).
        opacities: (N,) post-activation opacities used as mixture weights.
        num_samples: number of points to sample.

    Returns:
        pts: (num_samples, 3) sampled world-space positions.
        component_ids: (num_samples,) index of the Gaussian each sample came from.
    """
    device = positions.device

    # Sample component indices proportional to opacity
    weights = opacities / opacities.sum()
    component_ids = torch.multinomial(weights, num_samples, replacement=True)

    # Sample x = mu + R @ (s * z),  z ~ N(0, I)
    z = torch.randn(num_samples, 3, device=device)
    sel_R      = R[component_ids]         # (S, 3, 3)
    sel_scales = scales[component_ids]    # (S, 3)
    sel_mu     = positions[component_ids] # (S, 3)

    local_pts = sel_scales * z                                  # (S, 3)
    pts = sel_mu + (sel_R @ local_pts.unsqueeze(-1)).squeeze(-1) # (S, 3)
    return pts, component_ids


def compute_normals(R: torch.Tensor, scales: torch.Tensor, component_ids: torch.Tensor) -> torch.Tensor:
    """Compute surface normals for sampled points.

    The normal direction of each Gaussian is its smallest-scale axis (the
    "flat" direction of the ellipsoid), expressed as the corresponding column
    of its rotation matrix.

    Args:
        R: (N, 3, 3) rotation matrices.
        scales: (N, 3) Gaussian scales.
        component_ids: (S,) which Gaussian each sample came from.

    Returns:
        normals: (S, 3) unit normal vectors.
    """
    min_idx = scales.argmin(dim=-1)  # (N,) index of smallest scale per Gaussian

    # Select the column of R corresponding to the smallest-scale axis
    # R[i, :, min_idx[i]] is the world-space direction of that axis
    min_idx_expanded = min_idx.view(-1, 1, 1).expand(-1, 3, 1)  # (N, 3, 1)
    gaussian_normals = R.gather(2, min_idx_expanded).squeeze(-1)  # (N, 3)

    normals = gaussian_normals[component_ids]  # (S, 3)
    normals = torch.nn.functional.normalize(normals, dim=-1)
    return normals


# ============================================================================
# MESH EXTRACTION: POISSON RECONSTRUCTION
# ============================================================================

def poisson_reconstruct(
    pts_np: np.ndarray,
    normals_np: np.ndarray,
    poisson_depth: int,
    density_quantile: float,
) -> "o3d.geometry.TriangleMesh":
    """Run open3d Poisson surface reconstruction on an oriented point cloud.

    Args:
        pts_np: (S, 3) float32 point positions.
        normals_np: (S, 3) float32 oriented normals.
        poisson_depth: octree depth for Poisson solver (higher = more detail).
        density_quantile: vertices below this density quantile are removed
            to trim low-support regions (0 = keep all, 0.1 = aggressive trim).

    Returns:
        mesh: open3d TriangleMesh.
    """
    import open3d as o3d

    _trace(f"Poisson reconstruction (depth={poisson_depth}, density_quantile={density_quantile:.2f})")
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(pts_np.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals_np.astype(np.float64))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, linear_fit=False
    )
    # Remove vertices with low support (boundaries / floaters)
    if density_quantile > 0.0:
        densities_np = np.asarray(densities)
        threshold = np.quantile(densities_np, density_quantile)
        vertices_to_remove = densities_np < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.compute_vertex_normals()
    log.info(
        "Poisson mesh: %d verts, %d faces",
        len(mesh.vertices), len(mesh.triangles),
    )
    return mesh


def post_process_mesh(mesh: "o3d.geometry.TriangleMesh", num_clusters: int = 1) -> "o3d.geometry.TriangleMesh":
    """Keep only the largest connected component(s) and remove degenerate triangles.

    Adapted from pgsr-poisson/render.py.

    Args:
        mesh: input open3d TriangleMesh.
        num_clusters: number of largest clusters to keep.

    Returns:
        Cleaned mesh.
    """
    import open3d as o3d
    import copy

    mesh_out = copy.deepcopy(mesh)
    triangle_clusters, cluster_n_triangles, _ = mesh_out.cluster_connected_triangles()
    triangle_clusters  = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    n_min = np.sort(cluster_n_triangles)[-num_clusters]
    n_min = max(n_min, 50)
    remove_mask = cluster_n_triangles[triangle_clusters] < n_min
    mesh_out.remove_triangles_by_mask(remove_mask)
    mesh_out.remove_unreferenced_vertices()
    mesh_out.remove_degenerate_triangles()

    log.info(
        "Post-processed mesh: %d verts, %d faces",
        len(mesh_out.vertices), len(mesh_out.triangles),
    )
    return mesh_out


# ============================================================================
# MESH I/O
# ============================================================================

def save_obj(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write a triangle mesh to a Wavefront OBJ file."""
    with open(path, "w") as f:
        f.write(f"# {verts.shape[0]} vertices, {faces.shape[0]} faces\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    log.info("Saved surface OBJ: %s", path)


# ============================================================================
# TETRAHEDRALIZATION (fTetWild via wildmeshing)
# ============================================================================

def tetrahedralize(obj_path: str, msh_path: str, stop_quality: float) -> None:
    """Tetrahedralize a surface mesh with fTetWild and save as Gmsh .msh.

    The output can be loaded for Newton XPBD:
        import meshio
        m = meshio.read("object.msh")
        nodes, tets = m.points, m.cells_dict["tetra"]

    Args:
        obj_path: input OBJ surface mesh.
        msh_path: output Gmsh .msh tet mesh path.
        stop_quality: target tet quality (lower = higher quality, slower; range 5–20).
    """
    try:
        import wildmeshing
    except ImportError:
        log.error("wildmeshing (fTetWild) is required: pip install wildmeshing")
        sys.exit(1)
    try:
        import meshio
    except ImportError:
        log.error("meshio is required: pip install meshio")
        sys.exit(1)

    _trace(f"fTetWild tetrahedralization (stop_quality={stop_quality})")
    tetra = wildmeshing.Tetrahedralizer(stop_quality=stop_quality)
    tetra.load_mesh(obj_path)
    tetra.tetrahedralize()
    V, T = tetra.get_tet_mesh()
    V = np.asarray(V, dtype=np.float64)
    T = np.asarray(T, dtype=np.int32)
    log.info("Tet mesh: %d verts, %d tets", V.shape[0], T.shape[0])

    meshio.write(msh_path, meshio.Mesh(points=V, cells=[("tetra", T)]))
    log.info("Saved tet mesh: %s", msh_path)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract a surface mesh from a 3DGS PLY via Gaussian mixture sampling "
            "+ Poisson reconstruction, then optionally tetrahedralize with fTetWild."
        )
    )
    parser.add_argument("ply_path", help="Input 3DGS PLY file")
    parser.add_argument("output_dir", help="Output directory (created if missing)")
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Uniform rescaling of positions and scales (default: 1.0)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=500_000, metavar="N",
        help="Number of points to sample from the Gaussian mixture (default: 500000)",
    )
    parser.add_argument(
        "--opacity-threshold", type=float, default=0.05, metavar="T",
        help="Discard Gaussians with opacity below this threshold before sampling (default: 0.05)",
    )
    parser.add_argument(
        "--poisson-depth", type=int, default=9, metavar="D",
        help="Poisson reconstruction octree depth (default: 9; higher = more detail)",
    )
    parser.add_argument(
        "--density-quantile", type=float, default=0.05, metavar="Q",
        help="Remove Poisson vertices below this density quantile (default: 0.05; 0 = keep all)",
    )
    parser.add_argument(
        "--num-clusters", type=int, default=1, metavar="K",
        help="Number of largest connected components to keep (default: 1)",
    )
    parser.add_argument(
        "--no-tet", action="store_true",
        help="Skip fTetWild tetrahedralization (surface mesh only)",
    )
    parser.add_argument(
        "--tet-quality", type=float, default=10.0, metavar="Q",
        help="fTetWild stop_quality: lower = higher quality, slower (default: 10)",
    )
    parser.add_argument(
        "--object-name", type=str, default="object",
        help="Base name for output files (default: object)",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    if not os.path.isfile(args.ply_path):
        log.error("PLY not found: %s", args.ply_path)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    ply_out  = os.path.join(args.output_dir, f"{args.object_name}.ply")
    obj_path = os.path.join(args.output_dir, f"{args.object_name}.obj")
    msh_path = os.path.join(args.output_dir, f"{args.object_name}.msh")

    # --- Load ---
    _trace("Loading PLY")
    data = load_ply_3dgs(args.ply_path, device=args.device)
    apply_activations(data)
    apply_scaling(data, args.scale)

    # Filter low-opacity Gaussians
    mask = data["opacities"] > args.opacity_threshold
    for k in data:
        data[k] = data[k][mask]
    log.info("Kept %d / %d Gaussians (opacity > %.2f)", mask.sum().item(), mask.shape[0], args.opacity_threshold)

    # --- Build rotation matrices ---
    R = build_rotation_matrices(data["rotations"])

    # --- Sample from Gaussian mixture ---
    _trace(f"Sampling {args.num_samples} points from Gaussian mixture")
    pts, component_ids = sample_from_gaussian_mixture(
        data["positions"], R, data["scales"], data["opacities"], args.num_samples
    )
    normals = compute_normals(R, data["scales"], component_ids)

    pts_np     = pts.cpu().numpy().astype(np.float32)
    normals_np = normals.cpu().numpy().astype(np.float32)

    # --- Poisson reconstruction ---
    mesh = poisson_reconstruct(pts_np, normals_np, args.poisson_depth, args.density_quantile)
    mesh = post_process_mesh(mesh, num_clusters=args.num_clusters)

    import open3d as o3d
    o3d.io.write_triangle_mesh(ply_out, mesh, write_vertex_normals=True)
    log.info("Saved surface PLY: %s", ply_out)

    # Also save as OBJ for fTetWild
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    save_obj(obj_path, verts, faces)

    # --- fTetWild ---
    if not args.no_tet:
        tetrahedralize(obj_path, msh_path, stop_quality=args.tet_quality)

    _trace("Done.")
    print("\nOutputs:")
    print(f"  Surface PLY : {ply_out}")
    print(f"  Surface OBJ : {obj_path}")
    if not args.no_tet:
        print(f"  Tet mesh    : {msh_path}")


if __name__ == "__main__":
    main()
