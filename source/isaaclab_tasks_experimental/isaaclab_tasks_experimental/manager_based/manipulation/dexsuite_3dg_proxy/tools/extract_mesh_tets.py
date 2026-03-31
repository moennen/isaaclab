#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Extract a tetrahedral mesh from a 3DGS PLY file using marching tetrahedra.

Pipeline:
  1. Load the 3DGS PLY (positions, rotations, scales, opacities).
  2. Build tet vertices: Gaussian centers + jitter + boundary cage points.
  3. Delaunay-tetrahedralize those vertices (scipy.spatial.Delaunay).
  4. Evaluate the Gaussian alpha field at every tet vertex (chunked GPU).
  5. Run marching tetrahedra (tetmesh.py) with alpha > threshold as "inside".
  6. Post-process: keep largest connected component(s).
  7. Save surface mesh (.ply / .obj).
  8. (Optional) Tetrahedralize with fTetWild via wildmeshing (.msh).

The alpha field at a 3D query point x is:
    alpha(x) = 1 - prod_i (1 - sigmoid(opacity_i_raw) * exp(-0.5 * d_i^T Cov_i^{-1} d_i))
where d_i = x - mu_i and Cov_i = R_i @ diag(scales_i^2) @ R_i^T.
In practice we accumulate the log-transmittance in chunks for numerical stability.

Requirements:
    pip install open3d trimesh wildmeshing meshio tqdm scipy

Usage:
    python extract_mesh_tets.py input.ply output_dir/
    python extract_mesh_tets.py input.ply output_dir/ --alpha-threshold 0.5 --no-tet
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


# ============================================================================
# PLY LOADING  (identical helper to extract_mesh_tsdf.py)
# ============================================================================

def load_ply_3dgs(path: str, device: str = "cuda") -> dict:
    """Load a standard 3DGS PLY file.

    Returns dict with keys:
        positions  (N,3), rotations (N,4) wxyz, scales (N,3), opacities (N,).
    Activations are applied: scales = exp(raw_scales), opacities = sigmoid(raw_opacity).
    """
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("Install plyfile: pip install plyfile")

    log.info("Loading PLY: %s", path)
    plydata = PlyData.read(path)
    v = plydata["vertex"]
    props = [p.name for p in v.properties]

    def _get(name):
        return torch.tensor(np.asarray(v[name], dtype=np.float32), device=device)

    positions = torch.stack([_get("x"), _get("y"), _get("z")], dim=1)

    # rotations stored as rot_0..rot_3 = (w, x, y, z)
    rot_names = ["rot_0", "rot_1", "rot_2", "rot_3"]
    if all(r in props for r in rot_names):
        rotations = torch.stack([_get(r) for r in rot_names], dim=1)  # (N,4) wxyz
    else:
        rotations = torch.zeros(positions.shape[0], 4, device=device)
        rotations[:, 0] = 1.0  # identity

    # scales stored as log-scales
    scale_names = ["scale_0", "scale_1", "scale_2"]
    if all(s in props for s in scale_names):
        raw_scales = torch.stack([_get(s) for s in scale_names], dim=1)
        scales = torch.exp(raw_scales)
    else:
        scales = torch.ones(positions.shape[0], 3, device=device) * 0.01

    # opacity stored as raw logit
    if "opacity" in props:
        raw_opacity = _get("opacity")
        opacities = torch.sigmoid(raw_opacity)
    else:
        opacities = torch.ones(positions.shape[0], device=device) * 0.5

    log.info("  Loaded %d Gaussians", positions.shape[0])
    return dict(positions=positions, rotations=rotations, scales=scales, opacities=opacities)


# ============================================================================
# ROTATION MATRICES
# ============================================================================

def build_rotation_matrices(rotations: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions (N,4) to rotation matrices (N,3,3)."""
    w, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    norm = torch.sqrt(w * w + x * x + y * y + z * z).clamp(min=1e-8)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    R = torch.stack([
        1 - 2 * (y * y + z * z),     2 * (x * y - w * z),     2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z),     2 * (y * z - w * x),
            2 * (x * z - w * y),     2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(-1, 3, 3)
    return R


# ============================================================================
# BUILD TET VERTICES
# ============================================================================

def build_tet_vertices(
    positions: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    *,
    opacity_cutoff: float = 0.05,
    jitter_scale: float = 0.5,
    num_jitter_per_gaussian: int = 3,
    boundary_padding: float = 0.1,
    num_boundary_pts: int = 200,
    seed: int = 42,
) -> torch.Tensor:
    """Construct tet vertex set from Gaussian centers + jittered copies + boundary cage.

    Args:
        positions: (N,3) Gaussian centers.
        scales: (N,3) Gaussian radii per axis.
        opacities: (N,) Gaussian opacities in [0,1].
        opacity_cutoff: Discard Gaussians below this opacity.
        jitter_scale: Jitter radius relative to mean scale.
        num_jitter_per_gaussian: Extra jittered copies per Gaussian.
        boundary_padding: Fractional padding for the bounding box cage.
        num_boundary_pts: Number of random cage points outside the object.
        seed: RNG seed.

    Returns:
        (M,3) float32 tensor of all tet vertex positions.
    """
    torch.manual_seed(seed)
    device = positions.device

    mask = opacities > opacity_cutoff
    pos = positions[mask]
    sc = scales[mask]
    log.info("  Using %d / %d Gaussians for tet vertices (opacity > %.2f)",
             mask.sum().item(), positions.shape[0], opacity_cutoff)

    pts = [pos]

    # jittered copies
    if num_jitter_per_gaussian > 0:
        mean_scale = sc.mean(dim=1, keepdim=True)  # (M,1)
        noise = torch.randn(pos.shape[0] * num_jitter_per_gaussian, 3, device=device)
        centers_rep = pos.repeat_interleave(num_jitter_per_gaussian, dim=0)
        scale_rep = mean_scale.repeat_interleave(num_jitter_per_gaussian, dim=0)
        jitter = centers_rep + noise * scale_rep * jitter_scale
        pts.append(jitter)

    all_pts = torch.cat(pts, dim=0)

    # boundary cage — random points on an inflated bounding box surface
    lo = all_pts.min(dim=0).values
    hi = all_pts.max(dim=0).values
    diag = hi - lo
    lo_pad = lo - diag * boundary_padding
    hi_pad = hi + diag * boundary_padding

    cage_pts = torch.rand(num_boundary_pts, 3, device=device) * (hi_pad - lo_pad) + lo_pad
    # push them outside by biasing each point to be near one of the 6 faces
    face_idx = torch.randint(0, 6, (num_boundary_pts,), device=device)
    for i, (dim, side) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]):
        mask_f = face_idx == i
        cage_pts[mask_f, dim] = lo_pad[dim] if side == 0 else hi_pad[dim]

    all_pts = torch.cat([all_pts, cage_pts], dim=0)
    log.info("  Total tet vertices: %d", all_pts.shape[0])
    return all_pts


# ============================================================================
# GAUSSIAN ALPHA FIELD EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_alpha_field(
    query_pts: torch.Tensor,
    positions: torch.Tensor,
    R: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    *,
    chunk_size: int = 4096,
    opacity_cutoff: float = 0.01,
) -> torch.Tensor:
    """Evaluate Gaussian alpha at each query point (volumetric).

    alpha(x) = 1 - prod_i (1 - o_i * exp(-0.5 * d_i^T Cov_i^{-1} d_i))

    Computed as: alpha = 1 - exp(sum_i log(1 - contrib_i))

    Args:
        query_pts: (M,3) points to evaluate at.
        positions: (N,3) Gaussian centers.
        R: (N,3,3) rotation matrices.
        scales: (N,3) Gaussian radii per axis.
        opacities: (N,) activated opacities in [0,1].
        chunk_size: Number of query points per GPU chunk.
        opacity_cutoff: Skip Gaussians below this opacity.

    Returns:
        (M,) alpha values in [0,1].
    """
    device = query_pts.device
    mask = opacities > opacity_cutoff
    pos = positions[mask]       # (K,3)
    rot = R[mask]               # (K,3,3)
    sc = scales[mask]           # (K,3)
    op = opacities[mask]        # (K,)

    # Precompute Cov^{-1}: for each Gaussian, Cov = R @ diag(s^2) @ R^T
    # Cov^{-1} = R @ diag(1/s^2) @ R^T
    inv_s2 = 1.0 / (sc ** 2 + 1e-12)                    # (K,3)
    # store as (K,3,3): R @ diag(inv_s2) @ R^T
    # cov_inv[k] = rot[k] @ diag(inv_s2[k]) @ rot[k].T
    # We'll compute this on the fly per chunk to save VRAM

    M = query_pts.shape[0]
    log_transmittance = torch.zeros(M, device=device)

    for start in tqdm(range(0, M, chunk_size), desc="Alpha field eval", leave=False):
        end = min(start + chunk_size, M)
        q = query_pts[start:end]  # (B,3)

        # d[b,k] = q[b] - pos[k]  →  (B,K,3)
        d = q.unsqueeze(1) - pos.unsqueeze(0)          # (B,K,3)

        # Mahalanobis distance: d^T Cov^{-1} d
        # = d^T (R diag(inv_s2) R^T) d
        # = (R^T d)^T diag(inv_s2) (R^T d)
        # = sum_j inv_s2[j] * (R^T d)[j]^2
        # rot: (K,3,3),  d: (B,K,3)
        # R^T d: (B,K,3) — einsum bki,bkj->bkj where i is row of R^T
        d_rot = torch.einsum("kji,bkj->bki", rot, d)  # (B,K,3)
        maha2 = (d_rot ** 2 * inv_s2.unsqueeze(0)).sum(dim=-1)  # (B,K)

        contrib = op.unsqueeze(0) * torch.exp(-0.5 * maha2)     # (B,K)
        contrib = contrib.clamp(max=1.0 - 1e-7)

        # accumulate log(1 - contrib)
        log_transmittance[start:end] = torch.log(1.0 - contrib + 1e-12).sum(dim=-1)

    alpha = 1.0 - torch.exp(log_transmittance)
    return alpha


# ============================================================================
# POST-PROCESSING
# ============================================================================

def post_process_mesh(mesh, num_clusters: int = 1):
    """Keep the N largest connected components and remove degenerate faces."""
    import trimesh
    components = mesh.split(only_watertight=False)
    if not components:
        return mesh
    components.sort(key=lambda m: m.vertices.shape[0], reverse=True)
    kept = components[:num_clusters]
    if len(kept) == 1:
        result = kept[0]
    else:
        result = trimesh.util.concatenate(kept)
    # remove degenerate triangles
    result = result.process(validate=True)
    return result


# ============================================================================
# MARCHING TETRAHEDRA
# ============================================================================

def run_marching_tetrahedra(
    tet_verts: torch.Tensor,
    tets: torch.Tensor,
    sdf: torch.Tensor,
    alpha_threshold: float = 0.5,
):
    """Wrapper around tetmesh.marching_tetrahedra.

    Args:
        tet_verts: (V,3) float32 vertex positions.
        tets: (T,4) int64 tet connectivity.
        sdf: (V,) SDF values (positive = inside).
        alpha_threshold: isosurface value.

    Returns:
        vertices (np.ndarray), faces (np.ndarray).
    """
    # import local tetmesh.py
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from tetmesh import marching_tetrahedra

    scales_dummy = torch.ones(tet_verts.shape[0], device=tet_verts.device)

    verts_list, scale_list, faces_list, _ = marching_tetrahedra(
        tet_verts[None],          # (1, V, 3)
        tets,                     # (T, 4)
        sdf[None],                # (1, V)
        scales_dummy[None],       # (1, V)
    )

    end_points, end_sdf = verts_list[0]
    faces = faces_list[0].cpu().numpy()

    # interpolate vertex positions along each edge using SDF
    left_pts = end_points[:, 0, :]   # (E,3)
    right_pts = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]      # (E,1)
    right_sdf = end_sdf[:, 1, :]

    # linear interpolation: p = left + t*(right-left) where t = -left/(right-left)
    denom = (right_sdf - left_sdf).clamp(min=1e-8)
    t = (-left_sdf / denom).clamp(0.0, 1.0)
    verts = (left_pts + t * (right_pts - left_pts)).cpu().numpy()

    return verts, faces


# ============================================================================
# TETRAHEDRALIZATION WITH FTETWILD
# ============================================================================

def tetrahedralize(obj_path: str, msh_path: str, stop_quality: float = 10.0) -> bool:
    """Run fTetWild on a surface mesh OBJ to produce a tet mesh MSH."""
    try:
        import wildmeshing
    except ImportError:
        log.warning("wildmeshing not found (pip install wildmeshing). Skipping tet volume.")
        return False

    log.info("Tetrahedralizing %s → %s", obj_path, msh_path)
    tetra = wildmeshing.Tetrahedralizer(stop_quality=stop_quality)
    tetra.load_mesh(obj_path)
    tetra.tetrahedralize()
    tetra.save(msh_path)
    log.info("  Tet mesh saved: %s", msh_path)
    return True


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def extract_tet_mesh(
    ply_path: str,
    output_dir: str,
    *,
    opacity_cutoff: float = 0.05,
    alpha_threshold: float = 0.5,
    jitter_scale: float = 0.5,
    num_jitter: int = 3,
    chunk_size: int = 4096,
    num_clusters: int = 1,
    run_tet: bool = True,
    stop_quality: float = 10.0,
    device: str = "cuda",
    mesh_name: str = "mesh_tets",
) -> None:
    import trimesh
    from scipy.spatial import Delaunay

    os.makedirs(output_dir, exist_ok=True)
    stem = mesh_name

    # ------------------------------------------------------------------
    # 1. Load PLY
    # ------------------------------------------------------------------
    gaussians = load_ply_3dgs(ply_path, device=device)
    positions = gaussians["positions"]
    rotations = gaussians["rotations"]
    scales    = gaussians["scales"]
    opacities = gaussians["opacities"]
    R = build_rotation_matrices(rotations)

    # ------------------------------------------------------------------
    # 2. Build tet vertex set
    # ------------------------------------------------------------------
    log.info("Building tet vertices …")
    tet_verts = build_tet_vertices(
        positions, scales, opacities,
        opacity_cutoff=opacity_cutoff,
        jitter_scale=jitter_scale,
        num_jitter_per_gaussian=num_jitter,
    )

    # ------------------------------------------------------------------
    # 3. Delaunay tetrahedralization
    # ------------------------------------------------------------------
    log.info("Running Delaunay tetrahedralization on %d points …", tet_verts.shape[0])
    tet_verts_np = tet_verts.cpu().numpy()
    delaunay = Delaunay(tet_verts_np)
    tets_np = delaunay.simplices.astype(np.int64)
    log.info("  Generated %d tetrahedra", tets_np.shape[0])
    tets = torch.tensor(tets_np, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # 4. Evaluate alpha field at tet vertices
    # ------------------------------------------------------------------
    log.info("Evaluating Gaussian alpha field at %d tet vertices …", tet_verts.shape[0])
    alpha = evaluate_alpha_field(
        tet_verts, positions, R, scales, opacities,
        chunk_size=chunk_size,
        opacity_cutoff=opacity_cutoff,
    )
    sdf = alpha - alpha_threshold   # positive = inside
    log.info("  Alpha range: [%.3f, %.3f], inside fraction: %.2f%%",
             alpha.min().item(), alpha.max().item(),
             (alpha > alpha_threshold).float().mean().item() * 100)

    # ------------------------------------------------------------------
    # 5. Marching tetrahedra
    # ------------------------------------------------------------------
    log.info("Running marching tetrahedra …")
    verts, faces = run_marching_tetrahedra(tet_verts, tets, sdf, alpha_threshold=alpha_threshold)
    log.info("  Raw mesh: %d verts, %d faces", verts.shape[0], faces.shape[0])

    # ------------------------------------------------------------------
    # 6. Post-process
    # ------------------------------------------------------------------
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh = post_process_mesh(mesh, num_clusters=num_clusters)
    log.info("  Post-processed: %d verts, %d faces", len(mesh.vertices), len(mesh.faces))

    # ------------------------------------------------------------------
    # 7. Save surface mesh
    # ------------------------------------------------------------------
    ply_out  = os.path.join(output_dir, f"{stem}.ply")
    obj_out  = os.path.join(output_dir, f"{stem}.obj")
    mesh.export(ply_out)
    mesh.export(obj_out)
    log.info("Saved: %s", ply_out)
    log.info("Saved: %s", obj_out)

    # ------------------------------------------------------------------
    # 8. Optional fTetWild tet volume
    # ------------------------------------------------------------------
    if run_tet:
        msh_out = os.path.join(output_dir, f"{stem}.msh")
        tetrahedralize(obj_out, msh_out, stop_quality=stop_quality)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ply", help="Input 3DGS PLY file.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("--mesh-name", default="mesh_tets", help="Base name for output files.")
    parser.add_argument("--opacity-cutoff", type=float, default=0.05,
                        help="Discard Gaussians with opacity below this (default: 0.05).")
    parser.add_argument("--alpha-threshold", type=float, default=0.5,
                        help="Isosurface level for marching tetrahedra (default: 0.5).")
    parser.add_argument("--jitter-scale", type=float, default=0.5,
                        help="Jitter magnitude relative to Gaussian scale (default: 0.5).")
    parser.add_argument("--num-jitter", type=int, default=3,
                        help="Extra jittered vertices per Gaussian (default: 3).")
    parser.add_argument("--chunk-size", type=int, default=4096,
                        help="Query points per GPU chunk for alpha evaluation (default: 4096).")
    parser.add_argument("--num-clusters", type=int, default=1,
                        help="Number of connected components to keep (default: 1).")
    parser.add_argument("--no-tet", action="store_true",
                        help="Skip fTetWild tetrahedralization.")
    parser.add_argument("--stop-quality", type=float, default=10.0,
                        help="fTetWild stop quality (default: 10.0).")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of CUDA.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    extract_tet_mesh(
        ply_path=args.ply,
        output_dir=args.output_dir,
        mesh_name=args.mesh_name,
        opacity_cutoff=args.opacity_cutoff,
        alpha_threshold=args.alpha_threshold,
        jitter_scale=args.jitter_scale,
        num_jitter=args.num_jitter,
        chunk_size=args.chunk_size,
        num_clusters=args.num_clusters,
        run_tet=not args.no_tet,
        stop_quality=args.stop_quality,
        device=device,
    )
