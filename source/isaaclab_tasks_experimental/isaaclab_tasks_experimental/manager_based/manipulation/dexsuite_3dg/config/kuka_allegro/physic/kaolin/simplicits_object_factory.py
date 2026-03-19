# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create a rigid Simplicits object from a mesh using Kaolin (surface sampling + create_rigid)."""

from __future__ import annotations

import logging
from typing import Any

import torch

from .simplicits_cfg import SimplicitsObjectCfg

logger = logging.getLogger("dexsuite_3dg.kaolin.factory")

try:
    from kaolin.ops.mesh import sample_points as _kaolin_sample_points
    from kaolin.physics.simplicits import SimplicitsObject
except ImportError:
    _kaolin_sample_points = None  # type: ignore[assignment]
    SimplicitsObject = None  # type: ignore[assignment]

# Minimum radius [m] when computing from mesh to avoid numerical issues.
_DEFAULT_COLLISION_RADIUS_MIN = 1e-4


def compute_collision_particle_radius_from_mesh(
    vertices: torch.Tensor,
    num_samples: int,
    factor: float = 0.5,
) -> float:
    """Compute a collision particle radius from mesh extent and sample count.

    Uses extent / num_samples^(1/3) * factor so that radius scales with object
    size and particle spacing. Used when collision_particle_radius is not set in config.

    Args:
        vertices: Mesh vertices [m], shape (V, 3).
        num_samples: Number of particles (quadrature points).
        factor: Scale factor; 0.5 gives radius ~ half the average spacing.

    Returns:
        Radius [m], at least _DEFAULT_COLLISION_RADIUS_MIN.
    """
    if vertices.dim() == 3:
        vertices = vertices.squeeze(0)
    extent = float((vertices.max(dim=0).values - vertices.min(dim=0).values).max().item())
    if extent <= 0.0 or num_samples < 1:
        return _DEFAULT_COLLISION_RADIUS_MIN
    spacing = extent / (float(num_samples) ** (1.0 / 3.0))
    radius = spacing * factor
    return max(radius, _DEFAULT_COLLISION_RADIUS_MIN)


def create_rigid_simplicits_object_from_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    cfg: SimplicitsObjectCfg,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
) -> Any:
    """Create a rigid Simplicits object (1 handle) from mesh vertices and faces.

    Uses Kaolin's surface sampling (ops.mesh.sample_points) and
    SimplicitsObject.create_rigid. Material and sampling parameters come from cfg;
    shape comes only from the mesh.

    Args:
        vertices: Mesh vertices [m] in **geom local** frame (same convention as
            :func:`~.mesh_from_usd.get_vertices_faces_from_prim_path`), shape
            ``(num_vertices, 3)`` or ``(1, num_vertices, 3)``. When the Kaolin scene
            uses a non-identity ``init_transform``, world pose comes from that matrix,
            not from pre-transforming vertices.
        faces: Triangle indices, shape (num_faces, 3), long dtype.
        cfg: Simplicits object config (density, youngs_modulus, poisson_ratio, num_samples).
        device: Target device (e.g. 'cuda:0'). Inferred from vertices if None.
        dtype: Floating dtype for positions and material tensors.
        verbose: If True, log one-line summary (vertex/face/sample counts, approx_vol).

    Returns:
        A rigid SimplicitsObject (Kaolin) with num_handles == 1.
    """
    if SimplicitsObject is None:
        raise ImportError(
            "Kaolin is required for Simplicits object creation. Install kaolin or set simplicits_enabled=False."
        )

    if device is None:
        device = vertices.device if isinstance(vertices, torch.Tensor) else "cuda:0"
    device = torch.device(device) if isinstance(device, str) else device

    # Ensure batch dimension for Kaolin ops: (1, V, 3)
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)
    if vertices.device != device:
        vertices = vertices.to(device=device, dtype=dtype)
    else:
        vertices = vertices.to(dtype=dtype)
    if faces.device != device:
        faces = faces.to(device=device)
    if faces.dtype != torch.int64:
        faces = faces.long()

    num_samples = max(3, cfg.num_samples)

    if _kaolin_sample_points is None:
        raise ImportError("Kaolin is required for mesh sampling. Install kaolin or set simplicits_enabled=False.")

    points, _ = _kaolin_sample_points(vertices, faces, num_samples)
    pts = points.squeeze(0)

    # Approximate volume from axis-aligned bbox of sampled points
    mn = pts.min(dim=0).values
    mx = pts.max(dim=0).values
    bbox_vol = (mx - mn).clamp(min=1e-9).prod().item()
    appx_vol = max(bbox_vol * 0.5, 1e-6)

    # Material tensors (per-point, from config)
    n = pts.shape[0]
    yms = torch.full((n,), cfg.youngs_modulus, device=device, dtype=dtype)
    prs = torch.full((n,), cfg.poisson_ratio, device=device, dtype=dtype)
    rhos = torch.full((n,), cfg.density, device=device, dtype=dtype)
    appx_vol_t = torch.tensor([appx_vol], device=device, dtype=dtype)

    obj = SimplicitsObject.create_rigid(pts, yms, prs, rhos, appx_vol_t)

    if verbose:
        logger.info(
            "[DexSuite 3DG : Kaolin :] simplicits_from_mesh: verts=%s faces=%s samples=%s approx_vol=%.6f",
            vertices.shape[1],
            faces.shape[0],
            n,
            appx_vol,
        )

    return obj
