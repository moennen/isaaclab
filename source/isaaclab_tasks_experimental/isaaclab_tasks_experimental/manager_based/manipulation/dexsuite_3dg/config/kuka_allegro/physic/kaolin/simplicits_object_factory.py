# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Create a rigid Simplicits object from a mesh using Kaolin (surface sampling + create_rigid)."""

from __future__ import annotations

import logging
from typing import Any

import torch

from .simplicits_cfg import SimplicitsObjectCfg

logger = logging.getLogger("dexsuite_3dg.kaolin.factory")


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
        vertices: Mesh vertices, shape (num_vertices, 3) or (1, num_vertices, 3).
        faces: Triangle indices, shape (num_faces, 3), long dtype.
        cfg: Simplicits object config (density, youngs_modulus, poisson_ratio, num_samples).
        device: Target device (e.g. 'cuda:0'). Inferred from vertices if None.
        dtype: Floating dtype for positions and material tensors.
        verbose: If True, log one-line summary (vertex/face/sample counts, approx_vol).

    Returns:
        A rigid SimplicitsObject (Kaolin) with num_handles == 1.
    """
    try:
        from kaolin.physics.simplicits import SimplicitsObject
    except ImportError as e:
        raise ImportError(
            "Kaolin is required for Simplicits object creation. Install kaolin or set simplicits_enabled=False."
        ) from e

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

    # Surface sampling via Kaolin
    try:
        import kaolin.ops.mesh as ko_mesh
    except ImportError as e:
        raise ImportError(
            "Kaolin is required for mesh sampling. Install kaolin or set simplicits_enabled=False."
        ) from e

    points, _ = ko_mesh.sample_points(vertices, faces, num_samples)
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
            "simplicits_from_mesh: verts=%s faces=%s samples=%s approx_vol=%.6f",
            vertices.shape[1],
            faces.shape[0],
            n,
            appx_vol,
        )

    return obj
