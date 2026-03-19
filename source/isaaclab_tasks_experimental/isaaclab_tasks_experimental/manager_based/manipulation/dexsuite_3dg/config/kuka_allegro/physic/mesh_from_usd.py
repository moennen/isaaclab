# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Get mesh vertices and faces from a USD prim for Simplicits (manager/build).

Uses :func:`~isaaclab.utils.mesh.create_trimesh_from_geom_mesh` and
:func:`~isaaclab.utils.mesh.create_trimesh_from_geom_shape` from IsaacLab; this module
only adds prim path resolution (first Mesh or primitive under path) and torch tensor output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.utils.mesh import (
    PRIMITIVE_MESH_TYPES,
    create_trimesh_from_geom_mesh,
    create_trimesh_from_geom_shape,
)


def _geom_prims_under_path(stage: Any, prim_path: str) -> list[Any]:
    """First Mesh or primitive geom under prim_path (children or prim itself)."""
    types = ["Mesh"] + list(PRIMITIVE_MESH_TYPES)
    prims = sim_utils.get_all_matching_child_prims(
        prim_path,
        predicate=lambda p: p.GetTypeName() in types,
        stage=stage,
    )
    if prims:
        return prims
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid() and prim.GetTypeName() in types:
        return [prim]
    return []


def get_vertices_faces_from_prim_path(
    stage: Any,
    prim_path: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (vertices [V, 3], faces [F, 3]) from the first supported geom under prim_path.

    Uses IsaacLab :func:`~isaaclab.utils.mesh.create_trimesh_from_geom_mesh` (Mesh) and
    :func:`~isaaclab.utils.mesh.create_trimesh_from_geom_shape` (Cube, Sphere, etc.).
    If the prim at prim_path is an Xform, the first child with a supported type is used.

    Args:
        stage: USD stage.
        prim_path: Path to the object prim (e.g. /World/envs/env_0/Object).
        device: Device for returned tensors.
        dtype: Dtype for vertices; faces are int64.

    Returns:
        vertices: Shape (V, 3) [m] in prim local frame.
        faces: Shape (F, 3) vertex indices.

    Raises:
        KeyError: If no supported geometry is found under prim_path.
    """
    prims = _geom_prims_under_path(stage, prim_path)
    if not prims:
        raise KeyError(f"No supported geometry under {prim_path}")
    prim = prims[0]
    if prim.GetTypeName() == "Mesh":
        tm = create_trimesh_from_geom_mesh(prim)
    else:
        tm = create_trimesh_from_geom_shape(prim)
    vertices = torch.from_numpy(np.asarray(tm.vertices, dtype=np.float32)).to(device=device, dtype=dtype)
    faces = torch.from_numpy(np.asarray(tm.faces, dtype=np.int64)).to(device=device)
    return vertices, faces


def get_geom_world_transform_4x4(
    stage: Any,
    prim_path: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the mesh (geom) prim local-to-world 4x4 matrix as a torch tensor.

    Uses the same geom prim resolution as :func:`get_vertices_faces_from_prim_path`
    (first Mesh or primitive under ``prim_path``). Vertices from that path are in
    this prim's local frame; multiplying by this matrix maps them to world space.

    Args:
        stage: USD stage.
        prim_path: Path to the object root (e.g. ``.../env_0/Object``).
        device: Device for the returned tensor.
        dtype: Floating dtype for the matrix.

    Returns:
        (4, 4) tensor [m/m], row layout consistent with ``p_w = M[:3,:3] @ p_l + M[:3,3]``
        (column vectors) / equivalently world rows ``p_w_row = p_l_row @ M[:3,:3].T + M[:3,3]``.

    Raises:
        KeyError: If no supported geometry is found under ``prim_path``.
    """
    prims = _geom_prims_under_path(stage, prim_path)
    if not prims:
        raise KeyError(f"No supported geometry under {prim_path}")
    prim = prims[0]
    xformable = UsdGeom.Xformable(prim)
    m = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    mat = np.array(m, dtype=np.float64)
    return torch.tensor(mat, device=device, dtype=dtype)


def transform_points_mat4(pts: torch.Tensor, mat4: torch.Tensor) -> torch.Tensor:
    """Map 3D points by a 4x4 transform (column-vector convention ``p'_h = mat4 @ p_h``).

    For affine ``mat4`` with last row ``[0,0,0,1]``, this equals
    ``pts @ mat4[:3,:3].T + mat4[:3,3]``.

    Args:
        pts: (N, 3) [m].
        mat4: (4, 4).

    Returns:
        (N, 3) transformed [m].
    """
    if pts.dim() != 2 or pts.shape[-1] != 3:
        raise ValueError("pts must have shape (N, 3)")
    n = pts.shape[0]
    ones = torch.ones(n, 1, device=pts.device, dtype=pts.dtype)
    ph = torch.cat([pts, ones], dim=1)
    m = mat4.to(device=pts.device, dtype=pts.dtype)
    return (ph @ m.T)[:, :3]
