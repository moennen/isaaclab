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
