# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Simplicits particle rigid teleport math (no Kaolin / GPU build)."""

from __future__ import annotations

import torch

from ....config.kuka_allegro.physic.mesh_from_usd import transform_points_mat4


def test_transform_points_mat4_translation():
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 1.0]])
    t = torch.eye(4, dtype=torch.float32)
    t[:3, 3] = torch.tensor([10.0, -1.0, 3.0])
    out = transform_points_mat4(pts, t)
    assert torch.allclose(out, pts + torch.tensor([10.0, -1.0, 3.0]))


def test_transform_points_mat4_rigid_teleport_chain():
    """p_new = T_reset @ T_build^{-1} @ p_build when object moves rigidly in world."""
    t_build = torch.eye(4, dtype=torch.float32)
    t_build[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    t_build_inv = torch.linalg.inv(t_build)
    pts_build = torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]], dtype=torch.float32)
    t_reset = torch.eye(4, dtype=torch.float32)
    t_reset[:3, 3] = torch.tensor([100.0, 0.0, 0.0])
    t_delta = t_reset @ t_build_inv
    out = transform_points_mat4(pts_build, t_delta)
    assert torch.allclose(out, torch.tensor([[100.0, 0.0, 0.0], [101.0, 0.0, 0.0]]), atol=1e-5)


def test_transform_points_mat4_scale_translation():
    """Affine with non-uniform scale matches v @ R.T + t for diagonal upper 3x3."""
    t = torch.eye(4, dtype=torch.float32)
    t[:3, :3] = torch.diag(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32))
    t[:3, 3] = torch.tensor([0.5, -0.25, 1.0], dtype=torch.float32)
    v = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=torch.float32)
    vw = transform_points_mat4(v, t)
    a = t[:3, :3]
    trans = t[:3, 3]
    for i in range(v.shape[0]):
        assert torch.allclose(vw[i], v[i] @ a.T + trans, atol=1e-5)
