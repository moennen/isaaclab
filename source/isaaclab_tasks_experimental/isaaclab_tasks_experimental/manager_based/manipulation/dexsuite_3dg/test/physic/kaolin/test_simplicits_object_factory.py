# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Step 1 tests: mesh → rigid SimplicitsObject via Kaolin factory."""

from __future__ import annotations

import pytest

pytest.importorskip("kaolin")

import torch
from kaolin.physics.simplicits import SimplicitsObject

from ....config.kuka_allegro.physic.kaolin import (
    SimplicitsObjectCfg,
    compute_collision_particle_radius_from_mesh,
    create_rigid_simplicits_object_from_mesh,
)


def _cube_mesh(device: torch.device, dtype: torch.dtype):
    """Minimal unit-cube mesh: 8 vertices, 12 triangles."""
    vertices = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        device=device,
        dtype=dtype,
    )
    # 12 triangles (two per face)
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [1, 5, 6],
            [1, 6, 2],
            [5, 4, 7],
            [5, 7, 6],
            [4, 0, 3],
            [4, 3, 7],
            [3, 2, 6],
            [3, 6, 7],
            [4, 5, 1],
            [4, 1, 0],
        ],
        device=device,
        dtype=torch.int64,
    )
    return vertices, faces


class TestSimplicitsObjectFactory:
    """Tests for create_rigid_simplicits_object_from_mesh (Step 1)."""

    def test_returns_rigid_simplicits_object(self):
        """Factory returns a SimplicitsObject with num_handles == 1."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        vertices, faces = _cube_mesh(device, torch.float32)
        cfg = SimplicitsObjectCfg(num_samples=500)

        obj = create_rigid_simplicits_object_from_mesh(vertices, faces, cfg, device=device)

        assert isinstance(obj, SimplicitsObject)
        assert obj.num_handles == 1

    def test_mesh_from_synthetic_vertices_faces(self):
        """Mesh in (vertices, faces) produces valid rigid object."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        vertices, faces = _cube_mesh(device, torch.float32)
        cfg = SimplicitsObjectCfg(num_samples=200, density=400.0)

        obj = create_rigid_simplicits_object_from_mesh(vertices, faces, cfg, device=device)

        assert obj.num_handles == 1
        assert obj.pts.shape[0] == cfg.num_samples
        assert obj.pts.shape[1] == 3

    def test_device_handling_cpu(self):
        """Factory works on CPU when device='cpu'."""
        device = torch.device("cpu")
        vertices, faces = _cube_mesh(device, torch.float32)
        cfg = SimplicitsObjectCfg(num_samples=100)

        obj = create_rigid_simplicits_object_from_mesh(vertices, faces, cfg, device=device)

        assert obj.num_handles == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_handling_cuda(self):
        """Factory works on CUDA when device='cuda:0'."""
        device = torch.device("cuda:0")
        vertices, faces = _cube_mesh(device, torch.float32)
        cfg = SimplicitsObjectCfg(num_samples=100)

        obj = create_rigid_simplicits_object_from_mesh(vertices, faces, cfg, device=device)

        assert obj.num_handles == 1
        assert obj.pts.is_cuda

    def test_compute_collision_radius_from_mesh(self):
        """Computed radius is positive and scales with extent and num_samples."""
        v, _ = _cube_mesh(torch.device("cpu"), torch.float32)
        r100 = compute_collision_particle_radius_from_mesh(v, 100)
        r1000 = compute_collision_particle_radius_from_mesh(v, 1000)
        assert r100 > 0 and r1000 > 0
        # More samples -> smaller spacing -> smaller radius
        assert r1000 < r100
        # Scale mesh: extent 2 -> radius should be ~2x extent 1 (unit cube)
        v2 = v * 2.0
        r100_scaled = compute_collision_particle_radius_from_mesh(v2, 100)
        assert r100_scaled > r100

    def test_config_drives_material_and_samples(self):
        """Changing cfg (density, num_samples) is reflected in the object."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        vertices, faces = _cube_mesh(device, torch.float32)
        cfg = SimplicitsObjectCfg(num_samples=300, density=600.0)

        obj = create_rigid_simplicits_object_from_mesh(vertices, faces, cfg, device=device)

        assert obj.num_handles == 1
        assert obj.pts.shape[0] == 300
