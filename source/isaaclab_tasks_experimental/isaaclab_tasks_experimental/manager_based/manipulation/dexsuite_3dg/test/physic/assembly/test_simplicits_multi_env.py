# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Step 4 tests: multi-env SimplicitsModelBuilder assembly and one step."""

from __future__ import annotations

import pytest

pytest.importorskip("kaolin")

import torch
from kaolin.experimental.newton.solver import SimplicitsSolver

from pxr import Usd, UsdGeom

from ....config.kuka_allegro.physic.kaolin import SimplicitsObjectCfg
from ....config.kuka_allegro.physic.newton.simplicits_assembly import (
    build_multi_env_simplicits_model,
)


def _minimal_stage_n_envs(n: int):
    """Minimal USD stage with env_0 .. env_{n-1} (Robot, table, Object) each."""
    stage = Usd.Stage.CreateInMemory()
    assert stage is not None
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/envs", "Xform")
    for i in range(n):
        env_path = f"/World/envs/env_{i}"
        stage.DefinePrim(env_path, "Xform")
        stage.DefinePrim(f"{env_path}/Robot", "Xform")
        stage.DefinePrim(f"{env_path}/table", "Xform")
        stage.DefinePrim(f"{env_path}/Object", "Xform")
    return stage


def _cube_mesh(device: torch.device, dtype: torch.dtype):
    """Unit-cube mesh: 8 vertices, 12 triangles."""
    v = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        device=device,
        dtype=dtype,
    )
    f = torch.tensor(
        [
            [0, 1, 2], [0, 2, 3], [1, 5, 6], [1, 6, 2],
            [5, 4, 7], [5, 7, 6], [4, 0, 3], [4, 3, 7],
            [3, 2, 6], [3, 6, 7], [4, 5, 1], [4, 1, 0],
        ],
        device=device,
        dtype=torch.int64,
    )
    return v, f


class TestSimplicitsMultiEnv:
    """Step 4: multi-env build, finalize, per-env particle ranges, one step."""

    def test_build_n2_finalize_and_per_env_ranges(self):
        """Build for N=2 envs; model finalizes; per_env_particle_ranges has two (start, end)."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        stage = _minimal_stage_n_envs(2)
        vertices, faces = _cube_mesh(device, torch.float32)
        cfg = SimplicitsObjectCfg(num_samples=100)
        env_paths = ["/World/envs/env_0", "/World/envs/env_1"]
        env_meshes = [(vertices, faces)] * 2
        env_transforms = [
            torch.eye(4, device=device, dtype=torch.float32),
            torch.eye(4, device=device, dtype=torch.float32),
        ]
        env_transforms[1][2, 3] = 1.0  # env 1 object 1 m higher

        model, per_env_particle_ranges = build_multi_env_simplicits_model(
            stage=stage,
            env_paths=env_paths,
            object_relative_path="Object",
            env_meshes=env_meshes,
            simplicits_cfg=cfg,
            env_transforms=env_transforms,
            device=str(device),
        )

        assert model is not None
        assert model.simplicits_particle_start is not None and model.simplicits_particle_end is not None
        assert model.simplicits_particle_end > model.simplicits_particle_start
        assert len(per_env_particle_ranges) == 2
        for i, (start, end) in enumerate(per_env_particle_ranges):
            assert start >= model.simplicits_particle_start
            assert end <= model.simplicits_particle_end
            assert end > start
        # Ranges should be contiguous and disjoint
        assert per_env_particle_ranges[0][1] == per_env_particle_ranges[1][0]

    def test_multi_env_one_step(self):
        """Build N=2, create state, run one SimplicitsSolver.step; no crash; state advances."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        stage = _minimal_stage_n_envs(2)
        vertices, faces = _cube_mesh(device, torch.float32)
        cfg = SimplicitsObjectCfg(num_samples=80)
        env_paths = ["/World/envs/env_0", "/World/envs/env_1"]
        env_meshes = [(vertices, faces)] * 2
        env_transforms = [torch.eye(4, device=device, dtype=torch.float32)] * 2

        model, per_env_particle_ranges = build_multi_env_simplicits_model(
            stage=stage,
            env_paths=env_paths,
            object_relative_path="Object",
            env_meshes=env_meshes,
            simplicits_cfg=cfg,
            env_transforms=env_transforms,
            device=str(device),
        )

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.collide(state_0)
        solver = SimplicitsSolver(model)

        solver.step(state_0, state_1, control, contacts, 0.01)

        assert state_1.particle_q.size >= model.simplicits_particle_end
        assert len(per_env_particle_ranges) == 2
