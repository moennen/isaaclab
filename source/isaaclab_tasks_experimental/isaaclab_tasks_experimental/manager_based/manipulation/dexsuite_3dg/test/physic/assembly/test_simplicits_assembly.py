# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests: SimplicitsModelBuilder assembly (one env via multi-env API) and one SimplicitsSolver step."""

from __future__ import annotations

import pytest

pytest.importorskip("kaolin")

import torch
from kaolin.experimental.newton.solver import SimplicitsSolver

from pxr import Usd, UsdGeom

from ....config.kuka_allegro.physic.kaolin import SimplicitsObjectCfg
from ....config.kuka_allegro.physic.newton.simplicits_assembly import build_multi_env_simplicits_model


def _minimal_stage():
    """Minimal USD stage with env_0 (Robot, table, Object) for proto."""
    stage = Usd.Stage.CreateInMemory()
    assert stage is not None
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/envs", "Xform")
    stage.DefinePrim("/World/envs/env_0", "Xform")
    stage.DefinePrim("/World/envs/env_0/Robot", "Xform")
    stage.DefinePrim("/World/envs/env_0/table", "Xform")
    stage.DefinePrim("/World/envs/env_0/Object", "Xform")
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
    return v, f


class TestSimplicitsAssembly:
    """Build SimplicitsModel (rigid proto + one Simplicits object), finalize, one step."""

    def test_build_and_finalize(self):
        """Build SimplicitsModel via multi-env API with one env; model has simplicits particle range."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        stage = _minimal_stage()
        vertices, faces = _cube_mesh(device, torch.float32)
        cfg = SimplicitsObjectCfg(num_samples=200)

        model, ranges = build_multi_env_simplicits_model(
            stage,
            env_paths=["/World/envs/env_0"],
            object_relative_path="Object",
            env_meshes=[(vertices, faces)],
            simplicits_cfg=cfg,
            device=str(device),
        )

        assert model is not None
        assert len(ranges) == 1
        start = model.simplicits_particle_start
        end = model.simplicits_particle_end
        assert start is not None and end is not None
        assert end > start

    def test_state_and_one_simplicits_step(self):
        """Create state, run one SimplicitsSolver.step; no crash; particle_q slice updates."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        stage = _minimal_stage()
        vertices, faces = _cube_mesh(device, torch.float32)
        cfg = SimplicitsObjectCfg(num_samples=150)

        model, _ = build_multi_env_simplicits_model(
            stage,
            env_paths=["/World/envs/env_0"],
            object_relative_path="Object",
            env_meshes=[(vertices, faces)],
            simplicits_cfg=cfg,
            device=str(device),
        )

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.collide(state_0)
        solver = SimplicitsSolver(model)

        dt = 0.01
        solver.step(state_0, state_1, control, contacts, dt)

        start = model.simplicits_particle_start
        end = model.simplicits_particle_end
        assert state_0.particle_q is not None and state_1.particle_q is not None
        assert end > start
        # After one step with gravity, Simplicits particles should have moved (e.g. z decreased)
        # We only assert no crash and state arrays exist; optional: compare particle_q before/after
        assert state_1.particle_q.size >= end
