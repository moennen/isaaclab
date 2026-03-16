# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Step 5 tests: manager builder injection and solver wiring (simplicits on/off)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("kaolin")

import torch
from isaaclab_newton.physics import NewtonManager

from pxr import Sdf, Usd, UsdGeom

from isaaclab.physics import PhysicsManager

from ....config.kuka_allegro.physic.kaolin import SimplicitsObjectCfg
from ....config.kuka_allegro.physic.mesh_from_usd import get_vertices_faces_from_prim_path
from ....config.kuka_allegro.physic.newton.dexsuite_3dg_newton_cfg import Dexsuite3dgNewtonCfg
from ....config.kuka_allegro.physic.newton.dexsuite_3dg_newton_manager import (
    Dexsuite3dgNewtonManager,
    _discover_env_paths_and_xforms,
    _simplicits_enabled,
)
from ....config.kuka_allegro.physic.newton.simplicits_assembly import (
    build_multi_env_simplicits_model,
)


def _minimal_stage_one_env_with_cube():
    """Minimal USD stage: /World/envs/env_0 with Object/Box (Cube, size 1)."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/envs", "Xform")
    stage.DefinePrim("/World/envs/env_0", "Xform")
    stage.DefinePrim("/World/envs/env_0/Robot", "Xform")
    stage.DefinePrim("/World/envs/env_0/table", "Xform")
    stage.DefinePrim("/World/envs/env_0/Object", "Xform")
    box_prim = stage.DefinePrim("/World/envs/env_0/Object/Box", "Cube")
    UsdGeom.Cube(box_prim).GetSizeAttr().Set(1.0)
    env_prim = stage.GetPrimAtPath("/World/envs/env_0")
    env_prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Float3).Set((0.0, 0.0, 0.0))
    return stage


class TestManagerSimplicits:
    """Step 5: simplicits_enabled flag, discover env paths, start_simulation builds model."""

    def test_simplicits_enabled_requires_cfg(self):
        """_simplicits_enabled is False when cfg is None or simplicits_cfg is None."""
        with patch.object(PhysicsManager, "_cfg", None):
            assert _simplicits_enabled() is False
        with patch.object(
            PhysicsManager,
            "_cfg",
            Dexsuite3dgNewtonCfg(simplicits_enabled=True, simplicits_cfg=None),
        ):
            assert _simplicits_enabled() is False
        with patch.object(
            PhysicsManager,
            "_cfg",
            Dexsuite3dgNewtonCfg(simplicits_enabled=False, simplicits_cfg=SimplicitsObjectCfg()),
        ):
            assert _simplicits_enabled() is False

    def test_simplicits_enabled_true_when_both_set(self):
        """_simplicits_enabled is True when simplicits_enabled and simplicits_cfg are set."""
        with patch.object(
            PhysicsManager,
            "_cfg",
            Dexsuite3dgNewtonCfg(simplicits_enabled=True, simplicits_cfg=SimplicitsObjectCfg()),
        ):
            assert _simplicits_enabled() is True

    def test_discover_env_paths_and_xforms(self):
        """_discover_env_paths_and_xforms returns env paths and xforms from stage."""
        stage = _minimal_stage_one_env_with_cube()
        env_paths, xforms = _discover_env_paths_and_xforms(stage)
        assert len(env_paths) == 1
        assert env_paths[0] == "/World/envs/env_0"
        assert len(xforms) == 1
        pos, quat = xforms[0]
        assert pos == (0.0, 0.0, 0.0)
        assert quat == (0.0, 0.0, 0.0, 1.0)

    def test_simplicits_build_path_produces_model(self):
        """Manager build path: stage + mesh-from-USD + build_multi_env produces Simplicits model.

        Does not call start_simulation (avoids dispatch_event / Fabric which can block in tests).
        """
        stage = _minimal_stage_one_env_with_cube()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        env_paths, env_proto_xforms = _discover_env_paths_and_xforms(stage)
        assert len(env_paths) == 1
        env_meshes = []
        for env_path in env_paths:
            vertices, faces = get_vertices_faces_from_prim_path(
                stage, f"{env_path}/Object", device=device, dtype=torch.float32
            )
            env_meshes.append((vertices, faces))
        simplicits_cfg = SimplicitsObjectCfg(num_samples=50)
        env_transforms = [
            torch.eye(4, device=env_meshes[0][0].device, dtype=torch.float32),
        ]
        model, per_env_ranges = build_multi_env_simplicits_model(
            stage=stage,
            env_paths=env_paths,
            object_relative_path="Object",
            env_meshes=env_meshes,
            simplicits_cfg=simplicits_cfg,
            env_transforms=env_transforms,
            env_proto_xforms=env_proto_xforms,
            device=device,
            up_axis="Z",
            solver_type="mujoco_warp",
        )
        assert model is not None
        assert model.simplicits_particle_end > model.simplicits_particle_start
        assert len(per_env_ranges) == 1
        assert per_env_ranges[0][1] > per_env_ranges[0][0]

    def test_start_simulation_simplicits_path_wired(self):
        """_start_simulation_simplicits() is exercised; super().start_simulation() mocked to avoid blocking.

        Ensures the manager code path (branch, mesh load, build_multi_env, wrapper, set _builder,
        call super) runs and leaves _model and _per_env_particle_ranges set. The base
        start_simulation is stubbed so we do not run event dispatch or Fabric.
        """
        stage = _minimal_stage_one_env_with_cube()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cfg = Dexsuite3dgNewtonCfg(
            simplicits_enabled=True,
            simplicits_cfg=SimplicitsObjectCfg(num_samples=50),
        )

        def stub_start_simulation_impl(cls):
            """Minimal base start_simulation: finalize builder, set state/control. No events, no Fabric.
            Uses cls (the manager class that has _builder set) so we see Dexsuite3dgNewtonManager._builder.
            """
            cls._model = cls._builder.finalize(device=device)
            cls._state_0 = cls._model.state()
            cls._state_1 = cls._model.state()
            cls._control = cls._model.control()
            cls._model.num_envs = cls._num_envs

        stub_start_simulation = classmethod(stub_start_simulation_impl)

        manager_module = (
            "isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_3dg."
            "config.kuka_allegro.physic.newton.dexsuite_3dg_newton_manager"
        )
        with patch(f"{manager_module}.get_current_stage", return_value=stage):
            with patch.object(PhysicsManager, "_cfg", cfg):
                with patch.object(PhysicsManager, "_device", device):
                    with patch.object(
                        Dexsuite3dgNewtonManager,
                        "_up_axis",
                        "Z",
                    ):
                        with patch.object(
                            Dexsuite3dgNewtonManager,
                            "_gravity_vector",
                            (0.0, 0.0, -9.81),
                        ):
                            with patch.object(
                                NewtonManager,
                                "start_simulation",
                                stub_start_simulation,
                            ):
                                Dexsuite3dgNewtonManager.start_simulation()

        assert Dexsuite3dgNewtonManager._model is not None
        model = Dexsuite3dgNewtonManager._model
        assert model.simplicits_particle_start is not None
        assert model.simplicits_particle_end is not None
        assert model.simplicits_particle_end > model.simplicits_particle_start
        assert Dexsuite3dgNewtonManager._per_env_particle_ranges is not None
        assert len(Dexsuite3dgNewtonManager._per_env_particle_ranges) == 1
        assert Dexsuite3dgNewtonManager._num_envs == 1
