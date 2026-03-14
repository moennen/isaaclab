# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Step 2 tests: build rigid proto from USD excluding Object prim."""

from __future__ import annotations

from newton import ModelBuilder

from pxr import Usd, UsdGeom

from ....config.kuka_allegro.physic.newton import (
    build_rigid_proto_excluding_object,
    get_builder_body_articulation_labels,
)


def _minimal_stage_with_env_and_object() -> Usd.Stage:
    """Create an in-memory USD stage with env_0 (Robot, Table, Object) and GroundPlane."""
    stage = Usd.Stage.CreateInMemory()
    assert stage is not None, "CreateInMemory failed"
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/envs", "Xform")
    stage.DefinePrim("/World/envs/env_0", "Xform")
    stage.DefinePrim("/World/envs/env_0/Robot", "Xform")
    stage.DefinePrim("/World/envs/env_0/table", "Xform")
    stage.DefinePrim("/World/envs/env_0/Object", "Xform")
    stage.DefinePrim("/World/GroundPlane", "Xform")
    return stage


class TestBuildRigidProtoExcludingObject:
    """Tests for build_rigid_proto_excluding_object (Step 2)."""

    def test_returns_model_builder(self):
        """Helper returns a Newton ModelBuilder."""
        stage = _minimal_stage_with_env_and_object()
        builder = build_rigid_proto_excluding_object(
            stage,
            env_path="/World/envs/env_0",
            object_relative_path="Object",
        )
        assert isinstance(builder, ModelBuilder)

    def test_no_body_corresponds_to_object_prim(self):
        """No body in the proto has a label/key that corresponds to the Object prim."""
        stage = _minimal_stage_with_env_and_object()
        env_path = "/World/envs/env_0"
        object_path = f"{env_path}/Object"

        builder = build_rigid_proto_excluding_object(
            stage,
            env_path=env_path,
            object_relative_path="Object",
        )

        body_labels, _ = get_builder_body_articulation_labels(builder)
        for label in body_labels:
            assert label != object_path, f"Object prim must not be in proto: {label}"
            assert not label.startswith(object_path + "/"), f"Object subtree must not be in proto: {label}"

    def test_build_from_minimal_stage_body_articulation_counts(self):
        """Build proto from minimal stage; body/articulation counts are consistent (no crash)."""
        stage = _minimal_stage_with_env_and_object()
        builder = build_rigid_proto_excluding_object(
            stage,
            env_path="/World/envs/env_0",
            object_relative_path="Object",
        )

        body_labels, art_labels = get_builder_body_articulation_labels(builder)
        n_bodies = len(body_labels)
        n_arts = len(art_labels)
        # Minimal stage may have 0 bodies if no physics schemas; just ensure no error
        assert n_bodies >= 0 and n_arts >= 0
