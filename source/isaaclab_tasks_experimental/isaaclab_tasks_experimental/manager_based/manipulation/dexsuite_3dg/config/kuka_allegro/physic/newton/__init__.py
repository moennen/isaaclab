# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics config and extended manager for Dexsuite 3dg."""

from .dexsuite_3dg_builder_utils import (
    build_rigid_proto_excluding_object,
    get_builder_body_articulation_labels,
)
from .dexsuite_3dg_newton_cfg import Dexsuite3dgNewtonCfg
from .dexsuite_3dg_newton_manager import Dexsuite3dgNewtonManager
from .simplicits_assembly import build_single_env_simplicits_model

__all__ = [
    "Dexsuite3dgNewtonCfg",
    "Dexsuite3dgNewtonManager",
    "build_rigid_proto_excluding_object",
    "build_single_env_simplicits_model",
    "get_builder_body_articulation_labels",
]
