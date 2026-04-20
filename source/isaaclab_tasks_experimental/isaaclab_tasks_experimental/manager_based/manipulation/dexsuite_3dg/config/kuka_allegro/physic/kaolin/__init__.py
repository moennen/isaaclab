# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kaolin Simplicits config and factory (mesh → rigid Simplicits object)."""

from .simplicits_cfg import SimplicitsObjectCfg
from .simplicits_object_factory import (
    compute_collision_particle_radius_from_mesh,
    create_rigid_simplicits_object_from_mesh,
)

__all__ = [
    "SimplicitsObjectCfg",
    "compute_collision_particle_radius_from_mesh",
    "create_rigid_simplicits_object_from_mesh",
]
