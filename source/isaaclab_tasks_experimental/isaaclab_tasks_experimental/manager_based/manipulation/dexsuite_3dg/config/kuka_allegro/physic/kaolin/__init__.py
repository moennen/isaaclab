# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Kaolin Simplicits config and factory (mesh → rigid Simplicits object)."""

from .simplicits_cfg import SimplicitsObjectCfg
from .simplicits_object_factory import create_rigid_simplicits_object_from_mesh

__all__ = ["SimplicitsObjectCfg", "create_rigid_simplicits_object_from_mesh"]
