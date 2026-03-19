# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simplicits object parameters (material, sampling) for mesh → rigid Simplicits."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class SimplicitsObjectCfg:
    """Config for creating a rigid Simplicits object from a mesh.

    All parameters used by the Kaolin factory (material, sampling) live here
    so they sit alongside other task parameters; no hardcoded defaults in code.
    """

    density: float = 500.0
    """Density [kg/m³] applied to all sampled points."""

    youngs_modulus: float = 1e5
    """Young's modulus [Pa] for material stiffness."""

    poisson_ratio: float = 0.45
    """Poisson ratio (dimensionless)."""

    num_samples: int = 1000
    """Number of points to sample from the mesh surface (Kaolin surface sampling)."""

    collision_particle_radius: float | None = None
    """Radius [m] for collision (scene-wide and per-particle). If None, computed from mesh extent and num_samples."""
