# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import DEFORMABLE_TARGET_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .deformable_object import DeformableObject


@configclass
class DeformableObjectCfg(AssetBaseCfg):
    """Configuration parameters for a deformable object."""

    class_type: type[DeformableObject] | str = "{DIR}.deformable_object:DeformableObject"

    visualizer_cfg: VisualizationMarkersCfg = DEFORMABLE_TARGET_MARKER_CFG.replace(
        prim_path="/Visuals/DeformableTarget"
    )
    """The configuration object for the visualization markers. Defaults to DEFORMABLE_TARGET_MARKER_CFG.

    .. note::
        This attribute is only used when debug visualization is enabled.
    """

    # Newton cloth simulation parameters.
    # These are used by the Newton backend for cloth mesh creation and are ignored by PhysX.

    density: float = 0.02
    """Cloth density [kg/m^2]. Used by Newton backend only."""

    tri_ke: float = 1e4
    """Triangle area-preserving stiffness [Pa]. Used by Newton backend only."""

    tri_ka: float = 1e4
    """Triangle area stiffness [Pa]. Used by Newton backend only."""

    tri_kd: float = 1.5e-6
    """Triangle area damping. Used by Newton backend only."""

    edge_ke: float = 5.0
    """Bending stiffness. Used by Newton backend only."""

    edge_kd: float = 1e-2
    """Bending damping. Used by Newton backend only."""

    particle_radius: float = 0.008
    """Particle radius [m]. Used by Newton backend only."""

    mesh_usd_path: str | None = None
    """Path to an external USD file containing the mesh geometry.
    When set, the Newton backend reads the mesh from this file instead of from the stage prim.
    Used by Newton backend only."""

    mesh_prim_path: str | None = None
    """Prim path within the external USD file to read the mesh from (e.g. ``/root/shirt``).
    Required when :attr:`mesh_usd_path` is set. Used by Newton backend only."""

    soft_contact_ke: float = 1e4
    """Body-particle contact stiffness. Used by Newton backend only."""

    soft_contact_kd: float = 1e-2
    """Body-particle contact damping. Used by Newton backend only."""
