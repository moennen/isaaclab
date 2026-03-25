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

    # Newton simulation parameters.
    # These are used by the Newton backend and are ignored by PhysX.

    density: float = 0.02
    """Density [kg/m^2 for cloth, kg/m^3 for volumetric]. Used by Newton backend only."""

    particle_radius: float = 0.008
    """Particle radius [m] (controls rigid body–particle contact distance). Used by Newton backend only."""

    soft_contact_ke: float = 1e4
    """Body–particle contact stiffness [N/m]. Used by Newton backend only."""

    soft_contact_kd: float = 1e-2
    """Body–particle contact damping. Used by Newton backend only."""

    # -- Cloth (triangle surface mesh) parameters

    tri_ke: float = 1e4
    """Triangle area-preserving stiffness [Pa]. Used by Newton backend for cloth meshes."""

    tri_ka: float = 1e4
    """Triangle area stiffness [Pa]. Used by Newton backend for cloth meshes."""

    tri_kd: float = 1.5e-6
    """Triangle area damping. Used by Newton backend for cloth meshes."""

    edge_ke: float = 5.0
    """Bending stiffness. Used by Newton backend for cloth meshes."""

    edge_kd: float = 1e-2
    """Bending damping. Used by Newton backend for cloth meshes."""

    # -- Volumetric (tetrahedral FEM) parameters

    k_mu: float = 1e5
    """First Lame parameter (shear modulus) [Pa]. Used by Newton backend for tet meshes.

    Related to Young's modulus E and Poisson's ratio nu: k_mu = E / (2 * (1 + nu)).
    """

    k_lambda: float = 1e5
    """Second Lame parameter [Pa]. Used by Newton backend for tet meshes.

    Related to Young's modulus E and Poisson's ratio nu: k_lambda = E * nu / ((1 + nu) * (1 - 2 * nu)).
    """

    k_damp: float = 0.0
    """Damping stiffness for tetrahedral elements. Used by Newton backend for tet meshes."""
