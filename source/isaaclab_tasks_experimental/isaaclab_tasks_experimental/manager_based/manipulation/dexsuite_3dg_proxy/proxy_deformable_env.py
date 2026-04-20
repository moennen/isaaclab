# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env wrapper that patches NewtonManager to the extended proxy manager when VBD is enabled."""

from __future__ import annotations

import logging

import isaaclab_newton.physics as _newton_physics

from isaaclab.envs import ManagerBasedRLEnv

from .config.kuka_allegro.physic.newton import Dexsuite3dgProxyNewtonCfg, Dexsuite3dgProxyNewtonManager

logger = logging.getLogger(__name__)


def _is_proxy_newton_physics(cfg) -> bool:
    physics = getattr(getattr(cfg, "sim", None), "physics", None)
    if physics is None:
        return False
    class_type = getattr(physics, "class_type", None)
    if class_type is Dexsuite3dgProxyNewtonManager:
        return True
    if class_type is not None and "Dexsuite3dgProxyNewtonManager" in str(class_type):
        return True
    return False


def _apply_newton_manager_patch(cfg) -> None:
    """Monkey-patch NewtonManager → Dexsuite3dgProxyNewtonManager before scene construction."""
    if not _is_proxy_newton_physics(cfg):
        return
    _newton_physics.NewtonManager = Dexsuite3dgProxyNewtonManager
    import isaaclab_newton.physics.newton_manager as _newton_manager_module
    _newton_manager_module.NewtonManager = Dexsuite3dgProxyNewtonManager
    logger.info("[Proxy VBD] Patched NewtonManager → Dexsuite3dgProxyNewtonManager")


class Dexsuite3dgProxyDeformableEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv that uses Dexsuite3dgProxyNewtonManager for VBD soft body simulation.

    Patches ``isaaclab_newton.physics.NewtonManager`` to the extended proxy manager before
    the simulation context is created so all Newton assets use the correct class.
    """

    def __init__(self, cfg, **kwargs):
        _apply_newton_manager_patch(cfg)
        super().__init__(cfg, **kwargs)
