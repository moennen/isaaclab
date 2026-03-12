# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env wrapper that patches NewtonManager to the extended 3dg manager when in Newton mode."""

from __future__ import annotations

import logging

import isaaclab_newton.physics as _newton_physics

from isaaclab.envs import ManagerBasedRLEnv

from .config.kuka_allegro.physic.newton import Dexsuite3dgNewtonManager

logger = logging.getLogger(__name__)


def _is_newton_3dg_physics(cfg) -> bool:
    """Return True if the config uses the 3dg Newton physics (Dexsuite3dgNewtonManager)."""
    physics = getattr(getattr(cfg, "sim", None), "physics", None)
    if physics is None:
        return False
    class_type = getattr(physics, "class_type", None)
    if class_type is Dexsuite3dgNewtonManager:
        return True
    if class_type is not None and "Dexsuite3dgNewtonManager" in str(class_type):
        return True
    return False


def _require_newton_3dg(cfg) -> None:
    """If physics is not Newton 3dg, log an error and raise. Call before creating the sim."""
    if _is_newton_3dg_physics(cfg):
        return
    logger.error(
        "Dexsuite 3dg tasks (Isaac-Dexsuite-3dg-Kuka-Allegro-*) require Newton physics. "
        "Use presets=newton (e.g. presets=newton,cube). PhysX is not supported for this task."
    )
    raise ValueError(
        "Dexsuite 3dg tasks require presets=newton. "
        "Use e.g. --task=Isaac-Dexsuite-3dg-Kuka-Allegro-Lift-v0 presets=newton,cube"
    )


def _apply_newton_manager_patch_for_3dg(cfg) -> None:
    """If physics is our extended Newton manager, patch the module so assets see it."""
    if not _is_newton_3dg_physics(cfg):
        return
    _newton_physics.NewtonManager = Dexsuite3dgNewtonManager


class Dexsuite3dgManagerBasedRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv that uses :class:`Dexsuite3dgNewtonManager` when physics is Newton.

    Patches ``isaaclab_newton.physics.NewtonManager`` to the extended manager before
    the simulation context is created, so all Newton assets use the same class.
    Raises if run with PhysX; use presets=newton for this task.
    """

    def __init__(self, cfg, **kwargs):
        _require_newton_3dg(cfg)
        _apply_newton_manager_patch_for_3dg(cfg)
        super().__init__(cfg, **kwargs)
