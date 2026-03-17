# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env wrapper that patches NewtonManager to the extended 3dg manager when in Newton mode."""

from __future__ import annotations

import logging

import isaaclab_newton.physics as _newton_physics

from isaaclab.envs import ManagerBasedRLEnv

from .config.kuka_allegro.dexsuite_kuka_allegro_env_cfg import KukaAllegroPhysicsCfg
from .config.kuka_allegro.physic.newton import Dexsuite3dgNewtonCfg, Dexsuite3dgNewtonManager
from .config.kuka_allegro.physic.newton.simplicits_object_adapter import SimplicitsObjectAdapter

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
        "Default is Newton; do not use presets=physx for this task."
    )
    raise ValueError(
        "Dexsuite 3dg tasks require Newton physics (default). "
        "Use e.g. --task=Isaac-Dexsuite-3dg-Kuka-Allegro-Lift-v0 or presets=newton,cube"
    )


def _ensure_newton_visualizer_show_particles(cfg) -> None:
    """Ensure cfg.sim.visualizer_cfgs contains a Newton config with show_particles=True.

    When visualizer_cfgs is None, SimulationContext creates defaults (show_particles=False).
    We set or update the list so the context uses our Newton config with show_particles=True.
    """
    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    vcfgs = getattr(cfg.sim, "visualizer_cfgs", None)
    vlist = (vcfgs if isinstance(vcfgs, list) else [vcfgs]) if vcfgs is not None else []
    newton_cfg = next(
        (v for v in vlist if getattr(v, "visualizer_type", None) == "newton"),
        None,
    )
    if newton_cfg is not None:
        if hasattr(newton_cfg, "show_particles"):
            newton_cfg.show_particles = True
        return
    cfg.sim.visualizer_cfgs = vlist + [NewtonVisualizerCfg(show_particles=True)]


def _scene_uses_simplicits_adapter(cfg) -> bool:
    """True if the scene config uses SimplicitsObjectAdapter for the object."""
    scene = getattr(cfg, "scene", None)
    if scene is None:
        return False
    if scene == "simplicits":
        return True
    object_cfg = getattr(scene, "object", None)
    if object_cfg is None:
        return False
    object_class = getattr(object_cfg, "class_type", None)
    return object_class is SimplicitsObjectAdapter or (
        isinstance(object_class, str) and "SimplicitsObjectAdapter" in object_class
    )


def _apply_newton_manager_patch_for_3dg(cfg) -> None:
    """If physics is our extended Newton manager, patch the module so assets see it.

    Patch both isaaclab_newton.physics and isaaclab_newton.physics.newton_manager so that
    all consumers (assets, sensors, env events like randomize_physics_scene_gravity) use
    Dexsuite3dgNewtonManager and thus see the initialized model.
    When physics is the simplicits preset, require the scene to use the simplicits preset
    too (object = SimplicitsObjectAdapter), otherwise raise a clear error.
    """
    if not _is_newton_3dg_physics(cfg):
        return
    sim_physics = getattr(getattr(cfg, "sim", None), "physics", None)
    if (
        isinstance(sim_physics, Dexsuite3dgNewtonCfg)
        and getattr(sim_physics, "simplicits_enabled", False)
        and getattr(sim_physics, "simplicits_cfg", None) is not None
    ):
        if not _scene_uses_simplicits_adapter(cfg):
            raise ValueError(
                "Simplicits physics (env.sim.physics=simplicits) requires the simplicits scene "
                "preset so the object uses the pose adapter. Add: env.scene=simplicits"
            )
    _newton_physics.NewtonManager = Dexsuite3dgNewtonManager
    import isaaclab_newton.physics.newton_manager as _newton_manager_module

    _newton_manager_module.NewtonManager = Dexsuite3dgNewtonManager


class Dexsuite3dgManagerBasedRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv that uses :class:`Dexsuite3dgNewtonManager` when physics is Newton.

    Patches ``isaaclab_newton.physics.NewtonManager`` to the extended manager before
    the simulation context is created, so all Newton assets use the same class.
    Default physics is Newton; PhysX is not supported for this task.
    """

    def __init__(self, cfg, **kwargs):
        # When scene uses the simplicits adapter, force sim.physics to the simplicits preset so
        # SimulationContext gets Dexsuite3dgNewtonManager (required for reset() to run the extended manager).
        # Use the task's KukaAllegroPhysicsCfg.simplicits so we get the right config even when
        # a global preset (e.g. presets=cube) replaced physics with base NewtonCfg.
        if _scene_uses_simplicits_adapter(cfg):
            physics = getattr(getattr(cfg, "sim", None), "physics", None)
            simplicits_preset = getattr(physics, "simplicits", None) if physics is not None else None
            if simplicits_preset is None:
                simplicits_preset = getattr(KukaAllegroPhysicsCfg(), "simplicits", None)
            if simplicits_preset is not None:
                cfg.sim.physics = simplicits_preset
            # Ensure a Newton visualizer config with show_particles=True exists so SimulationContext
            # uses it instead of creating a default (which has show_particles=False). When
            # visualizer_cfgs is None, the context creates configs via _create_default_visualizer_configs.
            _ensure_newton_visualizer_show_particles(cfg)
        _require_newton_3dg(cfg)
        _apply_newton_manager_patch_for_3dg(cfg)
        super().__init__(cfg, **kwargs)
