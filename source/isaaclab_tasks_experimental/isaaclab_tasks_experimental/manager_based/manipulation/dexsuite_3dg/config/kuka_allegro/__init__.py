# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Dextra Kuka Allegro environments.

Uses :class:`Dexsuite3dgManagerBasedRLEnv` so the extended Newton manager is patched
when running with default (Newton) physics.
"""

import gymnasium as gym

from . import agents

# Env class that patches NewtonManager to Dexsuite3dgNewtonManager (from .physic.newton)
_DEXSUITE_3DG_ENV = f"{__name__.rsplit('.config', 1)[0]}.dexsuite_3dg_env:Dexsuite3dgManagerBasedRLEnv"

##
# Register Gym environments.
##

# State Observation
gym.register(
    id="Isaac-Dexsuite-3dg-Kuka-Allegro-Reorient-v0",
    entry_point=_DEXSUITE_3DG_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:Dexsuite3dgKukaAllegroReorientEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Dexsuite3dgKukaAllegroPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexsuite-3dg-Kuka-Allegro-Reorient-Play-v0",
    entry_point=_DEXSUITE_3DG_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:Dexsuite3dgKukaAllegroReorientEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Dexsuite3dgKukaAllegroPPORunnerCfg",
    },
)

# Dexsuite 3dg Lift Environments
gym.register(
    id="Isaac-Dexsuite-3dg-Kuka-Allegro-Lift-v0",
    entry_point=_DEXSUITE_3DG_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:Dexsuite3dgKukaAllegroLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Dexsuite3dgKukaAllegroPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Dexsuite-3dg-Kuka-Allegro-Lift-Play-v0",
    entry_point=_DEXSUITE_3DG_ENV,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:Dexsuite3dgKukaAllegroLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Dexsuite3dgKukaAllegroPPORunnerCfg",
    },
)
