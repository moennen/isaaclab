# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Dextra Kuka Allegro environments.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# State Observation
gym.register(
    id="Isaac-Dexsuite-3dg-Proxy-Kuka-Allegro-Reorient-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:Dexsuite3dgProxyKukaAllegroReorientEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Dexsuite3dgProxyKukaAllegroPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexsuite-3dg-Proxy-Kuka-Allegro-Reorient-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:Dexsuite3dgProxyKukaAllegroReorientEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Dexsuite3dgProxyKukaAllegroPPORunnerCfg",
    },
)

# Dexsuite 3dg Proxy Lift Environments
gym.register(
    id="Isaac-Dexsuite-3dg-Proxy-Kuka-Allegro-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:Dexsuite3dgProxyKukaAllegroLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Dexsuite3dgProxyKukaAllegroPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Dexsuite-3dg-Proxy-Kuka-Allegro-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:Dexsuite3dgProxyKukaAllegroLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Dexsuite3dgProxyKukaAllegroPPORunnerCfg",
    },
)

# Deformable (VBD soft body) environments
_DEFORMABLE_ENTRY = "isaaclab_tasks_experimental.manager_based.manipulation.dexsuite_3dg_proxy.proxy_deformable_env:Dexsuite3dgProxyDeformableEnv"

gym.register(
    id="Isaac-Dexsuite-3dg-Proxy-Kuka-Allegro-Deformable-Reorient-v0",
    entry_point=_DEFORMABLE_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:Dexsuite3dgProxyKukaAllegroDeformableReorientEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Dexsuite3dgProxyKukaAllegroPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexsuite-3dg-Proxy-Kuka-Allegro-Deformable-Reorient-Play-v0",
    entry_point=_DEFORMABLE_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:Dexsuite3dgProxyKukaAllegroDeformableReorientEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Dexsuite3dgProxyKukaAllegroPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexsuite-3dg-Proxy-Kuka-Allegro-Deformable-Lift-v0",
    entry_point=_DEFORMABLE_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:Dexsuite3dgProxyKukaAllegroDeformableLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Dexsuite3dgProxyKukaAllegroPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexsuite-3dg-Proxy-Kuka-Allegro-Deformable-Lift-Play-v0",
    entry_point=_DEFORMABLE_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:Dexsuite3dgProxyKukaAllegroDeformableLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Dexsuite3dgProxyKukaAllegroPPORunnerCfg",
    },
)
