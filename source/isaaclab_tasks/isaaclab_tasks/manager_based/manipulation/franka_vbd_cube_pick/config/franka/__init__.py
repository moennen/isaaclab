# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Newton + VBD only.  Launch with:
#   ./isaaclab.sh -p train.py --task Isaac-Pick-VBD-Cube-Franka-v0 presets=newton
#
# The VBD physics backend requires Newton (MuJoCo-Warp).
# launch_simulation() auto-detects NewtonCfg — no --experience flag needed.
gym.register(
    id="Isaac-Pick-VBD-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaVbdCubePickFrankaCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaVbdCubePickPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Pick-VBD-Cube-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaVbdCubePickFrankaCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaVbdCubePickPPORunnerCfg",
    },
    disable_env_checker=True,
)
