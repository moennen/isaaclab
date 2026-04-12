# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Physics backend is selected at runtime via the preset system:
#   PhysX (default):  ./isaaclab.sh -p train.py --task Isaac-Pick-Cube-Franka-v0
#   Newton:           ./isaaclab.sh -p train.py --task Isaac-Pick-Cube-Franka-v0 presets=newton
#
# Newton parameters are validated against the standalone Newton simulation
# (generate_sequences.py): solver=newton, integrator=implicitfast, iterations=20,
# cone=elliptic, impratio=1000, 10×2ms substeps, μ=0.75, ke=5e4, kd=5e2.
# launch_simulation() auto-detects NewtonCfg — no --experience flag needed.
gym.register(
    id="Isaac-Pick-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubePickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCubePickPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Pick-Cube-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubePickEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCubePickPPORunnerCfg",
    },
    disable_env_checker=True,
)
