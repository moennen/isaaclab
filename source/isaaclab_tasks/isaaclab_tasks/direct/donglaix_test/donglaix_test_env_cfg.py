# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class DonglaixTestEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition: 7 joint pos + 7 joint vel = 14 obs, 7 arm actions
    action_space = 7
    observation_space = 14
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # joint names to control (7 arm joints)
    arm_joint_names = ["panda_joint.*"]

    # action scale (scales position targets relative to default pose)
    action_scale = 0.5  # [rad]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_joint_pos = -0.1   # penalize deviation from default pose
    rew_scale_joint_vel = -0.01  # penalize fast motion

    # reset: joint position noise range around default [rad]
    initial_joint_pos_noise = 0.2
