# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg


@configclass
class DonglaixTestPhysicsCfg(PresetCfg):
    physx: PhysxCfg = PhysxCfg()
    default: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=20,
            nconmax=20,
            ls_iterations=20,
            cone="pyramidal",
            impratio=1,
            ls_parallel=False,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )
    newton: NewtonCfg = default


@configclass
class DonglaixTestEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # 7 arm joint pos + 7 arm joint vel = 14 obs, 7 arm actions
    action_space = 7
    observation_space = 14
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=DonglaixTestPhysicsCfg())

    # robot — HIGH_PD_CFG: stiffness=400, damping=80, disable_gravity=True for stable control
    robot_cfg: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # joint names to control (7 arm joints, excluding fingers)
    arm_joint_names = ["panda_joint[1-7]"]

    # action scale — position offset from default pose [rad]
    action_scale = 0.5

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_joint_pos = -0.1
    rew_scale_joint_vel = -0.01

    # reset noise range [rad]
    initial_joint_pos_noise = 0.2
