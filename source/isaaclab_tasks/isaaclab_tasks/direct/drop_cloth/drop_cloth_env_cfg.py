# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg


@configclass
class DropClothPhysicsCfg(PresetCfg):
    default: NewtonCfg = NewtonCfg(
        solver_cfg=VBDSolverCfg(
            iterations=5,
            particle_enable_self_contact=False,
            soft_contact_margin=0.01,
            particle_collision_detection_interval=-1,
        ),
        num_substeps=10,
        use_cuda_graph=False,  # Disable for debugging; re-enable once working
    )
    newton: NewtonCfg = default


@configclass
class DropClothEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 2.0
    # no RL actions; minimal observation space
    action_space = 0
    observation_space = 1
    state_space = 0

    # simulation — dt per physics step (before substeps)
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation, physics=DropClothPhysicsCfg())

    # scene — single environment, no robot
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)

    # cloth drop height above the ground [m]
    cloth_drop_height: float = 1.5
