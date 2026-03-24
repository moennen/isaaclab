# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.shapes import CylinderCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


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
    episode_length_s = 2.0
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

    # action scale — position offset from default pose [rad] (used in PhysX mode)
    action_scale = 0.5

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_joint_pos = -0.1
    rew_scale_joint_vel = -0.01

    # reset noise range [rad]
    initial_joint_pos_noise = 0.2

    # cylinder rigid body — falls from 0.5 m above EE each episode
    cylinder_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cylinder",
        spawn=CylinderCfg(
            radius=0.04,
            height=0.15,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 1.0)),
    )
