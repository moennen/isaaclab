# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Pick-Rigid-Cylinder environment: Franka robot + rigid cylinder."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.shapes import CylinderCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonModelCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG, FRANKA_PANDA_CFG
from isaaclab_tasks.utils import PresetCfg, preset


MODEL_CFG = NewtonModelCfg(
    soft_contact_ke=1e4,
    soft_contact_kd=1e-2,
    soft_contact_mu=0.5,
    shape_material_ke=5e3,
    shape_material_kd=5e2,
)


@configclass
class PickRigidCylinderPhysicsCfg(PresetCfg):
    """Physics presets for the Pick-Rigid-Cylinder environment.

    Presets:
        - ``default`` / ``newton`` / ``newton_mjwarp``: MuJoCo Warp rigid solver (recommended).
        - ``physx``: PhysX rigid solver.
    """

    physx: PhysxCfg = PhysxCfg()

    default: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=40,
            nconmax=40,
            ls_iterations=20,
            cone="pyramidal",
            impratio=1,
            ls_parallel=False,
            integrator="implicitfast",
            ccd_iterations=100,
        ),
        model_cfg=MODEL_CFG,
        num_substeps=10,
        use_cuda_graph=True,
    )

    newton: NewtonCfg = default
    newton_mjwarp: NewtonCfg = default


@configclass
class PickRigidCylinderEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # obs = joint_pos(7) + joint_vel(7) + cylinder_pos(3) = 17, act = 7
    action_space = 7
    observation_space = 17
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics=PickRigidCylinderPhysicsCfg(),
        visualizer_cfgs=NewtonVisualizerCfg(
            camera_position=(2.0, 2.0, 0.5),
            camera_target=(0.0, 0.0, 0.5),
            record_dir="",
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # robot
    robot_cfg = preset(
        default=FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot"),
        franka_high_pd=FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot"),
    )

    # joint names to control (7 arm joints, excluding fingers)
    arm_joint_names = ["panda_joint[1-7]"]

    # control mode: "position" or "velocity"
    control_mode: str = "position"

    # action scale applied to raw actions before use as targets
    action_scale = 0.5

    # rigid cylinder (standing upright, axis=Z)
    cylinder: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cylinder",
        spawn=CylinderCfg(
            radius=0.02,
            height=0.10,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.05),
        ),
    )

    # interactive IK: when True, spawn a draggable sphere and solve IK each step
    interactive_ik: bool = False

    # reward scales
    rew_scale_cylinder_height = 5.0
    """Reward for lifting cylinder higher [per m]."""

    rew_scale_ee_cylinder_dist = -2.0
    """Penalty for EE-to-cylinder distance [per m]."""

    rew_scale_joint_vel = -0.01
    """Penalty for joint velocities."""
