# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Pick-Cloth environment: Franka robot + cloth with coupled solver."""

import importlib.util
import os.path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.meshes import MeshFromFileCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import CoupledSolverCfg, FeatherstoneSolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonModelCfg, VBDSolverCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG, FRANKA_PANDA_CFG
from isaaclab_tasks.utils import PresetCfg, preset

# Locate shirt USD from Newton package (defer import to avoid pxr before SimulationApp).
_newton_spec = importlib.util.find_spec("newton")
_SHIRT_USD = os.path.join(
    os.path.dirname(_newton_spec.origin),
    "examples",
    "assets",
    "unisex_shirt.usd",
)

MODEL_CFG = NewtonModelCfg(
    soft_contact_ke=1e4,
    soft_contact_kd=1e-2,
    soft_contact_mu=0.5,
)


@configclass
class PickClothPhysicsCfg(PresetCfg):
    """Physics presets for the Pick-Cloth environment.

    Presets:
        - ``default`` / ``newton`` / ``newton_mjwarp``: MuJoCo Warp rigid solver + VBD cloth (recommended).
        - ``newton_featherstone``: Featherstone rigid solver + VBD cloth.
        - ``cloth_only``: VBD cloth only, no rigid-body solver.
    """

    default: NewtonCfg = NewtonCfg(
        solver_cfg=CoupledSolverCfg(
            rigid_solver_cfg=MJWarpSolverCfg(
                njmax=40,
                nconmax=20,
                ls_iterations=20,
                cone="pyramidal",
                impratio=1,
                ls_parallel=False,
                integrator="implicitfast",
                ccd_iterations=100,
            ),
            vbd_cfg=VBDSolverCfg(
                iterations=5,
                particle_enable_self_contact=True,
                particle_self_contact_radius=2e-3,  # good for substeps=10
                particle_self_contact_margin=2e-3,
                particle_topological_contact_filter_threshold=1,
                particle_rest_shape_contact_exclusion_radius=0.0,
                particle_vertex_contact_buffer_size=16,
                particle_edge_contact_buffer_size=20,
                particle_collision_detection_interval=-1,
                integrate_with_external_rigid_solver=True,
            ),
            soft_contact_margin=0.01,
        ),
        model_cfg=MODEL_CFG,
        num_substeps=10,
        use_cuda_graph=True,
    )

    newton: NewtonCfg = default
    newton_mjwarp: NewtonCfg = default

    newton_featherstone: NewtonCfg = NewtonCfg(
        solver_cfg=CoupledSolverCfg(
            rigid_solver_cfg=FeatherstoneSolverCfg(),
            vbd_cfg=VBDSolverCfg(
                iterations=5,
                particle_enable_self_contact=True,
                particle_self_contact_radius=1e-4,
                particle_self_contact_margin=2e-3,
                particle_topological_contact_filter_threshold=1,
                particle_rest_shape_contact_exclusion_radius=0.0,
                particle_vertex_contact_buffer_size=16,
                particle_edge_contact_buffer_size=20,
                particle_collision_detection_interval=-1,
                integrate_with_external_rigid_solver=True,
            ),
            soft_contact_margin=0.01,
        ),
        model_cfg=MODEL_CFG,
        num_substeps=30,
        use_cuda_graph=True,
    )



@configclass
class PickClothEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # With robot: obs = joint_pos(7) + joint_vel(7) + cloth_centroid(3) = 17, act = 7
    # Without robot (robot_cfg=None): obs = cloth_centroid(3) = 3, act = 0
    action_space = 7
    observation_space = 17
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics=PickClothPhysicsCfg(),
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

    # robot (use presets=cloth_only to run without a robot)
    robot_cfg = preset(
        default=FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot"),
        franka_high_pd=FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot"),
        cloth_only=None,
    )

    # joint names to control (7 arm joints, excluding fingers)
    arm_joint_names = ["panda_joint[1-7]"]

    # control mode: "position" (PD, actions are joint position offsets [rad])
    #               "velocity" (P on velocity, actions are joint velocity targets [rad/s])
    control_mode: str = "position"

    # action scale applied to raw actions before use as targets
    action_scale = 0.5

    # cloth asset — shirt mesh loaded from Newton assets
    cloth: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/cloth",
        spawn=MeshFromFileCfg(
            usd_path=_SHIRT_USD,
            usd_prim_path="/root/shirt",
            scale=0.01,  # shirt USD vertices are in cm -> convert to meters
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.9, 1.25, 0.20),  # in front of robot, reachable height
            # pos=(0.9, 1.25, 1.0),  # in front of robot, reachable height
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        density=0.02,
        tri_ke=1e4,
        tri_ka=1e4,
        tri_kd=1.5e-6,
        edge_ke=5.0,
        edge_kd=1e-2,
        particle_radius=0.01,
    )

    # interactive IK: when True, spawn a draggable sphere and solve IK each step
    interactive_ik: bool = False

    # reward scales
    rew_scale_cloth_height = 5.0
    """Reward for lifting cloth centroid higher [per m]."""

    rew_scale_ee_cloth_dist = -2.0
    """Penalty for EE-to-cloth-centroid distance [per m]."""

    rew_scale_joint_vel = -0.01
    """Penalty for joint velocities."""
