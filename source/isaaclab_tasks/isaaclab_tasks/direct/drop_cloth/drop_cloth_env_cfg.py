# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import os

import isaaclab.sim as sim_utils
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.meshes import MeshFromFileCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg

from isaaclab_tasks.utils import PresetCfg

# Defer `import newton` to avoid pulling in pxr before SimulationApp starts (Kit crash).
_newton_spec = importlib.util.find_spec("newton")
_SHIRT_USD = os.path.join(
    os.path.dirname(_newton_spec.origin),
    "examples",
    "assets",
    "unisex_shirt.usd",
)


@configclass
class DropClothPhysicsCfg(PresetCfg):
    particle_self_contact_radius = 0.002
    particle_self_contact_margin = 0.002
    soft_contact_ke = 1e4

    default: NewtonCfg = NewtonCfg(
        solver_cfg=VBDSolverCfg(
            iterations=5,
            integrate_with_external_rigid_solver=True,
            particle_self_contact_radius=particle_self_contact_radius,
            particle_self_contact_margin=particle_self_contact_margin,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.5,
            particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=-1,
            rigid_contact_k_start=soft_contact_ke,
        ),
        num_substeps=10,
        use_cuda_graph=False,
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

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation, physics=DropClothPhysicsCfg())

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)

    # cloth asset — mesh geometry loaded from USD file and spawned as UsdGeom.Mesh prim
    cloth: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/cloth",
        spawn=MeshFromFileCfg(
            usd_path=_SHIRT_USD,
            usd_prim_path="/root/shirt",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.0, 0.0, 1.0, 0.0),  # 180° around z-axis (w, x, y, z)
        ),
        mesh_scale=0.01,  # shirt USD vertices are in cm → convert to meters
        density=0.02,
        tri_ke=1e4,
        tri_ka=1e4,
        tri_kd=1.5e-6,
        edge_ke=5.0,
        edge_kd=1e-2,
        particle_radius=0.008,
        soft_contact_ke=1e4,
        soft_contact_kd=1e-2,
    )
