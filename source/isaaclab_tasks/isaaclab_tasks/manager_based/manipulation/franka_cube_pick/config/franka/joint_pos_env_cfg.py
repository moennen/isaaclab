# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda joint-position config for the cube pick + unreachable-signal task."""

from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

from isaaclab_tasks.manager_based.manipulation.franka_cube_pick import mdp
from isaaclab_tasks.manager_based.manipulation.franka_cube_pick.franka_cube_pick_env_cfg import (
    FrankaCubePickEnvCfg,
)


@configclass
class FrankaCubePickEnvCfg(FrankaCubePickEnvCfg):
    """Franka Panda with joint-position control for the cube pick task."""

    def __post_init__(self):
        super().__post_init__()

        # -- Robot --
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # -- Actions --
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # -- Cube object (rigid, on ground) --
        # 5 cm cube (scale=1.0) matching Newton validation: _CUBE_HALF_SIZE=0.025 m.
        # init_state z=0.025 = half-height, resting on ground at z=0.
        # Physics parameters match Newton standalone validation exactly:
        #   mass=0.1 kg  (density=400 kg/m³ × (0.05 m)³)
        #   static/dynamic friction=0.75  (Newton _CONTACT_MU=0.75)
        #   restitution=0.0  (inelastic contacts, matches Newton ke/kd damped contact)
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.025], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),  # 5 cm cube — matches Newton _CUBE_HALF_SIZE=0.025
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=0.1),  # 400 kg/m³ × (0.05)³ = 0.1 kg
                physics_material=RigidBodyMaterialCfg(
                    static_friction=0.75,   # Newton _CONTACT_MU=0.75
                    dynamic_friction=0.75,
                    restitution=0.0,        # fully inelastic — matches Newton damped contact
                ),
            ),
        )


@configclass
class FrankaCubePickEnvCfg_PLAY(FrankaCubePickEnvCfg):
    """Smaller scene for interactive play / debugging."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
