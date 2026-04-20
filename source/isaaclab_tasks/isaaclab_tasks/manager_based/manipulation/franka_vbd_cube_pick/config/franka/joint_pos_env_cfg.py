# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda joint-position config for the VBD cube pick + unreachable-signal task."""

from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

from isaaclab_tasks.manager_based.manipulation.franka_vbd_cube_pick import mdp
from isaaclab_tasks.manager_based.manipulation.franka_vbd_cube_pick.franka_vbd_cube_pick_env_cfg import (
    FrankaVbdCubePickEnvCfg,
)


@configclass
class FrankaVbdCubePickFrankaCfg(FrankaVbdCubePickEnvCfg):
    """Franka Panda with joint-position control for the VBD cube pick task."""

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


@configclass
class FrankaVbdCubePickFrankaCfg_PLAY(FrankaVbdCubePickFrankaCfg):
    """Smaller scene for interactive play / debugging."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
