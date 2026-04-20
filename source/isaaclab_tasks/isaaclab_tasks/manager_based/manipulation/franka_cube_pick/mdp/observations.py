# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for the Franka cube pick task.

Follows the dexsuite pattern: reads body state directly from asset data
(no FrameTransformer), so these functions work on both PhysX and Newton.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Cube position expressed in the robot root frame.

    Returns shape (num_envs, 3) [m].
    """
    robot = env.scene[robot_cfg.name]
    cube = env.scene[object_cfg.name]

    robot_pos_w = wp.to_torch(robot.data.root_pos_w)    # (num_envs, 3)
    robot_quat_w = wp.to_torch(robot.data.root_quat_w)  # (num_envs, 4)
    cube_pos_w = wp.to_torch(cube.data.root_pos_w)      # (num_envs, 3)

    cube_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, cube_pos_w)
    return cube_pos_b


def cube_quat_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Cube orientation (quaternion wxyz) expressed in the robot root frame.

    Returns shape (num_envs, 4).
    """
    robot = env.scene[robot_cfg.name]
    cube = env.scene[object_cfg.name]

    robot_pos_w = wp.to_torch(robot.data.root_pos_w)    # (num_envs, 3)
    robot_quat_w = wp.to_torch(robot.data.root_quat_w)  # (num_envs, 4)
    cube_pos_w = wp.to_torch(cube.data.root_pos_w)      # (num_envs, 3)
    cube_quat_w = wp.to_torch(cube.data.root_quat_w)    # (num_envs, 4)

    _, cube_quat_b = subtract_frame_transforms(
        robot_pos_w, robot_quat_w, cube_pos_w, cube_quat_w
    )
    return cube_quat_b


def ee_state_b(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End-effector body state expressed in the robot root frame.

    Reads panda_hand link state directly from the articulation — no
    FrameTransformer required, so this works on both PhysX and Newton.

    Returns shape (num_envs, 13): pos [m] (3) + quat wxyz (4) +
    linear velocity [m/s] (3) + angular velocity [rad/s] (3).
    """
    robot = env.scene[robot_cfg.name]

    robot_pos_w = wp.to_torch(robot.data.root_pos_w)    # (num_envs, 3)
    robot_quat_w = wp.to_torch(robot.data.root_quat_w)  # (num_envs, 4)

    # body_link_state_w: warp (num_envs, num_bodies) vec13f → torch (num_envs, num_bodies, 13)
    ee_state_w = wp.to_torch(robot.data.body_link_state_w)[:, ee_cfg.body_ids[0], :]  # (num_envs, 13)
    ee_pos_w = ee_state_w[:, :3]    # (num_envs, 3)
    ee_quat_w = ee_state_w[:, 3:7]  # (num_envs, 4)
    ee_linvel_w = ee_state_w[:, 7:10]   # (num_envs, 3)
    ee_angvel_w = ee_state_w[:, 10:13]  # (num_envs, 3)

    # Transform pose to robot root frame
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot_pos_w, robot_quat_w, ee_pos_w, ee_quat_w
    )

    # Rotate velocities to robot root frame
    ee_linvel_b = quat_rotate_inverse(robot_quat_w, ee_linvel_w)
    ee_angvel_b = quat_rotate_inverse(robot_quat_w, ee_angvel_w)

    return torch.cat([ee_pos_b, ee_quat_b, ee_linvel_b, ee_angvel_b], dim=-1)
