"""Custom observation functions for the Franka cube pick task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector position expressed in the robot root frame.

    Returns shape (num_envs, 3).
    """
    robot = env.scene["robot"]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # EE position in world frame
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)

    # Robot root position and orientation in world frame
    robot_pos_w = robot.data.root_pos_w        # (num_envs, 3)
    robot_quat_w = robot.data.root_quat_w      # (num_envs, 4)

    # Transform to robot root frame
    from isaaclab.utils.math import quat_rotate_inverse, subtract_frame_transforms
    ee_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, ee_pos_w)
    return ee_pos_b
