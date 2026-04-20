# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the Franka VBD cube pick + unreachable-signal task.

Cube position is read from :class:`FrankaVbdCubePickNewtonManager` (particle CoM)
instead of from a RigidObject.  All reward math lives in ``reward_utils.py``;
this module is the env-wrapping layer only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

from ..physics.vbd_newton_manager import FrankaVbdCubePickNewtonManager
from ..reward_utils import (
    compute_approach_cube,
    compute_cube_at_success,
    compute_go_to_signal,
    compute_grip_cube,
    compute_lift_cube,
    compute_signal_reached,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Internal helper — extract world-frame tensors from env scene
# ---------------------------------------------------------------------------


def _get_tensors(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return ``(cube_pos_w, robot_pos_w, ee_pos_w | None)`` from env scene."""
    robot = env.scene["robot"]
    robot_pos_w = wp.to_torch(robot.data.root_pos_w)  # (N, 3)

    # Cube CoM position from VBD manager obs cache.
    pose = FrankaVbdCubePickNewtonManager.get_object_pose()
    if pose is not None:
        cube_pos_w = wp.to_torch(pose[0]).float()  # (N, 3)
    else:
        cube_pos_w = torch.zeros_like(robot_pos_w)

    ee_pos_w = None
    if ee_cfg is not None:
        # body_link_pos_w: warp (N, B) vec3f → torch (N, B, 3)
        ee_pos_w = wp.to_torch(robot.data.body_link_pos_w)[:, ee_cfg.body_ids[0], :]  # (N, 3)

    return cube_pos_w, robot_pos_w, ee_pos_w


# ---------------------------------------------------------------------------
# Branch A — reachable
# ---------------------------------------------------------------------------


def approach_cube_reachable(
    env: ManagerBasedRLEnv,
    std: float,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    """Reward EE approaching the cube CoM, active only when cube is reachable."""
    cube_pos_w, robot_pos_w, ee_pos_w = _get_tensors(env, ee_cfg)
    return compute_approach_cube(
        ee_pos_w, cube_pos_w, robot_pos_w,
        env.cfg.reachable_radius_min, env.cfg.reachable_radius_max, std,
    )


def lift_cube_reachable(
    env: ManagerBasedRLEnv,
    lift_height: float,
) -> torch.Tensor:
    """Binary reward for lifting the cube CoM above lift_height, only when reachable."""
    cube_pos_w, robot_pos_w, _ = _get_tensors(env)
    return compute_lift_cube(
        cube_pos_w, robot_pos_w,
        env.cfg.reachable_radius_min, env.cfg.reachable_radius_max, lift_height,
    )


def cube_at_success_position(
    env: ManagerBasedRLEnv,
    std: float,
    lift_height: float,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    """Reward for bringing the lifted cube CoM to the success EE position."""
    cube_pos_w, robot_pos_w, ee_pos_w = _get_tensors(env, ee_cfg)
    success_pos = torch.tensor(
        env.cfg.success_ee_position, device=env.device, dtype=torch.float32
    ).unsqueeze(0)
    return compute_cube_at_success(
        ee_pos_w, cube_pos_w, robot_pos_w, success_pos,
        env.cfg.reachable_radius_min, env.cfg.reachable_radius_max, lift_height, std,
    )


def grip_cube_reachable(
    env: ManagerBasedRLEnv,
    grip_height: float = 0.05,
    closed_threshold: float = 0.02,
) -> torch.Tensor:
    """Binary reward for gripper closed with cube CoM off ground, when reachable."""
    robot = env.scene["robot"]
    cube_pos_w, robot_pos_w, _ = _get_tensors(env)
    joint_pos     = wp.to_torch(robot.data.joint_pos)
    gripper_width = joint_pos[:, 7] + joint_pos[:, 8]
    return compute_grip_cube(
        cube_pos_w, robot_pos_w, gripper_width,
        env.cfg.reachable_radius_min, env.cfg.reachable_radius_max,
        grip_height, closed_threshold,
    )


# ---------------------------------------------------------------------------
# Branch B — unreachable
# ---------------------------------------------------------------------------


def go_to_signal_position(
    env: ManagerBasedRLEnv,
    std: float,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    """Dense shaping reward for moving EE toward signal position when unreachable."""
    cube_pos_w, robot_pos_w, ee_pos_w = _get_tensors(env, ee_cfg)
    signal_pos = torch.tensor(
        env.cfg.signal_ee_position, device=env.device, dtype=torch.float32
    ).unsqueeze(0)
    return compute_go_to_signal(
        ee_pos_w, cube_pos_w, robot_pos_w, signal_pos,
        env.cfg.reachable_radius_min, env.cfg.reachable_radius_max, std,
    )


def signal_reached_unreachable(
    env: ManagerBasedRLEnv,
    signal_threshold: float = 0.05,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    """Binary reward: EE within signal_threshold of signal position when unreachable."""
    cube_pos_w, robot_pos_w, ee_pos_w = _get_tensors(env, ee_cfg)
    signal_pos = torch.tensor(
        env.cfg.signal_ee_position, device=env.device, dtype=torch.float32
    ).unsqueeze(0)
    return compute_signal_reached(
        ee_pos_w, cube_pos_w, robot_pos_w, signal_pos,
        env.cfg.reachable_radius_min, env.cfg.reachable_radius_max,
        signal_threshold,
    )
