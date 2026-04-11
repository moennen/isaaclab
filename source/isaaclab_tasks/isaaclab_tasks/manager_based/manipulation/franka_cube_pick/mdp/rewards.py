# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the Franka cube pick + unreachable-signal task.

Two-branch design
-----------------
Reachability is determined geometrically from the cube's XY position relative
to the robot base. The robot is NOT told which branch it is in — it must infer
reachability from its observations.

Branch A — cube REACHABLE:
  1. approach_cube_reachable  : EE closes in on cube
  2. lift_cube_reachable      : cube rises above lift_height
  3. cube_at_success_position : cube reaches the success position while lifted

Branch B — cube UNREACHABLE:
  4. go_to_signal_position    : EE moves to the designated signal position

All reward functions read the geometry constants from env.cfg so that derived
configs can override them without touching this file.

EE position is read from robot.data.body_link_pos_w (panda_hand link), which
works on both PhysX and Newton without requiring a FrameTransformer sensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

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
    object_cfg: SceneEntityCfg,
    ee_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return (cube_pos_w, robot_pos_w, ee_pos_w | None) from env scene."""
    robot = env.scene["robot"]
    cube: RigidObject = env.scene[object_cfg.name]

    cube_pos_w  = cube.data.root_pos_w           # (N, 3)
    robot_pos_w = robot.data.root_pos_w          # (N, 3)

    ee_pos_w = None
    if ee_cfg is not None:
        ee_pos_w = robot.data.body_link_pos_w[:, ee_cfg.body_ids[0], :]  # (N, 3)

    return cube_pos_w, robot_pos_w, ee_pos_w


# ---------------------------------------------------------------------------
# Branch A — reachable
# ---------------------------------------------------------------------------


def approach_cube_reachable(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    """Reward EE approaching the cube, active only when cube is reachable."""
    cube_pos_w, robot_pos_w, ee_pos_w = _get_tensors(env, object_cfg, ee_cfg)
    return compute_approach_cube(
        ee_pos_w, cube_pos_w, robot_pos_w,
        env.cfg.reachable_radius_min, env.cfg.reachable_radius_max, std,
    )


def lift_cube_reachable(
    env: ManagerBasedRLEnv,
    lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary reward for lifting the cube above lift_height, only when reachable."""
    cube_pos_w, robot_pos_w, _ = _get_tensors(env, object_cfg)
    return compute_lift_cube(
        cube_pos_w, robot_pos_w,
        env.cfg.reachable_radius_min, env.cfg.reachable_radius_max, lift_height,
    )


def cube_at_success_position(
    env: ManagerBasedRLEnv,
    std: float,
    lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    """Reward for bringing the lifted cube to the success EE position."""
    cube_pos_w, robot_pos_w, ee_pos_w = _get_tensors(env, object_cfg, ee_cfg)
    success_pos = torch.tensor(
        env.cfg.success_ee_position, device=env.device, dtype=torch.float32
    ).unsqueeze(0)
    return compute_cube_at_success(
        ee_pos_w, cube_pos_w, robot_pos_w, success_pos,
        env.cfg.reachable_radius_min, env.cfg.reachable_radius_max, lift_height, std,
    )


# ---------------------------------------------------------------------------
# Branch B — unreachable
# ---------------------------------------------------------------------------


def grip_cube_reachable(
    env: ManagerBasedRLEnv,
    grip_height: float = 0.05,
    closed_threshold: float = 0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary reward for gripper closed with cube off ground, when reachable.

    gripper_width = panda_finger_joint1 + panda_finger_joint2 (indices 7, 8).
    Fires when gripper_width < closed_threshold AND cube_z > grip_height.
    """
    robot = env.scene["robot"]
    cube: RigidObject = env.scene[object_cfg.name]
    cube_pos_w  = cube.data.root_pos_w
    robot_pos_w = robot.data.root_pos_w
    # panda_finger_joint1 and panda_finger_joint2 occupy fixed indices 7, 8
    # in the Franka joint ordering (panda_joint1..7 + two finger joints).
    gripper_width = robot.data.joint_pos[:, 7] + robot.data.joint_pos[:, 8]
    return compute_grip_cube(
        cube_pos_w, robot_pos_w, gripper_width,
        env.cfg.reachable_radius_min, env.cfg.reachable_radius_max,
        grip_height, closed_threshold,
    )


def go_to_signal_position(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    """Dense shaping reward for moving EE toward signal position when unreachable.

    Kept at weight 1.0; the binary signal_reached_unreachable (weight 10.0)
    is the primary completion reward.
    """
    cube_pos_w, robot_pos_w, ee_pos_w = _get_tensors(env, object_cfg, ee_cfg)
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
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    """Binary reward: EE within signal_threshold of signal position when unreachable.

    Mirrors lift_cube_reachable for the unreachable branch.
    """
    cube_pos_w, robot_pos_w, ee_pos_w = _get_tensors(env, object_cfg, ee_cfg)
    signal_pos = torch.tensor(
        env.cfg.signal_ee_position, device=env.device, dtype=torch.float32
    ).unsqueeze(0)
    return compute_signal_reached(
        ee_pos_w, cube_pos_w, robot_pos_w, signal_pos,
        env.cfg.reachable_radius_min, env.cfg.reachable_radius_max,
        signal_threshold,
    )
