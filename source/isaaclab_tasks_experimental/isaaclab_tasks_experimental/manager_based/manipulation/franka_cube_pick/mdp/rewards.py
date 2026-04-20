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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _is_reachable(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return a float mask (num_envs,) — 1.0 if cube is reachable, 0.0 otherwise.

    Reachability condition (horizontal plane only):
        r_min <= ||cube_xy - robot_xy||  <= r_max

    Values of r_min and r_max are read from env.cfg so derived configs can
    override them.
    """
    robot = env.scene["robot"]
    cube: RigidObject = env.scene[object_cfg.name]

    robot_xy = robot.data.root_pos_w[:, :2]  # (num_envs, 2)
    cube_xy = cube.data.root_pos_w[:, :2]    # (num_envs, 2)

    dist = torch.norm(cube_xy - robot_xy, dim=1)  # (num_envs,)

    r_min = env.cfg.reachable_radius_min
    r_max = env.cfg.reachable_radius_max

    reachable = (dist >= r_min) & (dist <= r_max)
    return reachable.float()


# ---------------------------------------------------------------------------
# Branch A — reachable
# ---------------------------------------------------------------------------


def approach_cube_reachable(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward EE approaching the cube, active only when cube is reachable.

    Returns 0 for unreachable cubes so the two branches don't mix gradients.
    """
    reachable = _is_reachable(env, object_cfg)

    cube: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = cube.data.root_pos_w                   # (num_envs, 3)
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]    # (num_envs, 3)

    dist = torch.norm(cube_pos_w - ee_pos_w, dim=1)
    reward = 1.0 - torch.tanh(dist / std)

    return reachable * reward


def lift_cube_reachable(
    env: ManagerBasedRLEnv,
    lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary reward for lifting the cube above lift_height, only when reachable."""
    reachable = _is_reachable(env, object_cfg)

    cube: RigidObject = env.scene[object_cfg.name]
    lifted = (cube.data.root_pos_w[:, 2] > lift_height).float()

    return reachable * lifted


def cube_at_success_position(
    env: ManagerBasedRLEnv,
    std: float,
    lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for bringing the lifted cube to the success EE position.

    Active only when cube is reachable AND lifted.
    Success position is read from env.cfg.success_ee_position (world frame,
    assuming robot base at origin).
    """
    reachable = _is_reachable(env, object_cfg)

    cube: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    lifted = (cube.data.root_pos_w[:, 2] > lift_height).float()

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)

    # Success position in world frame (robot base at origin, so world == robot frame)
    success_pos = torch.tensor(
        env.cfg.success_ee_position, device=env.device, dtype=torch.float32
    ).unsqueeze(0)  # (1, 3)

    dist = torch.norm(ee_pos_w - success_pos, dim=1)
    reward = 1.0 - torch.tanh(dist / std)

    return reachable * lifted * reward


# ---------------------------------------------------------------------------
# Branch B — unreachable
# ---------------------------------------------------------------------------


def go_to_signal_position(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for moving EE to the designated signal position, only when unreachable.

    Signal position is read from env.cfg.signal_ee_position (world frame).
    """
    reachable = _is_reachable(env, object_cfg)
    unreachable = 1.0 - reachable

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)

    signal_pos = torch.tensor(
        env.cfg.signal_ee_position, device=env.device, dtype=torch.float32
    ).unsqueeze(0)  # (1, 3)

    dist = torch.norm(ee_pos_w - signal_pos, dim=1)
    reward = 1.0 - torch.tanh(dist / std)

    return unreachable * reward
