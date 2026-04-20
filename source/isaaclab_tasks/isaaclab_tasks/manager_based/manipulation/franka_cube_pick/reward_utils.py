# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-tensor reward kernels for the Franka cube pick task.

This module is the **single source of truth** for all reward math.
It has no Isaac Lab / Isaac Sim dependencies — only torch — so it can be
imported by:

  * ``mdp/rewards.py``  — the Isaac Lab manager-based reward wrappers
  * ``scripts/_common/reward_eval.py``  — the standalone validation replayer
  * ``tests/``  — unit tests that run without a simulator

Any change to the reward logic must be made here. The env-integrated wrappers
in ``rewards.py`` and the standalone replayer both call these functions, so a
single edit propagates everywhere automatically.

Reward structure (two symmetric branches)
-----------------------------------------
Reachable branch:
  approach_cube_reachable   (1.0)  dense — EE closes on cube
  grip_cube_reachable       (5.0)  binary — gripper closed + cube off ground
  lift_cube_reachable      (10.0)  binary — cube above lift_height
  cube_at_success_position (15.0)  dense — EE at success pos while lifted

Unreachable branch:
  go_to_signal_position     (1.0)  dense — EE approaches signal (gradient only)
  signal_reached_unreachable(10.0) binary — EE within 5 cm of signal pos

Penalties (both branches):
  action_rate (-1e-4), joint_vel (-1e-4)
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Reachability mask
# ---------------------------------------------------------------------------


def reachable_mask(
    cube_pos_w: torch.Tensor,   # (N, 3) world frame
    robot_pos_w: torch.Tensor,  # (N, 3) world frame
    r_min: float,
    r_max: float,
) -> torch.Tensor:
    """Float mask (N,) — 1.0 if cube is in the reachable annulus, else 0.0.

    Reachability is determined from the horizontal (XY-plane) distance between
    the cube and the robot base only. The Z coordinate is ignored so that a
    lifted cube remains reachable for the duration of the pick.
    """
    dist_xy = torch.norm(cube_pos_w[:, :2] - robot_pos_w[:, :2], dim=1)
    return ((dist_xy >= r_min) & (dist_xy <= r_max)).float()


# ---------------------------------------------------------------------------
# Branch A — reachable
# ---------------------------------------------------------------------------


def compute_approach_cube(
    ee_pos_w: torch.Tensor,     # (N, 3)
    cube_pos_w: torch.Tensor,   # (N, 3)
    robot_pos_w: torch.Tensor,  # (N, 3)
    r_min: float,
    r_max: float,
    std: float = 0.1,
) -> torch.Tensor:
    """(N,) — EE approaching cube reward, gated by reachability mask."""
    mask = reachable_mask(cube_pos_w, robot_pos_w, r_min, r_max)
    dist = torch.norm(ee_pos_w - cube_pos_w, dim=1)
    return mask * (1.0 - torch.tanh(dist / std))


def compute_grip_cube(
    cube_pos_w: torch.Tensor,    # (N, 3)
    robot_pos_w: torch.Tensor,   # (N, 3)
    gripper_width: torch.Tensor, # (N,)  sum of finger joint positions (0=closed, 0.08=open)
    r_min: float,
    r_max: float,
    grip_height: float = 0.05,
    closed_threshold: float = 0.06,
) -> torch.Tensor:
    """(N,) — binary: gripper closed AND cube off ground when reachable.

    Bridges the approach→lift gap: fires when the robot has grasped the cube
    (gripper_width < closed_threshold) and the cube is slightly above the ground
    (cube_z > grip_height), within the reachable zone.

    grip_height=0.05 < lift_height=0.5 so this fires well before lift_cube.
    closed_threshold=0.06: each finger goes 0 (closed) to 0.04 (open). When
    squeezing a ~4 cm cube, contact resistance keeps q₁+q₂ ≈ 0.04 (cube
    half-width × 2); fully open sum = 0.08. Threshold 0.06 fires when grasping
    (0.04 < 0.06) and is off when fully open (0.08 > 0.06).
    """
    mask = reachable_mask(cube_pos_w, robot_pos_w, r_min, r_max)
    off_ground = (cube_pos_w[:, 2] > grip_height).float()
    closed     = (gripper_width < closed_threshold).float()
    return mask * off_ground * closed


def compute_lift_cube(
    cube_pos_w: torch.Tensor,   # (N, 3)
    robot_pos_w: torch.Tensor,  # (N, 3)
    r_min: float,
    r_max: float,
    lift_height: float = 0.5,
) -> torch.Tensor:
    """(N,) — binary: cube above lift_height when reachable."""
    mask = reachable_mask(cube_pos_w, robot_pos_w, r_min, r_max)
    lifted = (cube_pos_w[:, 2] > lift_height).float()
    return mask * lifted


def compute_cube_at_success(
    ee_pos_w: torch.Tensor,          # (N, 3)
    cube_pos_w: torch.Tensor,        # (N, 3)
    robot_pos_w: torch.Tensor,       # (N, 3)
    success_ee_pos: torch.Tensor,    # (1, 3) or (N, 3)
    r_min: float,
    r_max: float,
    lift_height: float = 0.5,
    std: float = 0.1,
) -> torch.Tensor:
    """(N,) — EE near success position, only when reachable and cube is lifted."""
    mask = reachable_mask(cube_pos_w, robot_pos_w, r_min, r_max)
    lifted = (cube_pos_w[:, 2] > lift_height).float()
    dist = torch.norm(ee_pos_w - success_ee_pos, dim=1)
    return mask * lifted * (1.0 - torch.tanh(dist / std))


# ---------------------------------------------------------------------------
# Branch B — unreachable
# ---------------------------------------------------------------------------


def compute_go_to_signal(
    ee_pos_w: torch.Tensor,         # (N, 3)
    cube_pos_w: torch.Tensor,       # (N, 3)
    robot_pos_w: torch.Tensor,      # (N, 3)
    signal_ee_pos: torch.Tensor,    # (1, 3) or (N, 3)
    r_min: float,
    r_max: float,
    std: float = 0.1,
) -> torch.Tensor:
    """(N,) — dense EE-to-signal approach reward when unreachable.

    Provides gradient to drive the EE toward the signal position.
    Kept at a low weight (1.0) to serve as shaping only; the binary
    signal_reached_unreachable term (weight 10.0) is the primary reward.
    """
    mask = reachable_mask(cube_pos_w, robot_pos_w, r_min, r_max)
    unreachable = 1.0 - mask
    dist = torch.norm(ee_pos_w - signal_ee_pos, dim=1)
    return unreachable * (1.0 - torch.tanh(dist / std))


def compute_signal_reached(
    ee_pos_w: torch.Tensor,         # (N, 3)
    cube_pos_w: torch.Tensor,       # (N, 3)
    robot_pos_w: torch.Tensor,      # (N, 3)
    signal_ee_pos: torch.Tensor,    # (1, 3) or (N, 3)
    r_min: float,
    r_max: float,
    threshold: float = 0.05,
) -> torch.Tensor:
    """(N,) — binary: EE within threshold of signal position when unreachable.

    Mirrors lift_cube_reachable for the unreachable branch.
    threshold=0.05 m: tight enough to require deliberate positioning.
    """
    unreachable = 1.0 - reachable_mask(cube_pos_w, robot_pos_w, r_min, r_max)
    dist = torch.norm(ee_pos_w - signal_ee_pos, dim=1)
    return unreachable * (dist < threshold).float()


# ---------------------------------------------------------------------------
# Penalties (both branches)
# ---------------------------------------------------------------------------


def compute_action_rate(
    action_curr: torch.Tensor,   # (N, A)
    action_prev: torch.Tensor,   # (N, A)
) -> torch.Tensor:
    """(N,) — smoothness penalty on action delta."""
    return -torch.sum((action_curr - action_prev) ** 2, dim=1)


def compute_joint_vel(
    joint_vel: torch.Tensor,   # (N, J)
) -> torch.Tensor:
    """(N,) — velocity penalty."""
    return -torch.sum(joint_vel ** 2, dim=1)


# ---------------------------------------------------------------------------
# Weights and composite computation
# ---------------------------------------------------------------------------


REWARD_WEIGHTS: dict[str, float] = {
    # --- Reachable branch ---
    "approach_cube_reachable":    1.0,   # dense shaping — EE closes on cube
    "grip_cube_reachable":        5.0,   # binary — gripper closed + cube off ground
    "lift_cube_reachable":       10.0,   # binary — cube above lift_height
    "cube_at_success_position":  15.0,   # dense — EE at success pos while lifted
    # --- Unreachable branch ---
    "go_to_signal_position":      1.0,   # dense shaping only (was 10.0; binary below)
    "signal_reached_unreachable": 10.0,  # binary — EE within 5 cm of signal
    # --- Penalties ---
    "action_rate":               -1e-4,
    "joint_vel":                 -1e-4,
}


def compute_all_rewards(
    ee_pos_w: torch.Tensor,         # (N, 3)
    cube_pos_w: torch.Tensor,       # (N, 3)
    robot_pos_w: torch.Tensor,      # (N, 3)
    gripper_width: torch.Tensor,    # (N,)   sum of finger joint positions
    joint_vel: torch.Tensor,        # (N, J)
    action_curr: torch.Tensor,      # (N, A)
    action_prev: torch.Tensor,      # (N, A)
    success_ee_pos: torch.Tensor,   # (1, 3)
    signal_ee_pos: torch.Tensor,    # (1, 3)
    r_min: float,
    r_max: float,
    lift_height: float = 0.5,
    std: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Return dict of (N,) reward tensors for each term, plus 'total' and 'reachable_mask'."""
    mask = reachable_mask(cube_pos_w, robot_pos_w, r_min, r_max)

    terms = {
        "approach_cube_reachable":   compute_approach_cube(
            ee_pos_w, cube_pos_w, robot_pos_w, r_min, r_max, std),
        "grip_cube_reachable":       compute_grip_cube(
            cube_pos_w, robot_pos_w, gripper_width, r_min, r_max),
        "lift_cube_reachable":       compute_lift_cube(
            cube_pos_w, robot_pos_w, r_min, r_max, lift_height),
        "cube_at_success_position":  compute_cube_at_success(
            ee_pos_w, cube_pos_w, robot_pos_w, success_ee_pos, r_min, r_max, lift_height, std),
        "go_to_signal_position":     compute_go_to_signal(
            ee_pos_w, cube_pos_w, robot_pos_w, signal_ee_pos, r_min, r_max, std),
        "signal_reached_unreachable": compute_signal_reached(
            ee_pos_w, cube_pos_w, robot_pos_w, signal_ee_pos, r_min, r_max),
        "action_rate":               compute_action_rate(action_curr, action_prev),
        "joint_vel":                 compute_joint_vel(joint_vel),
    }

    total = sum(REWARD_WEIGHTS[k] * v for k, v in terms.items())
    terms["total"] = total
    terms["reachable_mask"] = mask
    return terms
