# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP components for the Franka cube pick task."""

# Re-export standard mdp functions used in the env cfg
from isaaclab.envs.mdp import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
    action_rate_l2,
    joint_pos_rel,
    joint_vel_l2,
    joint_vel_rel,
    last_action,
    reset_root_state_uniform,
    reset_scene_to_default,
    root_height_below_minimum,
    time_out,
)

# Task-specific observation functions (dexsuite-style, no FrameTransformer)
from .observations import cube_pos_b, cube_quat_b, ee_state_b

# Task-specific reward functions (env-wrapping layer)
from .rewards import (
    approach_cube_reachable,
    cube_at_success_position,
    go_to_signal_position,
    lift_cube_reachable,
)

# Pure-tensor reward kernels — single source of truth, no Isaac Lab dependencies.
# reward_utils lives at the task root (not inside mdp/) so it can be imported
# without triggering the mdp/__init__.py → rewards.py → isaaclab.assets chain.
from ..reward_utils import REWARD_WEIGHTS, compute_all_rewards
