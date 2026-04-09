"""Standalone reward evaluation — re-exports from mdp/reward_utils.py.

This module is intentionally thin. All reward math lives in
``mdp/reward_utils.py``, which is the single source of truth shared by:

  * ``mdp/rewards.py``     — Isaac Lab manager-based reward wrappers (RL env)
  * This file              — standalone replayer / visualizer / unit tests
  * ``tests/``             — unit tests (no simulator required)

Do not add any reward logic here. Edit ``mdp/reward_utils.py`` instead —
the change will propagate to the RL env automatically.
"""

from isaaclab_tasks.manager_based.manipulation.franka_cube_pick.reward_utils import (  # noqa: F401
    REWARD_WEIGHTS,
    compute_action_rate,
    compute_all_rewards,
    compute_approach_cube,
    compute_cube_at_success,
    compute_go_to_signal,
    compute_joint_vel,
    compute_lift_cube,
    reachable_mask,
)
