"""Standalone reward evaluation — re-exports from reward_utils.py.

This module is intentionally thin. All reward math lives in
``reward_utils.py`` (task root), which is the single source of truth shared by:

  * ``mdp/rewards.py``     — Isaac Lab manager-based reward wrappers (RL env)
  * This file              — standalone replayer / visualizer / unit tests
  * ``tests/``             — unit tests (no simulator required)

Do not add any reward logic here. Edit ``reward_utils.py`` instead —
the change will propagate to the RL env automatically.

Implementation note: ``reward_utils.py`` is imported via direct file path
(importlib) rather than through the ``isaaclab_tasks`` package, because the
package __init__ triggers the full Isaac Lab import chain (pxr, omni, etc.)
which is not available in the standalone Newton environment.
"""

import importlib.util
from pathlib import Path

# reward_utils.py is at <task_root>/reward_utils.py
# This file is at <task_root>/scripts/_common/reward_eval.py
_REWARD_UTILS_PATH = Path(__file__).parent.parent.parent / "reward_utils.py"

_spec = importlib.util.spec_from_file_location("reward_utils", _REWARD_UTILS_PATH)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export everything that callers need
REWARD_WEIGHTS       = _mod.REWARD_WEIGHTS         # noqa: F401
compute_action_rate  = _mod.compute_action_rate    # noqa: F401
compute_all_rewards  = _mod.compute_all_rewards    # noqa: F401
compute_approach_cube = _mod.compute_approach_cube # noqa: F401
compute_cube_at_success = _mod.compute_cube_at_success  # noqa: F401
compute_go_to_signal = _mod.compute_go_to_signal   # noqa: F401
compute_joint_vel    = _mod.compute_joint_vel       # noqa: F401
compute_lift_cube    = _mod.compute_lift_cube       # noqa: F401
reachable_mask       = _mod.reachable_mask          # noqa: F401
