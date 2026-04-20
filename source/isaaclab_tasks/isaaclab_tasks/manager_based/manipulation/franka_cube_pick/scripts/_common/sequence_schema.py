"""JSON schema, dataclasses and I/O helpers shared by all three validation tools.

Sequence file (Tool 1 output / Tool 2 input)
--------------------------------------------
{
  "version": "1.0",
  "config": { geometry constants embedded at generation time },
  "sequences": [
    {
      "id": "seq_0000",
      "label": { "reachable": bool, "success": bool },
      "cube_init_pos_w": [x, y, z],
      "cube_horizontal_dist": float,           # from robot base
      "frames": [
        {
          "step": int,
          "t": float,
          "joint_pos_cmd": [9 floats],         # IK-solved targets fed to PD controller
          "joint_pos": [9 floats],             # actual joint positions after physics step
          "joint_vel": [9 floats],             # finite-diff of actual joint_pos
          "gripper_closed": bool,
          "robot_pos_w": [x, y, z],            # robot base world position
          "ee_pos_w": [x, y, z],
          "cube_pos_w": [x, y, z]
        }
      ]
    }
  ]
}

Replay output file (Tool 2 output / Tool 3 input)
-------------------------------------------------
{
  "version": "1.0",
  "source": "sequences.json",
  "sequences": [
    {
      "id": "seq_0000",
      "label": { "reachable": bool, "success": bool },
      "expected_high_reward": bool,
      "episode_rewards": {                     # summed over all frames
        "approach_cube_reachable": float,
        "lift_cube_reachable": float,
        "cube_at_success_position": float,
        "go_to_signal_position": float,
        "action_rate": float,
        "joint_vel": float,
        "total": float
      },
      "success_frame": int,                    # first frame index where success achieved, or -1
      "joint_pos_variance_mean": float,        # mean |gen joint_pos - replay joint_pos| across frames
      "frames": [
        {
          "step": int,
          "t": float,
          "reachable_mask": float,             # 1.0 or 0.0
          "robot_pos_w": [x, y, z],            # robot base world position
          "ee_pos_w": [x, y, z],
          "cube_pos_w": [x, y, z],
          "rewards": {
            "approach_cube_reachable": float,
            "lift_cube_reachable": float,
            "cube_at_success_position": float,
            "go_to_signal_position": float,
            "action_rate": float,
            "joint_vel": float,
            "total": float
          }
        }
      ]
    }
  ]
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

LABEL_REACHABLE_SUCCESS   = {"reachable": True,  "success": True}
LABEL_REACHABLE_FAILURE   = {"reachable": True,  "success": False}
LABEL_UNREACHABLE_SUCCESS = {"reachable": False, "success": True}
LABEL_UNREACHABLE_FAILURE = {"reachable": False, "success": False}


def label_description(label: dict) -> str:
    r = "reachable" if label["reachable"] else "unreachable"
    s = "success" if label["success"] else "failure"
    return f"{r}_{s}"


def expected_high_reward(label: dict) -> bool:
    """A sequence is expected to produce high total reward iff it is labelled success."""
    return label["success"]


# ---------------------------------------------------------------------------
# Geometry config (embedded in every sequence file so replay is self-contained)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "reachable_radius_min":  0.22,
    "reachable_radius_max":  0.65,
    "success_ee_position":   [0.5, 0.0, 0.5],
    "signal_ee_position":    [0.0, 0.0, 0.8],
    "lift_height":           0.45,   # reward threshold: cube must exceed this z [m]
    "lift_ee_target_z":      0.65,   # IK target z during lift phase — higher than lift_height
                                     # to compensate PD tracking lag and EE-cube offset (~8 cm
                                     # between panda_hand and cube centre when gripped)
    "grasp_approach_height": 0.25,   # metres above cube for pre-grasp waypoint
                                     # Must be large enough so fingers (≈12 cm below panda_hand)
                                     # clear the cube top even after PD lag (~17 mm).
                                     # 0.25 m → pre-grasp EE z=0.275 m → PD settles to ~0.258 m
                                     # → finger tips at ~0.138 m, well above cube top (0.05 m).
    "grasp_ee_height":       0.10,   # metres above cube centre for final grasp EE target
                                     # panda finger joint is 0.0584 m in hand-Z from hand centre;
                                     # 0.10 m places finger joints at cube equator while keeping
                                     # fingertips (~0.055 m below joint) above the ground plane.
    "cube_spawn_x":          [0.0, 0.8],
    "cube_spawn_y":          [-0.6, 0.6],
    "cube_half_height":      0.025,  # cube rests with centre at this z
    "grasp_orientation_wxyz":  [0.0, 1.0, 0.0, 0.0],   # EE pointing down
    "neutral_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],  # EE neutral
}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_sequences(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_sequences(data: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[sequence_schema] Saved {len(data['sequences'])} sequences → {path}")


def load_replay(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_replay(data: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[sequence_schema] Saved replay of {len(data['sequences'])} sequences → {path}")
