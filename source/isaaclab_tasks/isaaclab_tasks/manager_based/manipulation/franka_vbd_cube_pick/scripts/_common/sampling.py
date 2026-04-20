"""Cube position and label sampling — no Isaac Sim dependency.

Extracted from generate_sequences.py so these functions can be unit-tested
and reused without triggering the AppLauncher / Isaac Sim bootstrap.
"""

from __future__ import annotations

import math
import random

import torch


def sample_label(reachable_ratio: float, success_ratio: float) -> dict:
    """Sample a random label dict. Reachable and success are independent."""
    return {
        "reachable": random.random() < reachable_ratio,
        "success":   random.random() < success_ratio,
    }


def sample_cube_pos(label: dict, cfg: dict, device: torch.device) -> torch.Tensor:
    """Sample a cube XY position consistent with the given label.

    Retries up to 1000 times to find an XY whose horizontal distance from the
    robot base is inside (reachable) or outside (unreachable) the annulus defined
    by cfg['reachable_radius_min'] and cfg['reachable_radius_max'].

    Returns (3,) tensor [x, y, z] in world frame (robot base at origin).
    Raises RuntimeError if no valid position found after 1000 attempts.
    """
    r_min = cfg["reachable_radius_min"]
    r_max = cfg["reachable_radius_max"]
    x_range = cfg["cube_spawn_x"]
    y_range = cfg["cube_spawn_y"]
    z = cfg["cube_half_height"]

    for _ in range(1000):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        dist = math.sqrt(x**2 + y**2)
        is_reachable = (r_min <= dist <= r_max)
        if label["reachable"]:
            is_valid = is_reachable
        else:
            # Restrict unreachable to dist > r_max (outer ring only).
            # Inner ring (dist < r_min) puts the cube inside the arm's sweep path
            # during HOME→PRE_GRASP: the open fingers contact VBD particles and
            # falsely lift the cube before the gripper even closes.
            is_valid = (dist > r_max)
        if is_valid:
            return torch.tensor([x, y, z], device=device, dtype=torch.float32)

    from _common.sequence_schema import label_description
    raise RuntimeError(
        f"Could not sample cube position for label '{label_description(label)}' "
        f"after 1000 attempts. Check spawn range and reachability radii in cfg."
    )
