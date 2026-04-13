# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for the Franka VBD cube pick task.

The cube is a VBD soft body managed by :class:`FrankaVbdCubePickNewtonManager`.
There is no RigidObject asset entry for the cube in the Isaac Lab scene, so
standard ``reset_root_state_uniform`` cannot be used.  Instead, we call
:meth:`FrankaVbdCubePickNewtonManager.reset_particles` directly with a randomly
sampled pose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_cube_pose_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    cube_size: float = 0.05,
) -> None:
    """Randomise the deformable cube pose and reset particle velocities to zero.

    Samples ``x``, ``y`` (and optionally ``z``) from ``pose_range``, sets ``z``
    so the cube centre rests on the ground (``cube_size / 2``), and uses an
    identity quaternion (no initial rotation).

    Args:
        env: The manager-based RL environment.
        env_ids: Indices of the environments to reset.
        pose_range: Dict with optional keys ``"x"`` and ``"y"`` mapping to
            ``(min, max)`` ranges [m].  Keys not present default to 0.0.
        cube_size: Cube side length [m] used to compute the resting z height.
    """
    from ..physics.vbd_newton_manager import FrankaVbdCubePickNewtonManager

    n = env_ids.shape[0]
    device = env.device

    # Sample x, y positions.
    x_range = pose_range.get("x", (0.5, 0.5))
    y_range = pose_range.get("y", (0.0, 0.0))
    x = torch.empty(n, device=device).uniform_(*x_range)
    y = torch.empty(n, device=device).uniform_(*y_range)
    z = torch.full((n,), cube_size / 2.0, device=device)

    # Identity quaternion: w=1, xyz=0.
    quat = torch.zeros(n, 4, device=device)
    quat[:, 0] = 1.0  # w component first (wxyz convention)

    # root_pose layout: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
    root_pose = torch.stack([x, y, z, quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]], dim=1)

    FrankaVbdCubePickNewtonManager.reset_particles(env_ids, root_pose)
    FrankaVbdCubePickNewtonManager.reset_particle_velocities(env_ids)
