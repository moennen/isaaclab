# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted grasp motion sequence for the pick_rigid_cube environment.

Drives the existing Newton IK solver with a pre-defined sequence of end-effector
waypoints — approach, descend, grasp, lift, hold — without any RL policy.
Mirrors the scripted motion pattern from ``newton/examples/cloth/example_cloth_franka.py``.

Run with::

    ./isaaclab.sh -p scripts/environments/scripted_grasp.py \\
        --task Isaac-Pick-Rigid-Cube-Direct-v0 --num_envs 1 \\
        --visualizer newton presets=newton,franka_high_pd \\
        env.interactive_ik=true env.episode_length_s=1000.0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import torch
import warp as wp

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Scripted grasp agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


# ──────────────────────────────────────────────────────────────────────────────
# Scripted grasp controller
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Phase:
    name: str
    duration_s: float
    ee_offset: tuple[float, float, float]  # offset relative to cube center [m]
    gripper_closed: bool


# Key poses — offsets relative to cube center at phase entry, meters.
# Positive z = above the cube in world frame. XY = (0, 0) = directly above.
#
# Hand→TCP offset (from fr3_franka_hand.urdf, fr3_hand_tcp_joint):
#   xyz="0 0 0.1034" — TCP is 103.4 mm along the hand's local +Z from panda_hand body.
#   For a top-down grasp the hand's +Z points toward the ground, so the TCP is
#   10.34 cm BELOW the hand body in world frame.
#
# To place the TCP at cube center (z_offset = 0.0 from cube center):
#   hand body z_offset = TCP_z_offset + HAND_TCP_Z = 0.0 + 0.1034 = 0.1034 m
#
# For a clean side pinch, we want the TCP a little above the cube base so the
# fingers straddle the cube sides. Targeting TCP at cube center (z_offset=0.0)
# means the hand body sits at HAND_TCP_Z above the cube center.
HAND_TCP_Z      = 0.1034  # hand body → TCP offset along world Z for top-down grasp [m]
TCP_GRASP_Z     = 0.0     # desired TCP z-offset above cube center at grasp [m]
GRASP_Z_OFFSET  = TCP_GRASP_Z + HAND_TCP_Z  # = 0.1034 m hand body offset above cube center
LIFT_Z_OFFSET   = 0.40   # EE z-offset above cube center when lifted [m]
PREGRASP_Z_OFFSET = 0.30  # EE z-offset above object for the pre-grasp hover [m]

PHASES: list[Phase] = [
    Phase("pre_grasp", duration_s=2.0, ee_offset=(0.0, 0.0, PREGRASP_Z_OFFSET), gripper_closed=False),
    Phase("reach",     duration_s=2.0, ee_offset=(0.0, 0.0, GRASP_Z_OFFSET),    gripper_closed=False),
    Phase("grasp",     duration_s=1.0, ee_offset=(0.0, 0.0, GRASP_Z_OFFSET),    gripper_closed=True),
    Phase("lift",      duration_s=3.0, ee_offset=(0.0, 0.0, LIFT_Z_OFFSET),     gripper_closed=True),
]


class ScriptedGraspController:
    """Maps simulation time to an EE target and gripper state.

    While the gripper is open, the EE target tracks the live cube position.
    When the gripper closes, the cube position is snapshotted and held fixed
    for all subsequent closed-gripper phases, so the lift target doesn't
    drift upward as the cube rises.
    """

    def __init__(self, phases: list[Phase]):
        self._phases = phases
        self._phase_end_times = np.cumsum([p.duration_s for p in phases], dtype=np.float64)
        self.total_duration = float(self._phase_end_times[-1])
        self._prev_idx: int = -1
        self._ref_pos: wp.vec3 = wp.vec3(0.0, 0.0, 0.0)

    def reset(self) -> None:
        self._prev_idx = -1
        self._ref_pos = wp.vec3(0.0, 0.0, 0.0)

    def get_target(self, sim_time: float, cube_pos: wp.vec3) -> tuple[wp.vec3, bool, str]:
        """Return (ee_pos_world, gripper_closed, phase_name) for the given sim_time [s]."""
        idx = int(np.searchsorted(self._phase_end_times, sim_time))
        idx = min(idx, len(self._phases) - 1)
        phase = self._phases[idx]

        # Snapshot cube position when the gripper transitions from open to closed.
        # While the gripper is open, track the live cube position so the arm follows
        # the cube as it settles after reset.
        if idx != self._prev_idx:
            prev = self._phases[self._prev_idx] if self._prev_idx >= 0 else None
            if phase.gripper_closed and (prev is None or not prev.gripper_closed):
                self._ref_pos = cube_pos
            self._prev_idx = idx

        ref = self._ref_pos if phase.gripper_closed else cube_pos
        return ref + wp.vec3(*phase.ee_offset), phase.gripper_closed, phase.name


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)

    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        import gymnasium as gym
        env = gym.make(args_cli.task, cfg=env_cfg)

        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")

        env.reset()

        raw_env = env.unwrapped
        if not raw_env._ik_available:
            raise RuntimeError(
                "IK not available. Pass 'env.interactive_ik=true' on the command line "
                "and ensure Newton is the physics backend."
            )

        # Target EE orientation: top-down grasp (hand +Z pointing world -Z, fingers
        # opening along world ±Y). This is a 180° rotation around world X:
        #   wp.quat(x, y, z, w) = (1, 0, 0, 0)
        # The default FK rotation (~45° forward-tilt) is NOT suitable for a top-down
        # pinch grasp — it causes the hand to approach diagonally rather than straight down.
        grasp_rot = wp.quat(1.0, 0.0, 0.0, 0.0)

        controller = ScriptedGraspController(PHASES)
        sim = raw_env.sim
        actions = torch.zeros(env.action_space.shape, device=raw_env.device)

        sim_time = 0.0
        step_dt = float(raw_env.step_dt)
        prev_phase = ""

        while True:
            # ── visualizer lifecycle ──────────────────────────────────────────
            if sim.visualizers:
                if not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                    break
                if any(v.is_running() and not v.is_closed and v.is_training_paused() for v in sim.visualizers):
                    sim.render()
                    continue

            # ── compute scripted target ───────────────────────────────────────
            # Read object position from env observations (updated each step).
            obj_pos_tensor = raw_env._object_pos
            obj_pos = wp.vec3(*obj_pos_tensor[0].tolist())
            ee_pos, gripper_closed, phase_name = controller.get_target(sim_time, obj_pos)

            if phase_name != prev_phase:
                print(f"[scripted_grasp] t={sim_time:.2f}s  phase: {prev_phase!r} → {phase_name!r}"
                      f"  object={tuple(round(float(obj_pos[i]), 3) for i in range(3))}"
                      f"  target={tuple(round(float(ee_pos[i]), 3) for i in range(3))}"
                      f"  gripper={'closed' if gripper_closed else 'open'}")
                prev_phase = phase_name

            # ── inject into the env's IK solver ──────────────────────────────
            raw_env._ee_tf = wp.transform(ee_pos, grasp_rot)
            raw_env._gripper_closed = gripper_closed

            # ── step (IK path ignores RL actions) ────────────────────────────
            with torch.inference_mode():
                env.step(actions)

            sim_time += step_dt

            # Loop the sequence — trigger env reset via the existing flag instead of
            # calling env.reset() directly (avoids inference-tensor inplace update issues).
            if sim_time > controller.total_duration:
                sim_time = 0.0
                prev_phase = ""
                controller.reset()
                print("[scripted_grasp] Sequence complete — looping.")
                raw_env._request_reset = True

        env.close()


if __name__ == "__main__":
    main()
