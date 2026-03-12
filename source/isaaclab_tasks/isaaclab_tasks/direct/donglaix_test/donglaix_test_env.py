# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .donglaix_test_env_cfg import DonglaixTestEnvCfg


class DonglaixTestEnv(DirectRLEnv):
    cfg: DonglaixTestEnvCfg

    def __init__(self, cfg: DonglaixTestEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._arm_joint_idx, _ = self.robot.find_joints(self.cfg.arm_joint_names)
        self._default_joint_pos = wp.to_torch(self.robot.data.default_joint_pos).clone()

        self.joint_pos = wp.to_torch(self.robot.data.joint_pos)
        self.joint_vel = wp.to_torch(self.robot.data.joint_vel)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        targets = self._default_joint_pos[:, self._arm_joint_idx] + self.actions * self.cfg.action_scale
        self.robot.set_joint_position_target_index(target=targets, joint_ids=self._arm_joint_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._arm_joint_idx],
                self.joint_vel[:, self._arm_joint_idx],
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        joint_pos_dev = self.joint_pos[:, self._arm_joint_idx] - self._default_joint_pos[:, self._arm_joint_idx]
        rew_alive = self.cfg.rew_scale_alive * (1.0 - self.reset_terminated.float())
        rew_terminated = self.cfg.rew_scale_terminated * self.reset_terminated.float()
        rew_joint_pos = self.cfg.rew_scale_joint_pos * torch.sum(torch.square(joint_pos_dev), dim=-1)
        rew_joint_vel = self.cfg.rew_scale_joint_vel * torch.sum(torch.abs(self.joint_vel[:, self._arm_joint_idx]), dim=-1)
        return rew_alive + rew_terminated + rew_joint_pos + rew_joint_vel

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = wp.to_torch(self.robot.data.joint_pos)
        self.joint_vel = wp.to_torch(self.robot.data.joint_vel)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Joint state with noise around default
        joint_pos = wp.to_torch(self.robot.data.default_joint_pos)[env_ids].clone()
        joint_pos[:, self._arm_joint_idx] += sample_uniform(
            -self.cfg.initial_joint_pos_noise,
            self.cfg.initial_joint_pos_noise,
            joint_pos[:, self._arm_joint_idx].shape,
            joint_pos.device,
        )
        joint_vel = wp.to_torch(self.robot.data.default_joint_vel)[env_ids].clone()

        # Root pose — use non-deprecated API (default_root_pose / default_root_vel)
        default_root_pose = wp.to_torch(self.robot.data.default_root_pose)[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        default_root_vel = wp.to_torch(self.robot.data.default_root_vel)[env_ids].clone()

        # Update cached views so observations/rewards see the reset state immediately
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        # Write to simulation
        self.robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
