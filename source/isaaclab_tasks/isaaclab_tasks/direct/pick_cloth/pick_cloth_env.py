# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-Cloth environment: Franka robot interacts with cloth using a coupled solver."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.assets.deformable_object import DeformableObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .pick_cloth_env_cfg import PickClothEnvCfg

logger = logging.getLogger(__name__)


class PickClothEnv(DirectRLEnv):
    cfg: PickClothEnvCfg

    def __init__(self, cfg: PickClothEnvCfg, render_mode: str | None = None, **kwargs):
        # For velocity control, override actuator gains before the robot is spawned:
        # zero stiffness (no position tracking), high damping (velocity-tracking gain).
        # Featherstone torque: tau = ke*(pos_target - q) + kd*(vel_target - qd)
        # With ke=0: tau = kd*(vel_target - qd)  — proportional velocity control.
        if cfg.control_mode == "velocity":
            for actuator in cfg.robot_cfg.actuators.values():
                actuator.stiffness = 0.0
                actuator.damping = 200.0

        super().__init__(cfg, render_mode, **kwargs)

        self._arm_joint_idx, _ = self.robot.find_joints(self.cfg.arm_joint_names)
        self._default_joint_pos = wp.to_torch(self.robot.data.default_joint_pos).clone()

        self.joint_pos = wp.to_torch(self.robot.data.joint_pos)
        self.joint_vel = wp.to_torch(self.robot.data.joint_vel)

        # Find EE body index for reward computation
        ee_body_idx, _ = self.robot.find_bodies("panda_hand")
        self._ee_body_idx = int(ee_body_idx[0])

        logger.info("PickClothEnv: control_mode=%s, action_scale=%s", self.cfg.control_mode, cfg.action_scale)

    def _setup_scene(self):
        # Robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # Cloth asset (triangle surface mesh)
        self.cloth = DeformableObject(self.cfg.cloth)

        # Ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

    # ─── RL interface ────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        if self.cfg.control_mode == "velocity":
            # Velocity control: actions are target joint velocities [rad/s]
            vel_targets = self.actions * self.cfg.action_scale
            self.robot.set_joint_velocity_target_index(target=vel_targets, joint_ids=self._arm_joint_idx)
        else:
            # Position control: actions are offsets from default pose [rad]
            pos_targets = self._default_joint_pos[:, self._arm_joint_idx] + self.actions * self.cfg.action_scale
            self.robot.set_joint_position_target_index(target=pos_targets, joint_ids=self._arm_joint_idx)

    def _get_observations(self) -> dict:
        self.cloth.update(self.step_dt)

        # Cloth centroid: mean of all nodal positions
        # nodal_pos_w is a wp.array of shape (num_envs, num_particles) with dtype vec3f
        nodal_pos = wp.to_torch(self.cloth.data.nodal_pos_w)  # (num_envs, num_particles, 3)
        self._cloth_centroid = nodal_pos.mean(dim=1)  # (num_envs, 3)

        obs = torch.cat(
            (
                self.joint_pos[:, self._arm_joint_idx],   # (num_envs, 7)
                self.joint_vel[:, self._arm_joint_idx],   # (num_envs, 7)
                self._cloth_centroid,                      # (num_envs, 3)
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Cloth height reward — encourage lifting
        cloth_height = self._cloth_centroid[:, 2]  # z-component
        rew_cloth_height = self.cfg.rew_scale_cloth_height * cloth_height

        # EE-to-cloth distance penalty — encourage reaching toward cloth
        ee_pos = wp.to_torch(self.robot.data.body_pos_w)[:, self._ee_body_idx]  # (num_envs, 3)
        ee_cloth_dist = torch.norm(ee_pos - self._cloth_centroid, dim=-1)
        rew_ee_cloth_dist = self.cfg.rew_scale_ee_cloth_dist * ee_cloth_dist

        # Joint velocity penalty
        rew_joint_vel = self.cfg.rew_scale_joint_vel * torch.sum(
            torch.abs(self.joint_vel[:, self._arm_joint_idx]), dim=-1
        )

        return rew_cloth_height + rew_ee_cloth_dist + rew_joint_vel

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = wp.to_torch(self.robot.data.joint_pos)
        self.joint_vel = wp.to_torch(self.robot.data.joint_vel)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)

        # Reset robot joint state to defaults
        joint_pos = wp.to_torch(self.robot.data.default_joint_pos)[env_ids].clone()
        joint_vel = wp.to_torch(self.robot.data.default_joint_vel)[env_ids].clone()

        # Root pose — add env origins for world frame
        default_root_pose = wp.to_torch(self.robot.data.default_root_pose)[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        default_root_vel = wp.to_torch(self.robot.data.default_root_vel)[env_ids].clone()

        # Update cached views
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        # Write robot state to simulation
        self.robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        # Reset cloth to initial nodal positions
        env_ids_list = env_ids.cpu().tolist() if hasattr(env_ids, "cpu") else list(env_ids)
        self.cloth.reset(env_ids=env_ids_list)
