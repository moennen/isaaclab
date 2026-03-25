# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drop-cloth environment: a T-shirt falls under gravity onto the ground using the VBD solver."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets.deformable_object import DeformableObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .drop_cloth_env_cfg import DropClothEnvCfg

logger = logging.getLogger(__name__)


class DropClothEnv(DirectRLEnv):
    cfg: DropClothEnvCfg

    def __init__(self, cfg: DropClothEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        # Cloth asset (triangle surface mesh)
        self.cloth = DeformableObject(self.cfg.cloth)
        # Soft cuboid (volumetric tet mesh from PhysX tetrahedralization)
        self.cube = DeformableObject(self.cfg.cube)

        # Ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Clone environments (no robots to replicate)
        self.scene.clone_environments(copy_from_source=False)

    # ─── RL interface (no-op — demo only) ────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        pass

    def _apply_action(self) -> None:
        pass

    def _get_observations(self) -> dict:
        self.cloth.update(self.step_dt)
        self.cube.update(self.step_dt)
        return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)
        self.cloth.reset(env_ids=env_ids)
        self.cube.reset(env_ids=env_ids)
