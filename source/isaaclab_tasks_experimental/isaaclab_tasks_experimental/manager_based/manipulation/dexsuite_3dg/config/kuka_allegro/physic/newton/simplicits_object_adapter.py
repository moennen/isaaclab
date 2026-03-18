# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simplicits object pose adapter (Step 6).

When Simplicits is enabled, the spawn object is simulated as particles; there is no rigid body
for it in the Newton model. This adapter exposes the object pose (particle CoM per env) via the
same interface as RigidObject (.data.root_pos_w, .data.root_quat_w) so MDP (rewards, observations,
terminations, commands) and pretrained policies work unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import warp as wp

from isaaclab.assets import AssetBase, AssetBaseCfg
from isaaclab.assets.rigid_object import RigidObjectCfg

from .dexsuite_3dg_newton_manager import Dexsuite3dgNewtonManager

logger = logging.getLogger("dexsuite_3dg.simplicits.adapter")


@wp.kernel
def _fill_transform_from_pos_quat(
    pos_w: wp.array(dtype=wp.vec3f),  # type: ignore[misc]
    quat_w: wp.array(dtype=wp.quatf),  # type: ignore[misc]
    out: wp.array(dtype=wp.transformf),  # type: ignore[misc]
):
    i = wp.tid()
    out[i] = wp.transformf(pos_w[i], quat_w[i])


class _SimplicitsObjectData:
    """Minimal rigid-object data interface backed by Simplicits particle CoM.

    Exposes root_pos_w, root_quat_w (and root_link_pose_w, zero velocities),
    default_root_pose, and default_root_vel so that env.scene[\"object\"].data
    matches the RigidObject data interface used by MDP and reset events.
    """

    def __init__(
        self,
        root_pos_w: wp.array,
        root_quat_w: wp.array,
        root_link_pose_w: wp.array,
        root_com_vel_w: wp.array,
        default_root_pose: wp.array,
        default_root_vel: wp.array,
        device: str,
    ):
        self._root_pos_w = root_pos_w
        self._root_quat_w = root_quat_w
        self._root_link_pose_w = root_link_pose_w
        self._root_com_vel_w = root_com_vel_w
        self._default_root_pose = default_root_pose
        self._default_root_vel = default_root_vel
        self.device = device

    @property
    def root_pos_w(self) -> wp.array:
        """Root position in world frame; (num_envs,) wp.vec3f."""
        return self._root_pos_w

    @property
    def root_quat_w(self) -> wp.array:
        """Root orientation (x,y,z,w) in world frame; (num_envs,) wp.quatf."""
        return self._root_quat_w

    @property
    def root_link_pose_w(self) -> wp.array:
        """Root link pose (num_envs,) wp.transformf."""
        return self._root_link_pose_w

    @property
    def root_com_vel_w(self) -> wp.array:
        """Root COM velocity; zero for adapter."""
        return self._root_com_vel_w

    @property
    def root_link_vel_w(self) -> wp.array:
        """Root link velocity; zero for adapter."""
        return self._root_com_vel_w

    @property
    def root_com_pose_w(self) -> wp.array:
        """Root COM pose; same as link for single body."""
        return self._root_link_pose_w

    @property
    def default_root_pose(self) -> wp.array:
        """Default root pose for reset events; (num_instances,) wp.transformf."""
        return self._default_root_pose

    @property
    def default_root_vel(self) -> wp.array:
        """Default root velocity for reset events; (num_instances,) wp.spatial_vectorf."""
        return self._default_root_vel


class SimplicitsObjectAdapterCfg(RigidObjectCfg):
    """Config for the Simplicits object adapter. Used as scene[\"object\"] when simplicits_enabled."""

    class_type: type[SimplicitsObjectAdapter] = None  # type: ignore[assignment]


# Set after class definition
def _set_adapter_class():
    SimplicitsObjectAdapterCfg.class_type = SimplicitsObjectAdapter


class SimplicitsObjectAdapter(AssetBase):
    """Asset that exposes Simplicits spawn object pose (particle CoM) as RigidObject-like .data.

    Used when physics preset is simplicits: the scene uses this adapter instead of
    RigidObject for \"object\", so no Newton body is required and the pose is read from
    Dexsuite3dgNewtonManager.get_simplicits_object_pose() each step.
    """

    cfg: SimplicitsObjectAdapterCfg

    def __init__(self, cfg: AssetBaseCfg):
        super().__init__(cfg)
        self._num_instances = 0
        self._root_pos_w: wp.array | None = None
        self._root_quat_w: wp.array | None = None
        self._root_link_pose_w: wp.array | None = None
        self._root_com_vel_w: wp.array | None = None
        self._data: _SimplicitsObjectData | None = None
        self._root_view: Any = None

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def data(self) -> _SimplicitsObjectData:
        if self._data is None:
            raise RuntimeError("SimplicitsObjectAdapter not initialized (PHYSICS_READY not yet dispatched).")
        return self._data

    @property
    def num_bodies(self) -> int:
        return 1

    @property
    def body_names(self) -> list[str]:
        return ["object"]

    @property
    def root_view(self) -> Any:
        """Minimal view-like object for compatibility; count and link_names only."""
        return self._root_view

    def _initialize_impl(self) -> None:
        # Use Dexsuite3dgNewtonManager directly: this callback runs when PHYSICS_READY is
        # dispatched, which only happens after start_simulation(). With the simplicits scene,
        # the env patches the module so the running manager is Dexsuite3dgNewtonManager, which
        # sets _per_env_particle_ranges in _start_simulation_simplicits before dispatching.
        # sim_ctx.physics_manager can still be the base NewtonManager (from config), so do not
        # rely on it for the class.
        per_env = getattr(Dexsuite3dgNewtonManager, "_per_env_particle_ranges", None)
        if not per_env:
            raise RuntimeError(
                "[DexSuite 3DG : Newton :] SimplicitsObjectAdapter: _per_env_particle_ranges not set. "
                "Ensure simplicits_enabled and physics has been started."
            )
        self._num_instances = len(per_env)
        device = Dexsuite3dgNewtonManager.get_device()
        self._root_pos_w = wp.zeros(n=self._num_instances, dtype=wp.vec3f, device=device)
        self._root_quat_w = wp.zeros(n=self._num_instances, dtype=wp.quatf, device=device)
        self._root_link_pose_w = wp.zeros(
            n=self._num_instances,
            dtype=wp.transformf,
            device=device,
        )
        self._root_com_vel_w = wp.zeros(
            n=self._num_instances,
            dtype=wp.spatial_vectorf,
            device=device,
        )
        # Default pose/vel for reset events (e.g. reset_root_state_uniform); from cfg.init_state.
        init = self.cfg.init_state
        default_pose = np.tile(
            np.array(tuple(init.pos) + tuple(init.rot), dtype=np.float32),
            (self._num_instances, 1),
        )
        default_vel = np.tile(
            np.array(tuple(init.lin_vel) + tuple(init.ang_vel), dtype=np.float32),
            (self._num_instances, 1),
        )
        default_root_pose = wp.array(default_pose, dtype=wp.transformf, device=device)
        default_root_vel = wp.array(default_vel, dtype=wp.spatial_vectorf, device=device)
        self._data = _SimplicitsObjectData(
            root_pos_w=self._root_pos_w,
            root_quat_w=self._root_quat_w,
            root_link_pose_w=self._root_link_pose_w,
            root_com_vel_w=self._root_com_vel_w,
            default_root_pose=default_root_pose,
            default_root_vel=default_root_vel,
            device=device,
        )
        self._root_view = type(
            "_FakeView",
            (),
            {"count": self._num_instances, "link_names": ["object"]},
        )()
        logger.debug(
            "[DexSuite 3DG : Newton :] SimplicitsObjectAdapter initialized: num_envs=%s, root_pos_w/root_quat_w from particle CoM",
            self._num_instances,
        )

    def update(self, dt: float) -> None:
        pose = Dexsuite3dgNewtonManager.get_simplicits_object_pose()
        if pose is None:
            return
        if self._root_pos_w is None or self._root_quat_w is None or self._root_link_pose_w is None:
            return
        pos_w, quat_w = pose
        wp.copy(self._root_pos_w, pos_w)
        wp.copy(self._root_quat_w, quat_w)
        with wp.ScopedDevice(self._root_pos_w.device):
            wp.launch(
                _fill_transform_from_pos_quat,
                dim=self._num_instances,
                inputs=[pos_w, quat_w],
                outputs=[self._root_link_pose_w],
                device=self._root_pos_w.device,
            )
        # Sync velocity from Simplicits particle CoM so observations/rewards see plausible values.
        lin_vel_w = Dexsuite3dgNewtonManager.get_simplicits_object_velocity()
        if lin_vel_w is not None and self._root_com_vel_w is not None:
            lin_t = wp.to_torch(lin_vel_w)
            ang_t = torch.zeros_like(lin_t, device=lin_t.device, dtype=lin_t.dtype)
            spatial_t = torch.cat([lin_t, ang_t], dim=-1)
            spatial_w = wp.from_torch(spatial_t.contiguous(), dtype=wp.spatial_vectorf)
            wp.copy(self._root_com_vel_w, spatial_w)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Pose is driven by Simplicits state; no-op for adapter
        pass

    def write_data_to_sim(self) -> None:
        """No-op: object state is driven by Simplicits particles, not written from this asset."""
        pass

    def write_root_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """No-op: object pose is driven by Simplicits; reset events read default_root_pose but do not write back."""
        pass

    def write_root_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """No-op: object velocity is driven by Simplicits; reset events read default_root_vel but do not write back."""
        pass


_set_adapter_class()
