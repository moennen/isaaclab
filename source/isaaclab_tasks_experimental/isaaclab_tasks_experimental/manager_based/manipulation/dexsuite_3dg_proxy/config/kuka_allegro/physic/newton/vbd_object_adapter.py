# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD object pose adapter.

When VBD is enabled, the spawn object is simulated as particles; there is no rigid body
for it in the Newton model.  This adapter exposes the object pose (particle CoM per env)
via the same interface as RigidObject (.data.root_pos_w, .data.root_quat_w) so MDP
(rewards, observations, terminations, commands) and pretrained policies work unchanged.

Pattern mirrors ``SimplicitsObjectAdapter`` from dexsuite_3dg.
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
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from .proxy_newton_manager import Dexsuite3dgProxyNewtonManager

logger = logging.getLogger("dexsuite_3dg_proxy.vbd.adapter")


@wp.kernel
def _fill_transform_from_pos_quat(
    pos_w: wp.array(dtype=wp.vec3f),  # type: ignore[misc]
    quat_w: wp.array(dtype=wp.quatf),  # type: ignore[misc]
    out: wp.array(dtype=wp.transformf),  # type: ignore[misc]
):
    i = wp.tid()
    out[i] = wp.transformf(pos_w[i], quat_w[i])


class _VbdObjectData:
    """Minimal rigid-object data interface backed by VBD particle CoM.

    Exposes root_pos_w, root_quat_w (and root_link_pose_w, zero angular velocity)
    so that env.scene["object"].data matches the RigidObject data interface used by
    MDP and reset events.
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
        """Root position in world frame [m]; (num_envs,) wp.vec3f."""
        return self._root_pos_w

    @property
    def root_quat_w(self) -> wp.array:
        """Root orientation (x,y,z,w) in world frame; (num_envs,) wp.quatf."""
        return self._root_quat_w

    @property
    def root_link_pose_w(self) -> wp.array:
        """Root link pose; (num_envs,) wp.transformf."""
        return self._root_link_pose_w

    @property
    def root_com_vel_w(self) -> wp.array:
        """Root COM velocity (linear + zero angular); (num_envs,) wp.spatial_vectorf."""
        return self._root_com_vel_w

    @property
    def root_link_vel_w(self) -> wp.array:
        """Root link velocity; aliased to root_com_vel_w."""
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


class VbdObjectAdapterCfg(RigidObjectCfg):
    """Config for the VBD object adapter.  Used as scene["object"] when vbd_enabled."""

    class_type: type[VbdObjectAdapter] = None  # type: ignore[assignment]


def _set_adapter_class() -> None:
    VbdObjectAdapterCfg.class_type = VbdObjectAdapter


class VbdObjectAdapter(AssetBase):
    """Asset that exposes the VBD soft body pose (particle CoM) as a RigidObject-like .data.

    Used when the physics preset is deformable (vbd_enabled=True): the scene uses this
    adapter instead of RigidObject for "object", so no Newton rigid body is required for
    the ragdoll and the pose is read from
    :meth:`Dexsuite3dgProxyNewtonManager.get_object_pose` each step.
    """

    cfg: VbdObjectAdapterCfg

    def __init__(self, cfg: AssetBaseCfg):
        super().__init__(cfg)
        self._num_instances = 0
        self._root_pos_w: wp.array | None = None
        self._root_quat_w: wp.array | None = None
        self._root_link_pose_w: wp.array | None = None
        self._root_com_vel_w: wp.array | None = None
        self._data: _VbdObjectData | None = None
        self._root_view: Any = None

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def data(self) -> _VbdObjectData:
        if self._data is None:
            raise RuntimeError("VbdObjectAdapter not initialized (PHYSICS_READY not yet dispatched).")
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
        per_env = getattr(Dexsuite3dgProxyNewtonManager, "_per_env_particle_ranges", None)
        if not per_env:
            raise RuntimeError(
                "[dexsuite_3dg_proxy] VbdObjectAdapter: _per_env_particle_ranges not set. "
                "Ensure vbd_enabled=True and physics has been started."
            )
        self._num_instances = len(per_env)
        device = Dexsuite3dgProxyNewtonManager.get_device()

        self._root_pos_w = wp.zeros(n=self._num_instances, dtype=wp.vec3f, device=device)
        self._root_quat_w = wp.zeros(n=self._num_instances, dtype=wp.quatf, device=device)
        self._root_link_pose_w = wp.zeros(n=self._num_instances, dtype=wp.transformf, device=device)
        self._root_com_vel_w = wp.zeros(n=self._num_instances, dtype=wp.spatial_vectorf, device=device)

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

        self._data = _VbdObjectData(
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
            "[dexsuite_3dg_proxy] VbdObjectAdapter initialized (num_envs=%d, pose from particle CoM)",
            self._num_instances,
        )

    def update(self, dt: float) -> None:
        """Pull current VBD particle CoM pose/velocity from Newton manager each step."""
        pose = Dexsuite3dgProxyNewtonManager.get_object_pose()
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
        lin_vel_w = Dexsuite3dgProxyNewtonManager.get_object_velocity()
        if lin_vel_w is not None and self._root_com_vel_w is not None:
            lin_t = wp.to_torch(lin_vel_w)
            ang_t = torch.zeros_like(lin_t, device=lin_t.device, dtype=lin_t.dtype)
            spatial_t = torch.cat([lin_t, ang_t], dim=-1)
            spatial_w = wp.from_torch(spatial_t.contiguous(), dtype=wp.spatial_vectorf)
            wp.copy(self._root_com_vel_w, spatial_w)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Pose is driven by VBD particle state; nothing to do here.
        pass

    def write_data_to_sim(self) -> None:
        """No-op: object state is driven by VBD particles, not written from this asset."""
        pass

    def write_root_pose_to_sim_index(
        self,
        *,
        root_pose: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Teleport VBD particles to match MDP-sampled root pose (reset events).

        Args:
            root_pose: (N, 7) tensor — position (3) + quaternion xyzw (4) in world frame.
            env_ids: Env indices to reset (None = all).
        """
        root_t = wp.to_torch(root_pose).float() if not isinstance(root_pose, torch.Tensor) else root_pose.float()
        if env_ids is None:
            eid = torch.arange(self._num_instances, device=root_t.device, dtype=torch.long)
        else:
            eid = torch.as_tensor(env_ids, dtype=torch.long, device=root_t.device).reshape(-1)
        Dexsuite3dgProxyNewtonManager.reset_particles(eid, root_t)

    def write_root_velocity_to_sim_index(
        self,
        *,
        root_velocity: torch.Tensor | wp.array,
        env_ids: Sequence[int] | torch.Tensor | wp.array | None = None,
    ) -> None:
        """Zero (or set) particle velocities after pose reset.

        Args:
            root_velocity: (N, 6) tensor — linear (3) + angular (3) in world frame.
                           Angular is ignored; only linear CoM velocity is applied.
            env_ids: Env indices to reset (None = all).
        """
        if env_ids is None:
            eid = torch.arange(
                self._num_instances,
                device=root_velocity.device if isinstance(root_velocity, torch.Tensor) else "cuda:0",
                dtype=torch.long,
            )
        else:
            device = root_velocity.device if isinstance(root_velocity, torch.Tensor) else "cuda:0"
            eid = torch.as_tensor(env_ids, dtype=torch.long, device=device).reshape(-1)
        Dexsuite3dgProxyNewtonManager.reset_particle_velocities(eid)


_set_adapter_class()


# ---------------------------------------------------------------------------
# VBD-native MDP observation and reward helpers
# ---------------------------------------------------------------------------

class object_point_cloud_b_vbd(ManagerTermBase):
    """Object point cloud from VBD particle positions in the robot base frame.

    Replaces ``object_point_cloud_b`` (which samples USD prim geometry) when the
    object is a VBD soft body with no USD prim.  Samples ``num_points`` particles
    uniformly from the VBD particle set for each environment.

    The output shape is identical to ``object_point_cloud_b`` so the observation
    term is a drop-in replacement.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from isaaclab.assets import Articulation  # noqa: PLC0415
        self._ref_asset = env.scene[cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot")).name]
        self._num_points = cfg.params.get("num_points", 10)

    def __call__(
        self,
        env,
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        num_points: int = 10,
        flatten: bool = False,
        visualize: bool = False,
    ) -> torch.Tensor:
        ranges = Dexsuite3dgProxyNewtonManager._per_env_particle_ranges
        state = Dexsuite3dgProxyNewtonManager._state_0
        num_envs = env.num_envs

        if ranges is None or state is None or state.particle_q is None:
            zeros = torch.zeros(num_envs, self._num_points, 3, device=env.device)
            return zeros.view(num_envs, -1) if flatten else zeros

        pq = wp.to_torch(state.particle_q).float()
        ref_pos_w = wp.to_torch(self._ref_asset.data.root_pos_w).float()    # (num_envs, 3)
        ref_quat_w = wp.to_torch(self._ref_asset.data.root_quat_w).float()  # (num_envs, 4)

        points_w = torch.zeros(num_envs, self._num_points, 3, device=pq.device)
        for e in range(num_envs):
            start, end = ranges[e]
            n = end - start
            if n == 0:
                continue
            # Uniform random subsample of particles
            idx = torch.randperm(n, device=pq.device)[:self._num_points]
            if idx.shape[0] < self._num_points:
                idx = idx.repeat(self._num_points // idx.shape[0] + 1)[:self._num_points]
            points_w[e] = pq[start:end][idx]

        from isaaclab.utils.math import subtract_frame_transforms  # noqa: PLC0415
        ref_pos = ref_pos_w.unsqueeze(1).expand(-1, self._num_points, -1)
        ref_quat = ref_quat_w.unsqueeze(1).expand(-1, self._num_points, -1)
        points_b, _ = subtract_frame_transforms(ref_pos, ref_quat, points_w, None)
        return points_b.view(num_envs, -1) if flatten else points_b


def fingers_contact_force_b_vbd(
    env,
    fingertip_names: list[str],
    contact_threshold: float,
    signal_magnitude: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Particle-proximity contact signal for the robot fingertips (VBD mode).

    Replaces the PhysX ``ContactSensor``-based ``fingers_contact_force_b`` when the
    object is a VBD soft body.  There is no rigid body for the object in Newton, so
    PhysX / Newton contact sensors cannot detect finger↔ragdoll contact.

    Instead, for each fingertip we compute the minimum Euclidean distance to any
    particle that belongs to the soft object in the same environment.  If the distance
    is below ``contact_threshold`` the fingertip is considered to be in contact and a
    synthetic "force" vector of magnitude ``signal_magnitude`` is returned (z-component
    only, consistent with the axis used by ``fingers_contact_force_b``).

    The output shape and semantics are identical to ``fingers_contact_force_b`` so the
    observation term is a drop-in replacement.

    Args:
        env: The environment instance.
        fingertip_names: Ordered list of robot link names to check (e.g.
            ``["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]``).
        contact_threshold: Distance [m] below which a particle counts as contact.
            Use ``particle_radius * 2`` (default ``particle_radius = 0.005 m``).
        signal_magnitude: Magnitude of the returned synthetic force when in contact [N].
            Match the ``clip`` range of the observation group (default 1.0).
        asset_cfg: Scene entity for the robot.  Defaults to ``SceneEntityCfg("robot")``.

    Returns:
        Tensor of shape ``(num_envs, 3 * len(fingertip_names))``.
    """
    from isaaclab.assets import Articulation  # noqa: PLC0415

    robot: Articulation = env.scene[asset_cfg.name]

    # Cache tip_ids as a GPU tensor so fancy indexing never forces a CPU↔GPU sync.
    cache_key = (id(robot), tuple(fingertip_names))
    if not hasattr(fingers_contact_force_b_vbd, "_tip_ids_cache"):
        fingers_contact_force_b_vbd._tip_ids_cache = {}
    if cache_key not in fingers_contact_force_b_vbd._tip_ids_cache:
        body_names: list[str] = robot.body_names
        tip_ids = []
        for name in fingertip_names:
            matches = [i for i, n in enumerate(body_names) if n == name]
            if not matches:
                raise ValueError(
                    f"[VBD contact obs] Fingertip '{name}' not found in robot body_names. "
                    f"Available: {body_names}"
                )
            tip_ids.append(matches[0])
        fingers_contact_force_b_vbd._tip_ids_cache[cache_key] = torch.tensor(
            tip_ids, dtype=torch.long, device=robot.device
        )
    tip_ids_t = fingers_contact_force_b_vbd._tip_ids_cache[cache_key]

    body_pos_w = wp.to_torch(robot.data.body_pos_w)              # (num_envs, num_bodies, 3)
    tip_pos_w = body_pos_w[:, tip_ids_t, :].float()              # (num_envs, num_tips, 3)

    force_proxy = Dexsuite3dgProxyNewtonManager.get_fingertip_contact_proxy(
        tip_pos_w,
        contact_threshold=contact_threshold,
        signal_magnitude=signal_magnitude,
    )  # (num_envs, num_tips, 3)

    root_quat_w = wp.to_torch(robot.data.root_link_quat_w).float()  # (num_envs, 4)
    forces_b = quat_apply_inverse(
        root_quat_w.unsqueeze(1).expand(-1, len(fingertip_names), -1),
        force_proxy,
    )  # (num_envs, num_tips, 3)
    return forces_b.reshape(env.num_envs, -1)


def contacts_vbd(
    env,
    threshold: float,
    thumb_name: str,
    finger_names: list[str],
    contact_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Good-contact boolean: thumb + at least one finger touching particles (VBD mode).

    Drop-in replacement for ``contacts`` reward function when the object is a VBD soft body.

    Args:
        env: The environment instance.
        threshold: Magnitude threshold — API symmetry with ``contacts``; in VBD mode
            a contact either fires (1.0) or not (0.0), so any threshold < 1.0 works.
        thumb_name: Robot link name of the thumb fingertip.
        finger_names: Robot link names of the other fingertips.
        contact_threshold: Distance [m] below which a particle counts as contact.
        asset_cfg: Scene entity for the robot.

    Returns:
        Boolean tensor ``(num_envs,)``.
    """
    all_tips = [thumb_name] + list(finger_names)
    force_b = fingers_contact_force_b_vbd(
        env,
        fingertip_names=all_tips,
        contact_threshold=contact_threshold,
        signal_magnitude=1.0,
        asset_cfg=asset_cfg,
    ).view(env.num_envs, len(all_tips), 3)
    magnitudes = torch.linalg.norm(force_b, dim=-1)   # (num_envs, num_tips)
    thumb_contact = magnitudes[:, 0] > threshold
    any_finger = (magnitudes[:, 1:] > threshold).any(dim=-1)
    return thumb_contact & any_finger


def object_ee_distance_vbd(
    env,
    std: float,
    thumb_name: str,
    finger_names: list[str],
    contact_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """VBD version of ``object_ee_distance``: uses particle-proximity for contact gate."""
    from isaaclab.assets import RigidObject  # noqa: PLC0415

    robot = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    asset_pos = wp.to_torch(robot.data.body_pos_w)[:, asset_cfg.body_ids]
    object_pos = wp.to_torch(obj.data.root_pos_w)
    distance = torch.linalg.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    contact_bonus = contacts_vbd(
        env, threshold=0.5, thumb_name=thumb_name, finger_names=finger_names,
        contact_threshold=contact_threshold, asset_cfg=asset_cfg,
    ).float().clamp(0.1, 1.0)
    return ((1 - torch.tanh(distance / std)) * contact_bonus).nan_to_num_(nan=0.0)


def position_command_error_tanh_vbd(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    thumb_name: str,
    finger_names: list[str],
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """VBD version of ``position_command_error_tanh``."""
    from isaaclab.assets import RigidObject  # noqa: PLC0415
    from isaaclab.utils.math import combine_frame_transforms  # noqa: PLC0415

    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        wp.to_torch(asset.data.root_pos_w),
        wp.to_torch(asset.data.root_quat_w),
        des_pos_b,
    )
    distance = torch.linalg.norm(wp.to_torch(obj.data.root_pos_w) - des_pos_w, dim=1)
    contact = contacts_vbd(
        env, threshold=0.5, thumb_name=thumb_name, finger_names=finger_names,
        contact_threshold=contact_threshold, asset_cfg=asset_cfg,
    ).float()
    return ((1 - torch.tanh(distance / std)) * contact).nan_to_num_(nan=0.0)


def orientation_command_error_tanh_vbd(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    thumb_name: str,
    finger_names: list[str],
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """VBD version of ``orientation_command_error_tanh``."""
    from isaaclab.assets import RigidObject  # noqa: PLC0415
    from isaaclab.utils import math as math_utils  # noqa: PLC0415

    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    root_state = wp.to_torch(asset.data.root_state_w)
    des_quat_w = math_utils.quat_mul(root_state[:, 3:7], des_quat_b)
    quat_distance = math_utils.quat_error_magnitude(wp.to_torch(obj.data.root_quat_w), des_quat_w)
    contact = contacts_vbd(
        env, threshold=0.5, thumb_name=thumb_name, finger_names=finger_names,
        contact_threshold=contact_threshold, asset_cfg=asset_cfg,
    ).float()
    return ((1 - torch.tanh(quat_distance / std)) * contact).nan_to_num_(nan=0.0)


class success_reward_vbd(ManagerTermBase):
    """VBD version of ``success_reward``."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.succeeded[:] = False
        else:
            self.succeeded[env_ids] = False

    def __call__(
        self,
        env,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        align_asset_cfg: SceneEntityCfg,
        pos_std: float,
        thumb_name: str,
        finger_names: list[str],
        contact_threshold: float = 0.01,
        rot_std: float | None = None,
    ) -> torch.Tensor:
        from isaaclab.assets import RigidObject  # noqa: PLC0415
        from isaaclab.utils.math import combine_frame_transforms, compute_pose_error  # noqa: PLC0415

        asset: RigidObject = env.scene[asset_cfg.name]
        obj: RigidObject = env.scene[align_asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, des_quat_w = combine_frame_transforms(
            wp.to_torch(asset.data.root_pos_w),
            wp.to_torch(asset.data.root_quat_w),
            command[:, :3],
            command[:, 3:7],
        )
        pos_err, rot_err = compute_pose_error(
            des_pos_w, des_quat_w,
            wp.to_torch(obj.data.root_pos_w),
            wp.to_torch(obj.data.root_quat_w),
        )
        pos_dist = torch.linalg.norm(pos_err, dim=1)
        contact_mask = contacts_vbd(
            env, threshold=0.5, thumb_name=thumb_name, finger_names=finger_names,
            contact_threshold=contact_threshold, asset_cfg=asset_cfg,
        )
        if rot_std:
            rot_dist = torch.linalg.norm(rot_err, dim=1)
            reward = (1 - torch.tanh(pos_dist / pos_std)) * (1 - torch.tanh(rot_dist / rot_std)) * contact_mask.float()
            self.succeeded |= contact_mask & (pos_dist < pos_std) & (rot_dist < rot_std)
        else:
            reward = ((1 - torch.tanh(pos_dist / pos_std)) ** 2) * contact_mask.float()
            self.succeeded |= contact_mask & (pos_dist < pos_std)
        return reward.nan_to_num_(nan=0.0)


def contact_count_vbd(
    env,
    threshold: float,
    fingertip_names: list[str],
    contact_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fraction of fingertips touching particles (VBD mode).

    Drop-in replacement for ``contact_count`` reward function.

    Args:
        env: The environment instance.
        threshold: Magnitude threshold — see :func:`contacts_vbd`.
        fingertip_names: All fingertip link names to check.
        contact_threshold: Distance [m] below which a particle counts as contact.
        asset_cfg: Scene entity for the robot.

    Returns:
        Tensor ``(num_envs,)`` with fraction of fingertips in contact.
    """
    force_b = fingers_contact_force_b_vbd(
        env,
        fingertip_names=fingertip_names,
        contact_threshold=contact_threshold,
        signal_magnitude=1.0,
        asset_cfg=asset_cfg,
    ).view(env.num_envs, len(fingertip_names), 3)
    magnitudes = torch.linalg.norm(force_b, dim=-1)
    return (magnitudes > threshold).float().mean(dim=-1)
