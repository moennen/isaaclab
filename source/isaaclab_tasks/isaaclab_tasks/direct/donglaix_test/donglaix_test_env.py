# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
import warp as wp

from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.shapes import SphereCfg, spawn_sphere
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import sample_uniform

from .donglaix_test_env_cfg import DonglaixTestEnvCfg

logger = logging.getLogger(__name__)

# Prim path for the draggable IK target sphere
_SPHERE_PRIM_PATH = "/World/ik_target"


class DonglaixTestEnv(DirectRLEnv):
    cfg: DonglaixTestEnvCfg

    def __init__(self, cfg: DonglaixTestEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._arm_joint_idx, _ = self.robot.find_joints(self.cfg.arm_joint_names)
        self._default_joint_pos = wp.to_torch(self.robot.data.default_joint_pos).clone()

        self.joint_pos = wp.to_torch(self.robot.data.joint_pos)
        self.joint_vel = wp.to_torch(self.robot.data.joint_vel)

        # ------------------------------------------------------------------
        # Optional IK setup — Newton backend only
        # ------------------------------------------------------------------
        self._ik_available = False
        try:
            import newton
            import newton.ik as ik
            from isaaclab_newton.physics import NewtonManager

            newton_model = NewtonManager._model
            if newton_model is not None:
                ee_body_idx, _ = self.robot.find_bodies("panda_hand")
                self._ee_index = int(ee_body_idx[0])

                # Compute initial EE transform via FK
                ik_state = newton_model.state()
                newton.eval_fk(newton_model, newton_model.joint_q, newton_model.joint_qd, ik_state)
                body_q_np = ik_state.body_q.numpy()
                self._ee_tf = wp.transform(*body_q_np[self._ee_index])
                ee_pos = wp.transform_get_translation(self._ee_tf)

                # IK objectives
                self._pos_obj = ik.IKObjectivePosition(
                    link_index=self._ee_index,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=wp.array([ee_pos], dtype=wp.vec3),
                )
                self._joint_limit_obj = ik.IKObjectiveJointLimit(
                    joint_limit_lower=newton_model.joint_limit_lower,
                    joint_limit_upper=newton_model.joint_limit_upper,
                    weight=0.0,
                )

                # Joint config buffer for IK: shape (1 problem, joint_coord_count)
                self._ik_joint_q = wp.array(newton_model.joint_q, shape=(1, newton_model.joint_coord_count))

                self._ik_solver = ik.IKSolver(
                    model=newton_model,
                    n_problems=1,
                    objectives=[self._pos_obj, self._joint_limit_obj],
                    jacobian_mode=ik.IKJacobianType.ANALYTIC,
                )

                self._newton_model = newton_model
                self._ik_available = True
                logger.info("[DonglaixTestEnv] Newton IK initialized (EE index=%d)", self._ee_index)
        except Exception as exc:
            logger.info("[DonglaixTestEnv] IK not available: %s", exc)

        # ------------------------------------------------------------------
        # USD stage + IK target sphere setup
        # ------------------------------------------------------------------
        self._stage = get_current_stage()
        self._sphere_prim = self._stage.GetPrimAtPath(_SPHERE_PRIM_PATH)

        # Teleport sphere to actual EE position if IK is available
        if self._ik_available:
            ee_pos = wp.transform_get_translation(self._ee_tf)
            xform = UsdGeom.Xformable(self._sphere_prim)
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])))
                    break

        # Detect Omniverse mode: if omni.usd is importable the full Isaac Sim stack is active
        # and the user can drag the sphere in the Omniverse viewport.
        self._omniverse_mode = False
        try:
            import omni.usd  # noqa: F401

            self._omniverse_mode = True
        except ImportError:
            pass

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cylinder = RigidObject(self.cfg.cylinder_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        # filter_collisions is PhysX-specific; Newton handles collision groups
        # via the builder architecture (global vs per-env add_usd calls).
        if "physx" in self.scene.physics_backend:
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Spawn a visual-only red sphere as the draggable IK target.
        # In Omniverse mode: drag it in the viewport to move the IK target.
        # In Newton mode: it tracks the current IK target position passively.
        spawn_sphere(
            _SPHERE_PRIM_PATH,
            SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            translation=(0.5, 0.0, 0.5),  # placeholder — moved to EE pos in __init__
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        if self._ik_available:
            if self._omniverse_mode:
                # Omniverse mode: read sphere world position (user drags sphere in viewport)
                xform = UsdGeom.Xformable(self._sphere_prim)
                tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                pos = tf.ExtractTranslation()
                target_pos = wp.vec3(float(pos[0]), float(pos[1]), float(pos[2]))
                # Sync _ee_tf translation to match sphere (in-place update preserves the object identity
                # so the gizmo's stored reference stays valid)
                self._ee_tf[:] = wp.transform(target_pos, wp.transform_get_rotation(self._ee_tf))
            else:
                # Newton / headless mode: register a draggable gizmo so the user can
                # move the IK target in the Newton OpenGL viewer.  log_gizmo() mutates
                # self._ee_tf in-place via ImGuizmo when the user drags the handle,
                # so the updated position is picked up on the next physics step.
                if not hasattr(self, "_newton_viewer_gl"):
                    self._newton_viewer_gl = None
                    try:
                        from isaaclab_visualizers.newton import NewtonVisualizer

                        for v in self.sim.visualizers:
                            if isinstance(v, NewtonVisualizer) and v._viewer is not None:
                                self._newton_viewer_gl = v._viewer
                                break
                    except Exception:
                        pass

                    if self._newton_viewer_gl is not None:
                        # begin_frame() clears _gizmo_log each render step, so log_gizmo()
                        # called during the physics step is erased before _render_gizmos() runs.
                        # Patch begin_frame to re-register the gizmo right after the reset.
                        _orig_bf = self._newton_viewer_gl.begin_frame
                        _tf = self._ee_tf  # scalar wp.transform captured by reference
                        _viewer = self._newton_viewer_gl

                        def _begin_frame_with_gizmo(time, _orig=_orig_bf, _v=_viewer, _t=_tf):
                            _orig(time)
                            _v._gizmo_log["ik_target"] = _t

                        self._newton_viewer_gl.begin_frame = _begin_frame_with_gizmo
                        logger.info("[DonglaixTestEnv] Newton viewer gizmo registered")
                    else:
                        logger.warning("[DonglaixTestEnv] NewtonViewerGL not found in sim.visualizers")

                if self._newton_viewer_gl is not None:
                    # Render a red sphere at the IK target position in the Newton viewport.
                    # The draggable gizmo handle is injected via the begin_frame patch above.
                    device = self._newton_viewer_gl.device
                    target_pos_vec = wp.transform_get_translation(self._ee_tf)
                    self._newton_viewer_gl.log_points(
                        "ik_target_sphere",
                        points=wp.array([target_pos_vec], dtype=wp.vec3, device=device),
                        radii=wp.array([0.05], dtype=wp.float32, device=device),
                        colors=wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device),
                    )

                # Sync sphere prim to the (possibly gizmo-updated) target position
                target_pos = wp.transform_get_translation(self._ee_tf)
                xform = UsdGeom.Xformable(self._sphere_prim)
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        op.Set(Gf.Vec3d(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])))
                        break

            # Warm-start IK from current joint state
            current_q = wp.to_torch(self.robot.data.joint_pos)[0]
            ik_q_torch = wp.to_torch(self._ik_joint_q)
            n_robot = current_q.shape[0]
            ik_q_torch[0, :n_robot] = current_q[:n_robot]

            self._pos_obj.set_target_position(0, wp.transform_get_translation(self._ee_tf))
            self._ik_solver.step(self._ik_joint_q, self._ik_joint_q, iterations=24)

            solved_arm_q = wp.to_torch(self._ik_joint_q)[0, self._arm_joint_idx]
            self.robot.set_joint_position_target_index(target=solved_arm_q.unsqueeze(0), joint_ids=self._arm_joint_idx)
        else:
            # IK not available: action-driven position offsets
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
        rew_joint_vel = self.cfg.rew_scale_joint_vel * torch.sum(
            torch.abs(self.joint_vel[:, self._arm_joint_idx]), dim=-1
        )
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

        # Root pose — world frame (default is local env frame, add env origins)
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

        # Place cylinder 0.5 m above the EE (FK in Newton mode, fixed offset otherwise)
        if self._ik_available:
            import newton

            ik_state = self._newton_model.state()
            newton.eval_fk(self._newton_model, self._newton_model.joint_q, self._newton_model.joint_qd, ik_state)
            body_q_np = ik_state.body_q.numpy()
            ee_pos = body_q_np[self._ee_index][:3]
            cylinder_pos = torch.tensor(
                [[ee_pos[0], ee_pos[1], ee_pos[2] + 0.5]], dtype=torch.float32, device=self.device
            ).expand(len(env_ids), -1)
        else:
            cylinder_pos = torch.tensor([[0.5, 0.0, 1.5]], dtype=torch.float32, device=self.device).expand(
                len(env_ids), -1
            )

        cylinder_orient = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device).expand(
            len(env_ids), -1
        )
        cylinder_pose = torch.cat([cylinder_pos, cylinder_orient], dim=-1)
        cylinder_vel = torch.zeros(len(env_ids), 6, dtype=torch.float32, device=self.device)

        env_ids_list = env_ids.cpu().tolist() if hasattr(env_ids, "cpu") else list(env_ids)
        self.cylinder.reset(env_ids=env_ids_list)
        self.cylinder.write_root_pose_to_sim_index(root_pose=cylinder_pose, env_ids=env_ids)
        self.cylinder.write_root_velocity_to_sim_index(root_velocity=cylinder_vel, env_ids=env_ids)
