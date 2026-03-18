# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drop-cloth environment: a T-shirt falls under gravity onto the ground using the VBD solver."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
import warp as wp
from pxr import Usd, UsdGeom

import newton
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .drop_cloth_env_cfg import DropClothEnvCfg

import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# ─── cloth mesh asset ────────────────────────────────────────────────────────
_SHIRT_USD = os.path.join(
    os.path.dirname(newton.__file__),
    "examples",
    "assets",
    "unisex_shirt.usd",
)

# Cloth simulation parameters (meter scale)
# Reference: newton/examples/cloth/example_cloth_franka.py (same mesh, cm-space values)
_CLOTH_SCALE = 0.01  # USD vertices are in cm → convert to meters
_TRI_KE = 1e4  # area-preserving stiffness
_TRI_KA = 1e4  # area stiffness
_TRI_KD = 1.5e-6  # area damping (must be small — high value causes explosion)
_BENDING_KE = 5.0  # bending stiffness
_BENDING_KD = 1e-2  # bending damping
_PARTICLE_RADIUS = 0.008  # [m] (= 0.8 cm, matches reference particle_radius=0.8)
_SOFT_CONTACT_KE = 1e4  # body–particle contact stiffness
_SOFT_CONTACT_KD = 1e-2  # body–particle contact damping


class DropClothEnv(DirectRLEnv):
    cfg: DropClothEnvCfg

    def __init__(self, cfg: DropClothEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        # Register a MODEL_INIT callback to inject the cloth into the Newton builder
        # before it is finalized, and a PHYSICS_READY callback to configure the model.
        from isaaclab_newton.physics import NewtonManager

        NewtonManager.register_callback(self._add_cloth_to_newton_builder, PhysicsEvent.MODEL_INIT)
        NewtonManager.register_callback(self._configure_model, PhysicsEvent.PHYSICS_READY)

        # Ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Clone environments (no robots to replicate)
        self.scene.clone_environments(copy_from_source=False)

    # ─── Newton callbacks ─────────────────────────────────────────────────────

    def _add_cloth_to_newton_builder(self, payload=None) -> None:
        """Add the T-shirt cloth mesh to the Newton ModelBuilder.

        Called by NewtonManager just before ``builder.finalize()``.
        Reads the unisex shirt from its bundled USD asset, scales the vertices
        from cm to meters, rotates the shirt so its height axis aligns with the
        simulation's z-up world, and calls ``builder.color()`` to build the
        vertex colouring required by the VBD solver.
        """
        from isaaclab_newton.physics import NewtonManager

        builder = NewtonManager._builder
        if builder is None:
            return

        # Load the mesh from USD using OpenUSD directly
        usd_stage = Usd.Stage.Open(_SHIRT_USD)
        usd_prim = usd_stage.GetPrimAtPath("/root/shirt")
        mesh = UsdGeom.Mesh(usd_prim)

        # pts_cm = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
        # indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        # # Convert vertices from cm (USD) to meters
        # vertices = [wp.vec3(float(p[0] * _CLOTH_SCALE), float(p[1] * _CLOTH_SCALE), float(p[2] * _CLOTH_SCALE)) for p in pts_cm]

        shirt_mesh = newton.usd.get_mesh(usd_prim)
        mesh_points = shirt_mesh.vertices
        mesh_indices = shirt_mesh.indices
        vertices = [wp.vec3(v * _CLOTH_SCALE) for v in mesh_points]

        # Rotate -90° around x so the shirt's y-up axis maps to z-up (gravity is -z).
        # After this rotation the shirt occupies z ≈ [0.88, 1.52] m above the ground.
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi)

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, self.cfg.cloth_drop_height),
            rot=rot,
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=mesh_indices,
            density=0.02,
            tri_ke=_TRI_KE,
            tri_ka=_TRI_KA,
            tri_kd=_TRI_KD,
            edge_ke=_BENDING_KE,
            edge_kd=_BENDING_KD,
            particle_radius=_PARTICLE_RADIUS,
        )

        # Build vertex colouring required by VBD solver
        builder.color()

    def _configure_model(self, payload=None) -> None:
        """Configure the Newton model after finalization.

        Called by NewtonManager after ``builder.finalize()`` and initial FK.
        Zeroes the edge rest angles (flat rest state) and sets soft-contact
        stiffness parameters for realistic body–cloth interaction.
        Also snapshots the initial particle state for use in episode resets.
        """
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager._model
        if model is None or not hasattr(model, "edge_rest_angle"):
            return

        model.edge_rest_angle.zero_()
        model.soft_contact_ke = _SOFT_CONTACT_KE
        model.soft_contact_kd = _SOFT_CONTACT_KD

        # Snapshot initial particle positions (after finalize + FK) for reset.
        # Both state_0 and state_1 exist at this point (created in start_simulation).
        state = NewtonManager._state_0
        if state is not None and hasattr(state, "particle_q") and state.particle_q is not None:
            self._init_particle_q = wp.clone(state.particle_q)

        # Create a UsdGeom.Mesh prim so Kit's RTX viewport can render the cloth.
        # The prim lives at /World/cloth_vis (outside env namespaces so it isn't cloned).
        self._create_cloth_vis_prim(model)

    def _create_cloth_vis_prim(self, model) -> None:
        """Create a UsdGeom.Mesh prim at /World/cloth_vis for Kit viewport rendering.

        The prim is authored once with the cloth topology; its ``points`` attribute
        is updated every render step from the Newton particle state via
        :meth:`_update_cloth_vis`.
        """
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Gf, Sdf, Vt

        stage = get_current_stage()
        prim_path = "/World/cloth_vis"

        # Build face topology from model.tri_indices (Nx3 flat array on CPU).
        tri_idx = model.tri_indices.numpy()  # shape (num_tris * 3,) or (num_tris, 3)
        tri_idx = tri_idx.reshape(-1, 3)
        face_vertex_indices = Vt.IntArray(tri_idx.flatten().tolist())
        face_vertex_counts = Vt.IntArray([3] * len(tri_idx))

        # Initial vertex positions from particle state.
        from isaaclab_newton.physics import NewtonManager

        pts_np = NewtonManager._state_0.particle_q.numpy()  # (N, 3)
        points = Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts_np])

        mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(prim_path))
        mesh.GetPointsAttr().Set(points)
        mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
        mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
        mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

        self._cloth_vis_prim = mesh
        self._cloth_particle_count = len(pts_np)

    def _update_cloth_vis(self) -> None:
        """Write current Newton particle positions into the Kit cloth mesh prim."""
        if not hasattr(self, "_cloth_vis_prim"):
            return
        from isaaclab_newton.physics import NewtonManager
        from pxr import Gf, Vt

        state = NewtonManager._state_0
        if state is None or state.particle_q is None:
            return
        pts_np = state.particle_q.numpy()
        points = Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts_np])
        self._cloth_vis_prim.GetPointsAttr().Set(points)

    # ─── RL interface (no-op — demo only) ────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        pass

    def _apply_action(self) -> None:
        pass

    def _get_observations(self) -> dict:
        self._update_cloth_vis()
        return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        logger.info("episode_length_buf: %d, max_episode_length: %d", self.episode_length_buf, self.max_episode_length)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)

        if not hasattr(self, "_init_particle_q"):
            return

        from isaaclab_newton.physics import NewtonManager

        # Reset particle positions and velocities in both states.
        for state in (NewtonManager._state_0, NewtonManager._state_1):
            if state is None:
                continue
            if state.particle_q is not None:
                wp.copy(state.particle_q, self._init_particle_q)
            if state.particle_qd is not None:
                state.particle_qd.zero_()

        # Zero VBD solver's internal particle buffers so the next step
        # doesn't carry over stale inertial targets or displacements.
        solver = NewtonManager._solver
        if solver is not None:
            for attr in ("particle_q_prev", "inertia", "pos_prev_collision_detection", "particle_displacements"):
                buf = getattr(solver, attr, None)
                if buf is not None:
                    buf.zero_()
            if getattr(solver, "truncation_ts", None) is not None:
                solver.truncation_ts.fill_(1.0)
