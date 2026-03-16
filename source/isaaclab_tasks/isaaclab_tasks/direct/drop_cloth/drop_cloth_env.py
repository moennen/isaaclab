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

import newton as nwt
from isaaclab.envs import DirectRLEnv
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .drop_cloth_env_cfg import DropClothEnvCfg

# ─── cloth mesh asset ────────────────────────────────────────────────────────
_SHIRT_USD = os.path.join(
    os.path.dirname(nwt.__file__),
    "examples",
    "assets",
    "unisex_shirt.usd",
)

# Cloth simulation parameters (meter scale)
_CLOTH_SCALE = 0.01  # USD vertices are in cm → convert to meters
_TRI_KE = 1e3  # area-preserving stiffness [N/m²]
_TRI_KA = 1e3  # area stiffness [N/m²]
_TRI_KD = 1.5e-4  # area damping
_BENDING_KE = 0.1  # bending stiffness [N]
_BENDING_KD = 1e-3  # bending damping
_PARTICLE_RADIUS = 0.008  # [m]
_SOFT_CONTACT_KE = 1e4  # body–particle contact stiffness [N/m]
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

        pts_cm = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
        indices = list(mesh.GetFaceVertexIndicesAttr().Get())

        # Convert vertices from cm (USD) to meters
        vertices = [wp.vec3(float(p[0] * _CLOTH_SCALE), float(p[1] * _CLOTH_SCALE), float(p[2] * _CLOTH_SCALE)) for p in pts_cm]

        # Rotate -90° around x so the shirt's y-up axis maps to z-up (gravity is -z).
        # After this rotation the shirt occupies z ≈ [0.88, 1.52] m above the ground.
        rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -np.pi / 2)

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, self.cfg.cloth_drop_height),
            rot=rot,
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
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
        """
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager._model
        if model is None or not hasattr(model, "edge_rest_angle"):
            return

        model.edge_rest_angle.zero_()
        model.soft_contact_ke = _SOFT_CONTACT_KE
        model.soft_contact_kd = _SOFT_CONTACT_KD

    # ─── RL interface (no-op — demo only) ────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        pass

    def _apply_action(self) -> None:
        pass

    def _get_observations(self) -> dict:
        # Return a placeholder observation (1-dim zero tensor per env)
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
