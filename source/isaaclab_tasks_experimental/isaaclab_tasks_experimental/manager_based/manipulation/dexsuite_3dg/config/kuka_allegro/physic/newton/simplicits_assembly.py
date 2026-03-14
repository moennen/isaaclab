# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Single-env SimplicitsModelBuilder assembly (Step 3): rigid proto + one Simplicits object.

The builder and resulting model are owned by the extended Newton manager (Step 5):
Dexsuite3dgNewtonManager will create the SimplicitsModelBuilder, run this assembly
logic (or call this helper), and set cls._model from the finalized builder. This
module only provides the build helper for use by the manager and by tests.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .dexsuite_3dg_builder_utils import build_rigid_proto_excluding_object

logger = logging.getLogger("dexsuite_3dg.simplicits.assembly")


def build_single_env_simplicits_model(
    stage: Any,
    env_path: str,
    object_relative_path: str,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    simplicits_cfg: Any,
    init_transform: torch.Tensor | None = None,
    device: str = "cuda",
    up_axis: str = "Z",
    solver_type: str | None = None,
    num_qp: int | None = None,
    collision_particle_radius: float = 0.05,
    detection_ratio: float = 1.5,
    gravity: float = 9.81,
    verbose: bool = False,
) -> Any:
    """Build a single-env SimplicitsModel: rigid proto (Step 2) + one Simplicits object (Step 1), then finalize.

    Order: add rigid proto (per-env only), add Simplicits object at init_transform,
    add_simplicits_collisions, add_ground_plane, finalize(device). No cloning; one world.

    Args:
        stage: USD stage (e.g. get_current_stage()).
        env_path: Env root path for rigid proto (e.g. /World/envs/env_0).
        object_relative_path: Prim name to exclude from proto (e.g. Object).
        vertices: Mesh vertices for the Simplicits object, shape (V, 3).
        faces: Mesh faces, shape (F, 3).
        simplicits_cfg: SimplicitsObjectCfg for the mesh→Simplicits factory.
        init_transform: 4x4 or 3x4 initial transform for the Simplicits object; identity if None.
        device: Target device for finalize.
        up_axis: Newton up axis (e.g. Z).
        solver_type: Optional solver type for rigid proto registration (e.g. mujoco_warp).
        num_qp: Quadrature points for Simplicits object; defaults to simplicits_cfg.num_samples.
        collision_particle_radius: Passed to add_simplicits_collisions.
        detection_ratio: Passed to add_simplicits_collisions.
        gravity: Gravity magnitude (positive; applied along negative up axis).
        verbose: If True, log assembly steps.

    Returns:
        SimplicitsModel (from Kaolin) after finalize, ready for state() and SimplicitsSolver.step.
    """
    from newton import Axis

    try:
        from kaolin.experimental.newton.builder import SimplicitsModelBuilder
    except ImportError as e:
        raise ImportError(
            "Kaolin experimental Newton (SimplicitsModelBuilder) is required for assembly. "
            "Install kaolin with newton support."
        ) from e

    from ..kaolin import SimplicitsObjectCfg, create_rigid_simplicits_object_from_mesh

    if not isinstance(simplicits_cfg, SimplicitsObjectCfg):
        raise TypeError("simplicits_cfg must be SimplicitsObjectCfg")

    axis = Axis.from_string(up_axis) if isinstance(up_axis, str) else up_axis

    # Step 2: rigid proto (per-env only)
    proto = build_rigid_proto_excluding_object(
        stage,
        env_path=env_path,
        object_relative_path=object_relative_path,
        up_axis=up_axis,
        solver_type=solver_type,
    )
    if verbose:
        logger.debug("assembly: built rigid proto for env_path=%s", env_path)

    # Step 1: rigid Simplicits object from mesh
    sim_obj = create_rigid_simplicits_object_from_mesh(vertices, faces, simplicits_cfg, device=device)
    n_qp = num_qp if num_qp is not None else simplicits_cfg.num_samples
    if verbose:
        logger.debug("assembly: created Simplicits object, num_qp=%s", n_qp)

    if init_transform is None:
        init_transform = torch.eye(4, device=vertices.device, dtype=vertices.dtype)
    elif init_transform.dim() == 2 and init_transform.shape[0] == 3:
        # 3x4 -> 4x4
        row = torch.zeros(1, 4, device=init_transform.device, dtype=init_transform.dtype)
        row[0, 3] = 1.0
        init_transform = torch.cat([init_transform, row], dim=0)

    # Assemble SimplicitsModelBuilder
    smb = SimplicitsModelBuilder(up_axis=axis, gravity=-float(gravity))
    smb.add_builder(proto)
    smb.add_simplicits_object(sim_obj, num_qp=n_qp, init_transform=init_transform)
    smb.add_simplicits_collisions(
        collision_particle_radius=collision_particle_radius,
        detection_ratio=detection_ratio,
    )
    smb.add_ground_plane()

    model = smb.finalize(device=device, requires_grad=False)
    if verbose:
        logger.info(
            "assembly: finalize done simplicits_particle_start=%s simplicits_particle_end=%s",
            model.simplicits_particle_start,
            model.simplicits_particle_end,
        )
    return model
