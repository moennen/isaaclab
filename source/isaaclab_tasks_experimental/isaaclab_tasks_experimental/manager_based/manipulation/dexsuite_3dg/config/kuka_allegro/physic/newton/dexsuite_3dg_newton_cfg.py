# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics config that uses the extended Dexsuite 3dg manager."""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg


@configclass
class Dexsuite3dgNewtonCfg(NewtonCfg):
    """Newton config that uses :class:`Dexsuite3dgNewtonManager` for this task."""

    class_type: type | str = "{DIR}.dexsuite_3dg_newton_manager:Dexsuite3dgNewtonManager"
    """Use the extended manager so 3dg-specific overrides run in Newton mode."""

    solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg(
        solver="newton",
        integrator="implicitfast",
        njmax=300,
        nconmax=70,
        impratio=10.0,
        cone="elliptic",
        update_data_interval=2,
        iterations=100,
        ls_iterations=15,
        ls_parallel=False,
        use_mujoco_contacts=True,
        ccd_iterations=5000,
    )
    num_substeps: int = 2
    debug_mode: bool = False
