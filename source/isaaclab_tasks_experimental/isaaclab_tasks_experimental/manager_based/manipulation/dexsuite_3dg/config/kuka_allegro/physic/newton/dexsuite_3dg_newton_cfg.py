# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics config that uses the extended Dexsuite 3dg manager."""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.utils import configclass

from ..kaolin import SimplicitsObjectCfg


@configclass
class Dexsuite3dgNewtonCfg(NewtonCfg):
    """Newton config that uses :class:`Dexsuite3dgNewtonManager` for this task."""

    class_type: type | str = "{DIR}.dexsuite_3dg_newton_manager:Dexsuite3dgNewtonManager"
    """Use the extended manager so 3dg-specific overrides run in Newton mode."""

    simplicits_enabled: bool = False
    """When True, spawn object is simulated as rigid Simplicits (Step 5); requires simplicits_cfg."""

    simplicits_cfg: SimplicitsObjectCfg | None = None
    """Simplicits material/sampling/collision config. Used only when simplicits_enabled is True."""

    object_contact_ke: float | None = None
    """Override contact elastic stiffness [N/m] for the Object body's shapes (rigid mode only).

    When set, replaces Newton's default ``shape_material_ke`` for every shape that belongs to
    the prim named ``Object`` in the scene.  Use this to match the Simplicits soft-contact
    effective stiffness when retraining a policy for Simplicits deployment:

    .. code-block:: text

        ke_simplicits = soft_contact_ke × soft_contact_coeff   (e.g. 1e4 × 0.05 = 500 N/m)

    At ke=500 a 16 N finger force causes ~32 mm of penetration — comparable to Simplicits
    soft contacts.  Leave ``None`` to keep Newton's built-in defaults.
    """

    object_contact_kd: float | None = None
    """Override contact damping [N·s/m] for the Object body's shapes (rigid mode only).

    Paired with :attr:`object_contact_ke`.  For ke=500 critical damping is
    ``kd = 2 * sqrt(500) ≈ 45 N·s/m`` (``timeconst = 2/45 ≈ 0.044 s``, stable for
    ``dt = 0.0083 s``).  Leave ``None`` to keep Newton's built-in defaults.
    """

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
