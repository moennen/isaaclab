# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics config that uses the extended Dexsuite 3dg Proxy manager for VBD soft bodies."""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.utils import configclass


@configclass
class Dexsuite3dgProxyNewtonCfg(NewtonCfg):
    """Newton config that uses :class:`Dexsuite3dgProxyNewtonManager` for VBD soft body simulation."""

    class_type: type | str = "{DIR}.proxy_newton_manager:Dexsuite3dgProxyNewtonManager"
    """Use the extended manager so VBD soft body overrides run in Newton mode."""

    vbd_enabled: bool = False
    """When True, the object is simulated as a VBD soft body using the tet mesh at tet_mesh_path."""

    tet_mesh_path: str = ""
    """Absolute path to the tetrahedral mesh (.msh, Gmsh format) produced by mesh_to_tet.py.

    The mesh is loaded via meshio:
        mesh.points              → nodes  (V, 3)
        mesh.cells_dict["tetra"] → tets   (T, 4)
    """

    density: float = 1e3
    """Mass density [kg/m³]. Default: 1000 (water-like, suitable for a soft doll)."""

    k_mu: float = 1e4
    """Lamé first parameter μ [Pa] — controls shear stiffness.

    Typical values:
        1e3 – 1e4 : very soft (gel-like)
        1e4 – 1e5 : moderately soft (rubber / soft tissue)
        1e5+      : stiff (may require more substeps)
    """

    k_lambda: float = 1e4
    """Lamé second parameter λ [Pa] — controls bulk (volume) stiffness.

    For near-incompressible materials set k_lambda >> k_mu (e.g. k_lambda = 10 * k_mu).
    """

    k_damp: float = 1e-2
    """Damping coefficient. Higher values damp oscillations faster but reduce realism."""

    particle_radius: float = 0.005
    """Particle collision radius [m]. Controls when particles begin contact with rigid shapes.

    Too small: particles tunnel through fingers.
    Too large: object feels inflated / fingers cannot fully grasp.
    Start at 0.005 m (5 mm) for a hand-sized object.
    """

    soft_contact_ke: float = 1e4
    """Particle-rigid contact stiffness [N/m]. Higher = harder contact surface."""

    soft_contact_kd: float = 100.0
    """Particle-rigid contact damping [N·s/m]. Rule of thumb: ~2*sqrt(ke * particle_mass)."""

    soft_contact_mu: float = 0.8
    """Particle-rigid friction coefficient."""

    vbd_iterations: int = 10
    """Number of VBD solver iterations per substep. Increase (20-30) if instabilities appear."""

    solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg(
        solver="newton",
        integrator="implicitfast",
        njmax=150,
        nconmax=40,
        impratio=50.0,
        cone="elliptic",
        update_data_interval=2,
        iterations=100,
        ls_iterations=15,
        ls_parallel=False,
        use_mujoco_contacts=True,
        ccd_iterations=200,
    )
