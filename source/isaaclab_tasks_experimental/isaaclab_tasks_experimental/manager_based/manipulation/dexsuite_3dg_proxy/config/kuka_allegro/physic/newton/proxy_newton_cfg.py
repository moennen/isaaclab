# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics config that uses the extended Dexsuite 3dg Proxy manager for VBD soft bodies.

Physics coupling architecture: same-substep two-way coupling
-------------------------------------------------------------
Newton's two-solver pattern is used:

  1. ``SolverMuJoCo``  — integrates the Allegro rigid body (joints, fingers).
  2. ``SolverVBD``     — integrates only the soft-body particles
     (``integrate_with_external_rigid_solver=True``).

The coupling is **two-way within each substep**:

  a. ``collide(s0, contacts)`` — detect particle-rigid contacts at current positions.
  b. ``apply_soft_body_reactions(contacts → s0.body_f)`` — accumulate
     equal-and-opposite contact reaction forces onto the rigid bodies using
     ``state.body_f`` (a first-class Newton API read by ``SolverMuJoCo.step()``
     via ``apply_mjc_body_f_kernel`` → ``xfrc_applied``).
  c. ``MuJoCo.step(s0 → s1)`` — rigid step; reads ``body_f`` so finger joints
     feel resistance from the soft object in the **same substep**.
  d. ``VBD.step(s0 → s1, contacts)`` — soft step; uses the same contact buffer
     so action and reaction are computed from identical contact geometry.

This is operator-splitting (IMEX) with **zero time lag**: contacts are detected
at ``s0`` positions and both the rigid reaction and the soft correction are
applied simultaneously rather than iteratively.  The approximation is that
rigid and soft corrections are sequentially applied within one substep rather
than solved monolithically — the same approach used in every off-the-shelf
physics engine that supports deformable-rigid coupling.

Critical parameter constraint (Newton VBD internal formulation)
---------------------------------------------------------------
``soft_contact_kd`` and ``k_damp`` are **position-level stiffness multipliers**,
not velocity-level damping coefficients.  The effective stiffness added per
substep is::

    stiffness = kd × ke / dt

With ``ke=1e4`` and ``dt = 1 / (decimation_rate × substeps)``::

    kd = 1e-5  →  stiffness ≈      50 N/m   ← stable
    kd = 1e-2  →  stiffness ≈  50 000 N/m   ← unstable
    kd = 100   →  stiffness ≈   5×10⁸ N/m   ← immediate VBD divergence at
                                                first contact

Keep ``soft_contact_kd`` and ``k_damp`` at ``1e-5``.  The rule of thumb
``2*sqrt(ke * mass)`` applies only to velocity-based damping formulations, not
to Newton's position-level scheme.
"""

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

    k_damp: float = 1e-5
    """VBD material damping coefficient.

    .. warning::
        MUST be kept near zero.  In Newton's VBD solver this is a *position-level*
        stiffness multiplier, not a velocity-level damping coefficient.  The
        effective stiffness it adds is ``k_damp × k_mu / dt``.  A value of
        ``1e-2`` with ``k_mu=1e4`` and ``dt=0.002 s`` gives 50 000 N/m — enough
        to cause VBD divergence under moderate contact loads.  Set to ``1e-5``
        and leave it there; oscillation damping comes from ``soft_contact_kd``
        and solver iterations, not this coefficient.
    """

    particle_radius: float = 0.015
    """Particle collision radius [m]. Controls when particles begin contact with rigid shapes.

    Too small: particles tunnel through fingers.
    Too large: object feels inflated / fingers cannot fully grasp.
    Default 0.015 m (~3/4 of the ~20 mm inter-particle spacing in the 100k_tet training mesh).
    """

    soft_contact_ke: float = 1e4
    """Particle-rigid contact stiffness [N/m]. Higher = harder contact surface."""

    soft_contact_kd: float = 1e-5
    """Particle-rigid contact damping coefficient.

    .. warning::
        MUST be kept near zero.  In Newton's VBD solver this is a *position-level*
        stiffness multiplier, not a physical ``N·s/m`` damping coefficient.  The
        effective stiffness it adds is ``soft_contact_kd × soft_contact_ke / dt``.
        A value of ``100`` with ``ke=1e4`` and ``dt=0.002 s`` gives 5×10⁸ N/m —
        immediate VBD divergence at first particle-rigid contact.

        The rule of thumb ``~2*sqrt(ke * particle_mass)`` applies only to
        velocity-based damping formulations and does **not** apply here.  Set
        to ``1e-5`` and leave it there.
    """

    soft_contact_mu: float = 2.0
    """Particle-rigid friction coefficient.

    Set to 2.0 based on grasp-lift validation: the two-way coupling now feeds
    the tangential (vertical) friction reaction back to MuJoCo ``body_f``, so
    the joint controllers feel the load during LIFT.  At mu=0.8 the friction
    force was insufficient to support the object weight; mu=2.0 is the
    empirically validated minimum for a reliable lift.
    """

    vbd_iterations: int = 20
    """Number of VBD Gauss-Seidel iterations per substep.

    10 iterations is insufficient for convergence under multi-finger contact loads
    and causes particle velocity explosion.  20 is the empirically validated minimum;
    increase to 25-30 if VBD NaN is observed during policy training.
    """

    vbd_two_way_coupling: bool = True
    """Enable same-substep two-way coupling between VBD particles and rigid bodies.

    When True, :func:`apply_soft_body_reactions` accumulates equal-and-opposite
    contact forces into ``state.body_f`` before each rigid step, so finger joints
    feel resistance from the soft object in the same substep.

    Set to False for one-way coupling (object reacts to fingers, fingers do not
    feel the object).  Useful for debugging or ablation studies.
    """

    vbd_max_contacts_per_env: int = 400
    """Upper bound on soft contacts per environment at any given simulation step.

    Controls the VBD contact kernel launch dim and contact buffer size.
    With 255 particles and ~49 shapes, typical contact counts are 50–200 per env;
    400 gives comfortable headroom.  Increase if you see
    '[Proxy VBD] soft_contact_count overflow' warnings.
    """

    vbd_shapes_per_world: int | None = None
    """Override the number of shapes per world used for VBD contact kernels.
    None = auto-detect from warmup (recommended first run to find the right value).
    After the first run, check the log for '[Proxy VBD] contact shape range' and set
    this to the reported tight value to avoid redundant kernel threads from Kuka arm
    shapes that never contact the doll."""

    vbd_max_particle_velocity: float = 10.0
    """Maximum particle speed [m/s] allowed after each VBD iteration.

    Displacements exceeding ``vbd_max_particle_velocity × dt_substep`` are scaled
    back proportionally (direction preserved).

    Prevents velocity runaway when a moving rigid body suddenly contacts many
    particles simultaneously — the typical cause of VBD NaN explosions in
    rigid-VBD two-way coupling.  Set to ``inf`` to disable clamping.
    """

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
