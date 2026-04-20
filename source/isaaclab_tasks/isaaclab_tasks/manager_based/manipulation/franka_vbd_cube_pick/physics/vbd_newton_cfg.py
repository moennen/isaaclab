# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics config for the Franka VBD cube pick task.

Physics coupling architecture: same-substep two-way coupling
-------------------------------------------------------------
Newton's two-solver pattern is used:

  1. ``SolverMuJoCo``  — integrates the Franka Panda rigid body (joints, fingers).
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

Keep ``soft_contact_kd`` and ``k_damp`` at ``1e-5``.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg


@configclass
class FrankaVbdCubePickNewtonCfg(NewtonCfg):
    """Newton config that uses :class:`FrankaVbdCubePickNewtonManager` for VBD soft-body cube.

    The cube is created procedurally via ``add_soft_grid()`` using the parameters
    below.  Material stiffness is specified via ``young_modulus`` and
    ``poisson_ratio``; the Lamé parameters (``k_mu``, ``k_lambda``) are computed
    internally by the manager.
    """

    class_type: type | str = "{DIR}.vbd_newton_manager:FrankaVbdCubePickNewtonManager"
    """Use the extended manager so VBD soft body runs in Newton mode."""

    # -- Cube geometry --

    cube_size: float = 0.05
    """Side length of the deformable cube [m].  Default: 5 cm."""

    cube_resolution: int = 5
    """Number of VBD cells per cube edge.

    A resolution of ``N`` produces ``(N+1)³`` particles and ``5 N³`` tetrahedra.
    Default: 5 → 216 particles, 625 tets per env.
    Matches ``generate_sequences.py`` (``_CUBE_RESOLUTION = 5``).
    """

    # -- Material parameters (user-facing) --

    young_modulus: float = 2e5
    """Young's modulus [Pa] — controls overall stiffness.

    Typical values:
        1e3 – 5e3  : very soft (gel-like, requires small timesteps)
        1e4 – 5e4  : too soft — ke/k_mu > 1 causes VBD contact instability (launch explosions)
        1e5 – 5e5  : rubber-like sweet spot — ke/k_mu ≈ 0.1 → stable VBD contact + good convergence
        1e6+       : stiff — VBD convergence degrades (needs more iterations)
    Default 2e5 gives ~20% compression under grip forces (clearly deformable, VBD-stable).
    """

    poisson_ratio: float = 0.4
    """Poisson's ratio (dimensionless) — controls volume preservation.

    Must be in (0, 0.5).  Values close to 0.5 approach incompressibility.
    Default 0.4 gives moderate volume stiffness (rubber-like).
    """

    density: float = 400.0
    """Mass density [kg/m³].

    Default: 400 kg/m³ — matches the rigid cube in ``franka_cube_pick``
    (0.1 kg over (0.05 m)³ = 400 kg/m³).
    """

    k_damp: float = 1e-5
    """VBD material damping coefficient.

    .. warning::
        MUST be kept near zero.  In Newton's VBD solver this is a *position-level*
        stiffness multiplier, not a velocity-level damping coefficient.  The
        effective stiffness it adds is ``k_damp × k_mu / dt``.  A value of
        ``1e-2`` with ``k_mu=1e4`` and ``dt=0.002 s`` gives 50 000 N/m — enough
        to cause VBD divergence under moderate contact loads.  Set to ``1e-5``
        and leave it there.
    """

    particle_radius: float = 0.009
    """Particle collision radius [m].

    Controls when particles begin contact with rigid shapes.  At resolution 5
    the inter-particle spacing is 10 mm; a radius of 9 mm means only the 1st
    and 2nd particle layers contact the finger, keeping normal forces manageable.
    Matches ``generate_sequences.py`` (``_PARTICLE_RADIUS = 0.009``).
    """

    # -- Contact parameters --

    soft_contact_ke: float = 1e3
    """Particle-side contact stiffness [N/m] — the reaction kernel ke.

    VBD internally uses ``avg_ke = 0.5 * (soft_contact_ke + shape_contact_ke)``
    for the contact force on particles.  The intentional asymmetry
    (reaction_ke = 1e3 << VBD_ke ≈ 25500 N/m) is required for reliable lift;
    see ``vbd_coupling.py`` for the full derivation.
    Matches ``generate_sequences.py`` (``_SOFT_CONTACT_KE = 1e3``).
    """

    soft_contact_kd: float = 1e-3
    """Particle-rigid contact position-level stiffness multiplier.

    .. warning::
        In Newton's VBD solver this is a *position-level* stiffness multiplier,
        not a ``N·s/m`` damping coefficient.  The effective stiffness it adds is
        ``kd × ke / dt``.  With ``ke = 1e3`` and ``dt = 2 ms`` this gives
        ``1e-3 × 1e3 / 2e-3 = 500 N/m`` — small and safe.  Do **not** increase
        ``kd`` above ``~1e-2`` or instability results.
    Matches ``generate_sequences.py`` (``_SOFT_CONTACT_KD = 1e-3``).
    """

    soft_contact_mu: float = 3.0
    """Particle-side friction coefficient.

    The effective friction is the geometric mean of the particle-side and
    shape-side values: ``mu_eff = sqrt(soft_contact_mu × shape_contact_mu)``.
    With ``soft_contact_mu = 3.0`` and ``shape_contact_mu = 0.75`` (see below)
    ``mu_eff = sqrt(3.0 × 0.75) = 1.50``, which is validated for reliable lift.
    Matches ``generate_sequences.py`` (``_SOFT_CONTACT_MU = 3.0``).
    """

    # -- Finger / gripper shape contact material --

    shape_contact_ke: float = 5.0e4
    """Contact stiffness [N/m] applied to finger (and other non-arm body) shapes.

    With ``soft_contact_ke = 1e3`` the VBD average stiffness is
    ``avg_ke = 0.5 × (1e3 + 5e4) = 25 500 N/m``.  The intentional ke asymmetry
    (reaction_ke = 1 000 << VBD_ke = 25 500) drives the friction scale to
    ``mu × VBD_ke × δ / ε_u ≈ 47 800 N/m``, giving tracking efficiency η ≈ 0.96.
    Matches ``generate_sequences.py`` (``_CONTACT_KE = 5.0e4``).
    """

    shape_contact_kd: float = 5.0e2
    """Contact damping coefficient for finger shapes.

    Matches ``generate_sequences.py`` (``_CONTACT_KD = 5.0e2``).
    """

    shape_contact_kf: float = 1.0e3
    """Contact friction stiffness for finger shapes.

    Matches ``generate_sequences.py`` (``_CONTACT_KF = 1.0e3``).
    """

    shape_contact_mu: float = 0.75
    """Shape-side friction coefficient for finger shapes.

    Combined with ``soft_contact_mu = 3.0`` gives effective
    ``mu_eff = sqrt(3.0 × 0.75) = 1.50``.
    Matches ``generate_sequences.py`` (``_CONTACT_MU = 0.75``).
    """

    # -- VBD solver parameters --

    vbd_iterations: int = 40
    """Number of VBD Gauss-Seidel iterations per substep.

    10 iterations is insufficient for convergence under multi-finger contact
    loads.  40 is validated for reliable grasp and lift.
    Matches ``generate_sequences.py`` (``_VBD_ITERATIONS = 40``).
    """

    vbd_two_way_coupling: bool = True
    """Enable same-substep two-way coupling between VBD particles and rigid bodies.

    When True, :func:`apply_soft_body_reactions` accumulates equal-and-opposite
    contact forces into ``state.body_f`` before each rigid step, so finger joints
    feel resistance from the soft object in the same substep.

    Set to False for one-way coupling (useful for debugging).
    """

    vbd_max_contacts_per_env: int = 300
    """Upper bound on soft contacts per environment at any given simulation step.

    With 216 particles (resolution 5) and finger + ground shapes, typical contact
    counts are 30–120 per env; 300 gives comfortable headroom.
    Matches ``generate_sequences.py`` (``_MAX_CONTACTS_PER_ENV = 300``).
    """

    vbd_shapes_per_world: int | None = None
    """Override the number of shapes per world used for VBD contact kernels.

    ``None`` = auto-detect from warmup (recommended first run to find the
    right value).  After the first run, check the log for
    ``[VBD] contact shape range`` and set this to the reported tight value to
    avoid redundant kernel threads from Franka arm shapes that never touch the cube.
    """

    vbd_max_particle_velocity: float = 10.0
    """Maximum particle speed [m/s] allowed after each VBD iteration.

    Displacements exceeding ``vbd_max_particle_velocity × dt_substep`` are scaled
    back proportionally (direction preserved).  Prevents NaN explosion when a
    rigid body suddenly contacts many particles simultaneously.  Set to ``inf``
    to disable clamping.
    """

    vbd_rigid_contact_k_start: float | None = None
    """Initial contact stiffness [N/m] for VBD body-particle contact warmstart.

    Sets ``rigid_contact_k_start`` on :class:`~newton.solvers.SolverVBD`.  The
    default VBD value is 100 N/m, which gives far too little friction on the
    first iteration (≈ 0.05 N vs 4.9 N gravity).  Setting this to
    ``avg_ke = 0.5 × (soft_contact_ke + shape_contact_ke)`` makes VBD use full
    material stiffness from iteration 1 and is required for reliable grasp.

    ``None`` (default) automatically computes
    ``0.5 × (soft_contact_ke + shape_contact_ke)`` at solver-init time.
    Matches ``generate_sequences.py`` (``rigid_contact_k_start = _avg_contact_ke``).
    """

    solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg(
        solver="newton",
        integrator="implicitfast",
        iterations=100,
        ls_parallel=True,
        ls_iterations=15,
        cone="elliptic",
        impratio=50.0,
        njmax=150,
        nconmax=40,
        use_mujoco_contacts=True,
    )
