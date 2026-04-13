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

    cube_resolution: int = 3
    """Number of VBD cells per cube edge.

    A resolution of ``N`` produces ``(N+1)³`` particles and ``5 N³`` tetrahedra.
    Default: 3 → 64 particles, 135 tets per env.
    Increase for more detailed deformation (costs more compute).
    """

    # -- Material parameters (user-facing) --

    young_modulus: float = 2e4
    """Young's modulus [Pa] — controls overall stiffness.

    Typical values:
        1e3 – 5e3  : very soft (gel-like, requires small timesteps)
        1e4 – 5e4  : moderately soft (rubber / foam — default range)
        1e5+       : stiff (may require more VBD iterations)
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

    particle_radius: float = 0.015
    """Particle collision radius [m].

    Controls when particles begin contact with rigid shapes.  Typical value:
    ``cube_size / cube_resolution * 0.9``.  Default: 0.015 m for a 5 cm / 3
    resolution cube (inter-particle spacing ≈ 0.0167 m).
    """

    # -- Contact parameters --

    soft_contact_ke: float = 1e4
    """Particle-rigid contact stiffness [N/m].  Higher = harder contact surface."""

    soft_contact_kd: float = 1e-5
    """Particle-rigid contact damping coefficient.

    .. warning::
        MUST be kept near zero.  In Newton's VBD solver this is a *position-level*
        stiffness multiplier, not a physical ``N·s/m`` damping coefficient.
        Set to ``1e-5`` and leave it there.
    """

    soft_contact_mu: float = 1.5
    """Particle-rigid friction coefficient.

    Validated against the two-way coupling full reaction: ``mu >= 1.5`` is
    needed for reliable lift against gravity.  Lower values cause the cube to
    slide out of the gripper during the lift phase.
    """

    # -- VBD solver parameters --

    vbd_iterations: int = 20
    """Number of VBD Gauss-Seidel iterations per substep.

    10 iterations is insufficient for convergence under multi-finger contact
    loads.  20 is the empirically validated minimum; increase to 25–30 if VBD
    NaN is observed during policy training.
    """

    vbd_two_way_coupling: bool = True
    """Enable same-substep two-way coupling between VBD particles and rigid bodies.

    When True, :func:`apply_soft_body_reactions` accumulates equal-and-opposite
    contact forces into ``state.body_f`` before each rigid step, so finger joints
    feel resistance from the soft object in the same substep.

    Set to False for one-way coupling (useful for debugging).
    """

    vbd_max_contacts_per_env: int = 200
    """Upper bound on soft contacts per environment at any given simulation step.

    With 64 particles and ~49 shapes, typical contact counts are 10–80 per env;
    200 gives comfortable headroom.  Increase if tunneling appears.
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
