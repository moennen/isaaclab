# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simplicits physics parameters (material, sampling, contact, solver) for mesh → rigid Simplicits.

All tunable parameters for the Simplicits pipeline live here — material, particle sampling,
contact mechanics, and solver settings — so every physics knob is in one place.

Quick tuning guide
------------------
Fingers penetrate the object:
    Raise ``soft_contact_coeff`` (e.g. 0.5 → 2.0) or ``soft_contact_ke`` (e.g. 1e4 → 3e4).

Robot backs off violently on first touch:
    Lower ``soft_contact_coeff`` (e.g. 0.5 → 0.1).
    Alternatively, increase ``num_substeps`` in the env config to reduce per-substep dt.

Gravity appears to do nothing (object floats):
    Lower ``newton_conv_tol`` (e.g. 1e-9 → 1e-11). The default 1e-9 is already safe
    for dt ≈ 0.008 s; this is only needed if dt increases significantly.

Object deformation looks too stiff or too soft:
    Adjust ``youngs_modulus``. For a nearly-rigid foam-like object: 1e4–1e5 Pa.
    For rubber: 1e6–1e7 Pa. Note: with 1 handle (rigid Simplicits), deformation
    is a uniform affine warp — very stiff material reduces to near-rigid motion.
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class SimplicitsObjectCfg:
    """All physics parameters for a rigid Simplicits object.

    Covers four groups:

    * **Material** (``density``, ``youngs_modulus``, ``poisson_ratio``): govern inertia and
      elastic response of the Simplicits reduced-DOF model.
    * **Sampling** (``num_samples``, ``collision_particle_radius``): determine particle
      density and contact detection geometry.
    * **Contact mechanics** (``soft_contact_ke``, ``soft_contact_coeff``,
      ``contact_detection_ratio``): control the strength and reach of soft–rigid contact forces
      and the Newton's 3rd-law reaction transferred back to the robot.
    * **Solver** (``newton_conv_tol``): convergence criterion for the inner Newton-Raphson loop
      of the Simplicits implicit integrator.
    """

    # ------------------------------------------------------------------
    # Material
    # ------------------------------------------------------------------

    density: float = 500.0
    """Density [kg/m³] applied to all sampled particles.

    Governs object mass (mass = density × appx_vol). Typical values:
    wood ~600, foam ~100–300, rubber ~1200.
    """

    youngs_modulus: float = 1e5
    """Young's modulus [Pa] for elastic stiffness.

    Controls resistance to deformation. With one rigid handle the object deforms
    as a uniform affine warp, so this mainly affects oscillation frequency after
    contact rather than large-scale shape change.
    Soft foam: 1e3–1e4 Pa · Rubber: 1e6–1e7 Pa · Rigid-like: ≥ 1e8 Pa.
    """

    poisson_ratio: float = 0.45
    """Poisson ratio (dimensionless, 0 < ν < 0.5).

    Near-incompressible materials (foam, rubber): 0.4–0.49.
    Values above 0.499 may cause numerical issues.
    """

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    num_samples: int = 2400
    """Number of quadrature / collision particles sampled from the mesh surface.

    More samples → smoother contact forces and better coverage, but higher
    collision-detection and Simplicits-solver cost per step.
    Typical range: 500 (fast, coarse) – 5000 (accurate, slow).
    """

    collision_particle_radius: float | None = None
    """Radius [m] for each collision particle (scene-wide).

    Used as both the per-particle sphere radius and the base for contact
    detection (see ``contact_detection_ratio``). If ``None``, auto-computed
    from mesh extent and ``num_samples`` as ``extent / num_samples^(1/3) * 0.5``.
    """

    # ------------------------------------------------------------------
    # Contact mechanics
    # ------------------------------------------------------------------

    soft_contact_ke: float = 1e4
    """Contact spring stiffness [N/m] (sets ``model.soft_contact_ke``).

    Base spring constant used by :class:`SimplicitsParticleNewtonShapeSoftContact`.
    The raw per-contact force magnitude is::

        F ≈ soft_contact_ke × soft_contact_coeff × penetration_depth   [N]

    Increasing this makes contacts stiffer but can destabilise the solver
    if raised above ~1e5 without also tightening ``newton_conv_tol``.
    Tune ``soft_contact_coeff`` for day-to-day stiffness adjustment and
    reserve ``soft_contact_ke`` for order-of-magnitude changes.
    """

    soft_contact_coeff: float = 0.5
    """Gradient scale for soft–rigid contact energy (dimensionless).

    Multiplies *both* the Simplicits energy gradient (how much the soft body
    deforms away from the finger) and the Newton's 3rd-law reaction force
    transferred back to the robot joints.

    Effective contact force ≈ ``soft_contact_ke × soft_contact_coeff × penetration``.

    * Fingers still penetrate → raise (e.g. 0.5 → 2.0).
    * Robot backs off too fast on first touch → lower (e.g. 0.5 → 0.1).
    """

    contact_detection_ratio: float = 1.5
    """Detection radius multiplier relative to ``collision_particle_radius`` (dimensionless).

    A contact is activated when a rigid surface is within
    ``contact_detection_ratio × collision_particle_radius`` of a particle.
    Values near 1.0 may miss shallow contacts; values above 2.0 can cause
    pre-contact forces before the finger visually touches the object.
    """

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    newton_conv_tol: float = 1e-9
    """Newton-Raphson convergence tolerance for the Simplicits implicit integrator.

    The inner NR loop exits when ``|dx^T G| < newton_conv_tol`` where G is the
    potential gradient scaled by ``dt²``. At dt ≈ 0.008 s this scales as ~1e-7
    for a 250 g object under gravity, so the default 1e-9 ensures at least one
    iteration runs and gravity is applied correctly. Raise to 1e-7 only if
    performance is critical and slight gravity drift is acceptable.
    """
