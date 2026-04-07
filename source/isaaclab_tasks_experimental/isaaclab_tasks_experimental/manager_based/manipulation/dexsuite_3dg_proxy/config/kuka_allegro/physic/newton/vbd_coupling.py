# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Same-substep two-way coupling between Newton's VBD soft-body solver and MuJoCo rigid solver.

The key insight is that Newton's ``SolverMuJoCo`` reads external forces from
``state.body_f`` (a ``wp.array(dtype=wp.spatial_vector)``) before solving the
joint constraints.  By writing the equal-and-opposite soft-contact reaction
forces into ``body_f`` *before* calling ``MuJoCo.step()``, the rigid bodies
(finger joints) feel resistance from the deformable object in the **same
substep** that VBD applies the contact force to the particles.

Substep order for two-way coupling
------------------------------------
::

    state.clear_forces()                     # zero body_f
    collide(state_0, contacts)               # detect particle-rigid contacts
    apply_soft_body_reactions(contacts, ...) # fill body_f with reactions
    mujoco_solver.step(state_0 → state_1)    # reads body_f
    vbd_solver.step(state_0 → state_1)       # uses same contacts

This is operator-splitting (IMEX) with **zero time lag**: contact detection,
reaction injection, and both solver steps all use the same contact geometry.

Contact force formula
---------------------
Mirrors ``evaluate_body_particle_contact`` in
``newton/_src/solvers/vbd/rigid_vbd_kernels.py``.

Normal-only mode (legacy, ``dt`` not provided):

.. code-block:: python

    bx          = transform_point(body_q[body], contact_body_pos)   # world
    n           = contact_normal                                      # body→particle
    penetration = -(dot(n, particle_pos - bx) - particle_radius)
    F_particle  = n * ke * penetration        # VBD pushes particle out
    F_body      = -F_particle                 # Newton's 3rd law
    torque      = cross(bx - body_com_world, F_body)

Full coupling mode (``dt`` and ``particle_q_prev`` provided):

.. code-block:: python

    # Normal (same as above)
    F_normal = n * ke * penetration

    # Coulomb friction: geometric-mean mu, IPC regularisation
    mu      = sqrt(soft_contact_mu * shape_material_mu[shape])
    bv      = body_lin_v + cross(body_ang_v, r) + body_surface_vel
    slip    = (particle_pos - particle_pos_prev) - bv * dt
    u_t     = slip - n * dot(n, slip)          # tangential component
    f_fric  = -mu * ke * penetration * u_t / max(|u_t|, eps)

    F_particle = F_normal + f_fric
    F_body     = -F_particle                   # Newton's 3rd law
    torque     = cross(bx - body_com_world, F_body)

The full-coupling mode feeds the tangential friction reaction back to MuJoCo,
so the finger actuators feel the tangential load needed to carry the object
against gravity — the mechanism required for a successful lift.

Imports
-------
Both ``proxy_newton_manager.py`` and ``tools/validate_grasp.py`` import from
this module so the physics is defined exactly once.
"""

from __future__ import annotations

import warp as wp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Kernel launch dimension.  Threads beyond the actual contact count
# early-exit immediately so over-allocating is cheap.  Increase if
# ``_kernel_body_particle_reaction`` warns about contact overflow.
VBD_MAX_CONTACTS: int = 2048

# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _kernel_body_particle_reaction(
    contact_count:    wp.array(dtype=wp.int32),
    contact_particle: wp.array(dtype=wp.int32),
    contact_shape:    wp.array(dtype=wp.int32),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_normal:   wp.array(dtype=wp.vec3),
    particle_q:       wp.array(dtype=wp.vec3),
    particle_radius:  wp.array(dtype=wp.float32),
    body_q:           wp.array(dtype=wp.transform),
    body_com:         wp.array(dtype=wp.vec3),
    shape_body:       wp.array(dtype=wp.int32),
    soft_contact_ke:  float,
    body_f:           wp.array(dtype=wp.spatial_vector),
):
    """Newton's-third-law reaction from soft particles onto rigid bodies (normal only).

    Mirrors the normal-force term in ``evaluate_body_particle_contact()``
    (``newton/_src/solvers/vbd/rigid_vbd_kernels.py``).  One thread per
    contact slot; threads beyond the actual contact count early-exit.

    ``body_f`` is accumulated with atomic adds so multiple contacts on the
    same body are summed correctly.  The caller must zero ``state.body_f``
    via ``state.clear_forces()`` at the start of each substep.
    """
    tid = wp.tid()
    if tid >= contact_count[0]:
        return  # beyond actual contact count — early exit

    p_idx    = contact_particle[tid]
    s_idx    = contact_shape[tid]
    body_idx = shape_body[s_idx]
    if body_idx < 0:
        return  # static shape — no corresponding rigid body

    X_wb = body_q[body_idx]
    # Contact point on the rigid surface in world frame.
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    # Normal: from rigid surface toward particle (world frame).
    n  = contact_normal[tid]

    # Penetration depth — identical to evaluate_body_particle_contact.
    penetration = -(wp.dot(n, particle_q[p_idx] - bx) - particle_radius[p_idx])
    if penetration <= 0.0:
        return  # contact in buffer but not actually penetrating

    # Force on particle (VBD spring: pushes particle away from surface).
    f_on_particle = n * (soft_contact_ke * penetration)

    # Equal-and-opposite reaction on the rigid body (Newton's third law).
    reaction = -f_on_particle
    com_w    = wp.transform_point(X_wb, body_com[body_idx])
    torque   = wp.cross(bx - com_w, reaction)

    # body_f layout: [linear_force(3), torque(3)] — see Newton State docs.
    wp.atomic_add(body_f, body_idx,
                  wp.spatial_vector(reaction[0], reaction[1], reaction[2],
                                    torque[0],   torque[1],   torque[2]))


@wp.kernel
def _kernel_body_particle_reaction_with_friction(
    contact_count:     wp.array(dtype=wp.int32),
    contact_particle:  wp.array(dtype=wp.int32),
    contact_shape:     wp.array(dtype=wp.int32),
    contact_body_pos:  wp.array(dtype=wp.vec3),
    contact_body_vel:  wp.array(dtype=wp.vec3),
    contact_normal:    wp.array(dtype=wp.vec3),
    particle_q:        wp.array(dtype=wp.vec3),
    particle_q_prev:   wp.array(dtype=wp.vec3),
    particle_radius:   wp.array(dtype=wp.float32),
    body_q:            wp.array(dtype=wp.transform),
    body_qd:           wp.array(dtype=wp.spatial_vector),
    body_com:          wp.array(dtype=wp.vec3),
    shape_body:        wp.array(dtype=wp.int32),
    shape_material_mu: wp.array(dtype=wp.float32),
    soft_contact_ke:   float,
    soft_contact_mu:   float,
    friction_epsilon:  float,
    dt:                float,
    body_f:            wp.array(dtype=wp.spatial_vector),
):
    """Newton's-third-law reaction including Coulomb friction.

    Full coupling: mirrors the complete contact model (normal + tangential
    friction) from ``evaluate_body_particle_contact()`` in
    ``newton/_src/solvers/vbd/rigid_vbd_kernels.py``.

    The friction component is the reaction that was missing from the
    normal-only kernel.  Contact normals are roughly horizontal (fingers
    pressing inward), so friction is vertical — this is the force that
    lets the actuators carry the object against gravity during the LIFT phase.

    Friction model:
    - Combined mu: geometric mean of ``soft_contact_mu`` and ``shape_material_mu[shape]``
      (identical to VBD's ``evaluate_body_particle_contact``, line 751).
    - Relative slip: particle displacement minus body-surface displacement over ``dt``.
    - IPC-regularised isotropic Coulomb cone: smooth at zero slip, 1/|u_t| for large slip.
    """
    tid = wp.tid()
    if tid >= contact_count[0]:
        return

    p_idx    = contact_particle[tid]
    s_idx    = contact_shape[tid]
    body_idx = shape_body[s_idx]
    if body_idx < 0:
        return

    X_wb = body_q[body_idx]
    bx   = wp.transform_point(X_wb, contact_body_pos[tid])
    n    = contact_normal[tid]

    penetration = -(wp.dot(n, particle_q[p_idx] - bx) - particle_radius[p_idx])
    if penetration <= 0.0:
        return

    normal_load   = soft_contact_ke * penetration
    f_on_particle = n * normal_load

    # Body CoM in world frame — needed for both friction (lever arm) and torque.
    com_w = wp.transform_point(X_wb, body_com[body_idx])

    # Combined friction coefficient: geometric mean of model-level and shape-level mu.
    # Identical to evaluate_body_particle_contact() line 751 in rigid_vbd_kernels.py.
    mu = wp.sqrt(soft_contact_mu * shape_material_mu[s_idx])

    if mu > 0.0:
        # Body surface velocity at the contact point (world frame).
        body_v_s   = body_qd[body_idx]
        body_lin_v = wp.spatial_top(body_v_s)
        body_ang_v = wp.spatial_bottom(body_v_s)
        r          = bx - com_w
        bv         = (body_lin_v + wp.cross(body_ang_v, r)
                      + wp.transform_vector(X_wb, contact_body_vel[tid]))

        # Relative slip displacement: particle motion minus body-surface motion.
        dx                   = particle_q[p_idx] - particle_q_prev[p_idx]
        relative_translation = dx - bv * dt

        # IPC-regularised isotropic Coulomb friction — same formula as rigid_vbd_kernels.py.
        dot_nu = wp.dot(n, relative_translation)
        u_t    = relative_translation - n * dot_nu  # tangential component only
        u_norm = wp.length(u_t)
        eps_u  = friction_epsilon * dt

        if u_norm > 0.0:
            if u_norm > eps_u:
                f1_over_x = 1.0 / u_norm
            else:
                f1_over_x = (-u_norm / eps_u + 2.0) / eps_u
            f_on_particle = f_on_particle - (mu * normal_load * f1_over_x) * u_t

    # Equal-and-opposite reaction on the rigid body (Newton's third law).
    reaction = -f_on_particle
    torque   = wp.cross(bx - com_w, reaction)

    wp.atomic_add(body_f, body_idx,
                  wp.spatial_vector(reaction[0], reaction[1], reaction[2],
                                    torque[0],   torque[1],   torque[2]))


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def apply_soft_body_reactions(
    contacts,
    state,
    model,
    max_contacts: int = VBD_MAX_CONTACTS,
    *,
    particle_q_prev=None,
    friction_epsilon: float = 1e-2,
    dt: float | None = None,
) -> None:
    """Inject soft-contact reaction forces into ``state.body_f`` for two-way coupling.

    Call after ``collide()`` and **before** ``mujoco_solver.step()`` so that the
    rigid solver sees the soft-body reaction in the same substep that VBD
    computes the contact force on the particle (zero-lag same-substep coupling).

    The caller must call ``state.clear_forces()`` at the start of each substep
    so reactions from the previous substep do not accumulate.

    Two coupling modes are available:

    **Normal-only** (legacy, default when ``dt`` is not provided):
        Reaction = ``-n * ke * penetration``.  Fingers feel the normal push-back
        but not the tangential friction load.

    **Full coupling** (when ``dt`` and ``particle_q_prev`` are provided):
        Reaction = ``-(normal + friction)``.  Mirrors the complete contact model
        from ``evaluate_body_particle_contact()`` in VBD, using the same
        geometric-mean mu and IPC-regularised Coulomb friction.  This feeds
        the tangential friction reaction back to the finger actuators, which is
        the force needed to carry the object against gravity during a lift.

    Args:
        contacts: Newton ``Contacts`` object populated by ``collide()`` or
            ``CollisionPipeline.collide()``.
        state: Current Newton ``State``; ``state.body_f`` is written in-place.
            ``state.body_qd`` is read (body velocities) when full coupling is used.
        model: Newton ``Model`` owning the simulation.
        max_contacts: Kernel launch dimension (number of contact slots).
            Must be ``>= contacts.soft_contact_max``.  Threads beyond the
            actual contact count early-exit immediately.
        particle_q_prev: Particle positions at the start of the substep
            (world frame).  Typically ``state_prev.particle_q`` from the
            previous substep buffer.  Required for full coupling.
        friction_epsilon: IPC friction regularisation length [m].  Smooths the
            friction cone near zero slip.  Default ``1e-2`` matches
            ``SolverVBD`` default.
        dt: Substep timestep [s].  Required for full coupling.  When ``None``
            (default) the normal-only kernel is used regardless of
            ``particle_q_prev``.
    """
    if contacts is None or state.body_f is None:
        return

    if dt is not None and particle_q_prev is not None:
        # Full coupling: normal + Coulomb friction reaction.
        wp.launch(
            _kernel_body_particle_reaction_with_friction,
            dim=max_contacts,
            inputs=[
                contacts.soft_contact_count,
                contacts.soft_contact_particle,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                state.particle_q,
                particle_q_prev,
                model.particle_radius,
                state.body_q,
                state.body_qd,
                model.body_com,
                model.shape_body,
                model.shape_material_mu,
                float(model.soft_contact_ke),
                float(model.soft_contact_mu),
                float(friction_epsilon),
                float(dt),
                state.body_f,
            ],
        )
    else:
        # Normal-only (legacy) path — preserves original behaviour for callers
        # that do not provide dt / particle_q_prev.
        wp.launch(
            _kernel_body_particle_reaction,
            dim=max_contacts,
            inputs=[
                contacts.soft_contact_count,
                contacts.soft_contact_particle,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_normal,
                state.particle_q,
                model.particle_radius,
                state.body_q,
                model.body_com,
                model.shape_body,
                float(model.soft_contact_ke),
                state.body_f,
            ],
        )
