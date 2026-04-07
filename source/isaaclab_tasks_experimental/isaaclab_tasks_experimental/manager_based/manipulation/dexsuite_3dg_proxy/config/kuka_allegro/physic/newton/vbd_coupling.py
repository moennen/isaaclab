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
``newton/_src/solvers/vbd/rigid_vbd_kernels.py`` (normal-force term only):

.. code-block:: python

    bx          = transform_point(body_q[body], contact_body_pos)   # world
    n           = contact_normal                                      # body→particle
    penetration = -(dot(n, particle_pos - bx) - particle_radius)
    F_particle  = n * ke * penetration        # VBD pushes particle out
    F_body      = -F_particle                 # Newton's 3rd law
    torque      = cross(bx - body_com_world, F_body)

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
# Warp kernel
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
    """Newton's-third-law reaction from soft particles onto rigid bodies.

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


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def apply_soft_body_reactions(contacts, state, model, max_contacts: int = VBD_MAX_CONTACTS) -> None:
    """Inject soft-contact reaction forces into ``state.body_f`` for two-way coupling.

    Call after ``collide()`` and **before** ``mujoco_solver.step()`` so that the
    rigid solver sees the soft-body reaction in the same substep that VBD
    computes the contact force on the particle (zero-lag same-substep coupling).

    The caller must call ``state.clear_forces()`` at the start of each substep
    so reactions from the previous substep do not accumulate.

    Args:
        contacts: Newton ``Contacts`` object populated by ``collide()`` or
            ``CollisionPipeline.collide()``.
        state: Current Newton ``State``; ``state.body_f`` is written in-place.
        model: Newton ``Model`` owning the simulation.
        max_contacts: Kernel launch dimension (number of contact slots).
            Must be ``>= contacts.soft_contact_max``.  Threads beyond the
            actual contact count early-exit immediately.
    """
    if contacts is None or state.body_f is None:
        return
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
