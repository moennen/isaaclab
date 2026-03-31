# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled solver that alternates a rigid-body solver and VBD (cloth) per substep.

This implements one-way coupling (rigid -> cloth): the rigid solver advances
body dynamics first, then VBD reads the updated body poses to compute
cloth-body contacts and integrate particles.

The rigid solver can be either :class:`SolverFeatherstone` or :class:`SolverMuJoCo`.

Reference: ``newton/examples/cloth/example_cloth_franka.py`` (lines 596-633).
"""

from __future__ import annotations

import logging

import warp as wp
from newton import CollisionPipeline, Contacts, Control, Model, State
from newton.solvers import SolverBase, SolverVBD

logger = logging.getLogger(__name__)


class CoupledSolver:
    """Coupled rigid-body + VBD solver for rigid-body/cloth interaction.

    Per substep the stepping pattern is:

    1. Clear forces on both states.
    2. **Rigid step** — advance rigid bodies (Featherstone or MuJoCo).
       For Featherstone: temporarily mask particles and zero gravity.
       For MuJoCo: operates on its own internal model, no masking needed.
    3. Clear spurious particle forces.
    4. **Collision detection** — ``CollisionPipeline.collide()`` finds cloth-body contacts.
    5. **VBD step** — integrates particles only (``integrate_with_external_rigid_solver=True``).

    The solver owns the :class:`CollisionPipeline` and :class:`Contacts` internally
    so that ``NewtonManager._simulate`` does not need to call ``collide()`` externally.
    """

    def __init__(
        self,
        model: Model,
        rigid_solver: SolverBase,
        rigid_solver_type: str,
        vbd_kwargs: dict,
        collision_pipeline: CollisionPipeline,
        contacts: Contacts,
    ):
        """
        Args:
            model: The Newton model.
            rigid_solver: Pre-constructed rigid-body solver (Featherstone or MuJoCo).
            rigid_solver_type: ``"featherstone"`` or ``"mujoco_warp"``.
            vbd_kwargs: Keyword arguments for :class:`SolverVBD`.
            collision_pipeline: Collision pipeline for cloth-body contacts.
            contacts: Contacts buffer for the collision pipeline.
        """
        self._model = model
        self.rigid_solver = rigid_solver
        self._rigid_solver_type = rigid_solver_type

        # VBD solver for cloth — must use external rigid integration
        vbd_kwargs["integrate_with_external_rigid_solver"] = True
        self.vbd = SolverVBD(model, **vbd_kwargs)

        # Collision pipeline and contacts buffer (owned by this solver)
        self.collision_pipeline = collision_pipeline
        self.contacts = contacts

        # For Featherstone: pre-allocate gravity arrays for swapping.
        # Using wp.array.assign() is CUDA-graph safe (captured as a memcpy).
        self._is_featherstone = rigid_solver_type == "featherstone"
        if self._is_featherstone:
            self._gravity_zero = wp.zeros(1, dtype=wp.vec3, device=model.device)
            self._gravity_real = wp.array([model.gravity.numpy()[0]], dtype=wp.vec3, device=model.device)

        logger.info(
            "CoupledSolver initialized: %s + VBD(%s)",
            rigid_solver_type,
            {k: v for k, v in vbd_kwargs.items() if k != "integrate_with_external_rigid_solver"},
        )

    def rebuild_bvh(self, state: State) -> None:
        """Rebuild BVH for VBD collision detection."""
        self.vbd.rebuild_bvh(state)

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """One coupled substep: rigid solver then VBD.

        Args:
            state_in: Current state (read/write).
            state_out: Next state (write).
            control: Joint-level control inputs.
            contacts: Ignored — the solver uses its own internal contacts.
            dt: Substep timestep [s].
        """
        model = self._model

        # 1. Clear forces
        state_in.clear_forces()
        state_out.clear_forces()

        # 2. Rigid-body step
        if self._is_featherstone:
            # Featherstone: mask particles and zero gravity so it only advances rigid bodies
            saved_particle_count = model.particle_count
            saved_shape_contact_pair_count = model.shape_contact_pair_count
            model.particle_count = 0
            model.gravity.assign(self._gravity_zero)
            model.shape_contact_pair_count = 0

            self.rigid_solver.step(state_in, state_out, control, None, dt)

            model.particle_count = saved_particle_count
            model.gravity.assign(self._gravity_real)
            model.shape_contact_pair_count = saved_shape_contact_pair_count
        else:
            # MuJoCo: operates on its own internal model; no particle masking needed.
            # MuJoCo uses single-state stepping (state_in == state_out is fine).
            self.rigid_solver.step(state_in, state_in, control, None, dt)

        # 3. Clear spurious particle forces from rigid step
        state_in.particle_f.zero_()

        # 4. Collision detection (cloth-body contacts)
        self.collision_pipeline.collide(state_in, self.contacts)

        # 5. VBD step — particles only, reads updated rigid poses
        self.vbd.step(state_in, state_out, control, self.contacts, dt)

    def notify_model_changed(self, change: int) -> None:
        """Forward model-change notifications to both sub-solvers."""
        self.rigid_solver.notify_model_changed(change)
        self.vbd.notify_model_changed(change)
