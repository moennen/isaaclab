# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extended NewtonManager for Dexsuite 3dg task (Newton mode only).

Use this module together with the env wrapper that patches ``isaaclab_newton.physics.NewtonManager``
before the simulation context is created. See docs/NEWTON_MANAGER_EXTENSION.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_newton.physics import NewtonManager

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext


class Dexsuite3dgNewtonManager(NewtonManager):
    """Newton manager extended for Dexsuite 3dg (e.g. custom step/reset or logging).

    Override :meth:`initialize`, :meth:`step`, :meth:`reset`, or :meth:`_simulate`
    to add task-specific behavior. Do not change IsaacLab's NewtonManager.
    """

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize the manager. Add any 3dg-specific setup here."""
        super().initialize(sim_context)
        # e.g. cls._some_3dg_state = ...

    # Uncomment and customize as needed:
    #
    # @classmethod
    # def step(cls, dt: float) -> None:
    #     """Optional: wrap step with 3dg-specific logic."""
    #     super().step(dt)
    #
    # @classmethod
    # def reset(cls, env_ids: list[int] | None = None) -> None:
    #     """Optional: wrap reset with 3dg-specific logic."""
    #     super().reset(env_ids)
