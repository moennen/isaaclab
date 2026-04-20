# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics module for the Franka VBD cube pick task.

Provides:
  - ``FrankaVbdCubePickNewtonCfg``: Newton config with VBD material parameters.
  - ``FrankaVbdCubePickNewtonManager``: Extended NewtonManager with VBD soft-body cube.
  - ``apply_soft_body_reactions``: Two-way coupling kernel (shared with validation tools).
"""

from .vbd_coupling import apply_soft_body_reactions
from .vbd_newton_cfg import FrankaVbdCubePickNewtonCfg
from .vbd_newton_manager import FrankaVbdCubePickNewtonManager
