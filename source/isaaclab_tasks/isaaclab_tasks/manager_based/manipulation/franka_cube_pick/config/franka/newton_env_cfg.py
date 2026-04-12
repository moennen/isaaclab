# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-physics env cfg for the Franka cube pick task.

Physics parameters match the validated standalone Newton simulation exactly
(scripts/generate_sequences.py, scripts/replay_sequences.py):

  Solver:   SolverMuJoCo  solver="newton"  integrator="implicitfast"
            iterations=20  ls_iterations=100  ls_parallel=True
            cone="elliptic"  impratio=1000
  Timing:   10 × 2 ms substeps → 20 ms per action → 50 Hz control rate
  Contact:  ke=5e4  kd=5e2  kf=1e3  mu=0.75
  Cube:     5 cm side (half-size=0.025 m)  mass=0.1 kg  friction=0.75  restitution=0.0

Activation
----------
Newton physics backend is currently TODO in the develop branch of IsaacLab.
When ``NewtonCfg`` / ``NewtonPhysicsManager`` land upstream, uncomment the
``self.sim.physics = NewtonCfg(...)`` block in ``FrankaCubePickNewtonEnvCfg``
below and train with the Newton kit:

    cd /path/to/IsaacLab
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Pick-Cube-Franka-Newton-v0 \\
        --num_envs 4096 \\
        --headless \\
        --experience apps/isaacsim_5/isaaclab.python.headless.newton.kit

Until then the env runs PhysX with Newton-matched timing (10 × 2 ms substeps)
and cube material parameters (mass=0.1 kg, mu=0.75, restitution=0.0).
"""

from isaaclab.utils import configclass

from .joint_pos_env_cfg import FrankaCubePickEnvCfg


@configclass
class FrankaCubePickNewtonEnvCfg(FrankaCubePickEnvCfg):
    """Franka cube-pick env configured for Newton physics backend.

    Inherits robot, cube, actions, observations, rewards, and terminations from
    the joint-position env cfg.  Overrides simulation timing to match Newton's
    validated 10 × 2 ms substep structure (same 50 Hz control rate).

    Cube material parameters already match Newton validation:
      mass=0.1 kg  static_friction=dynamic_friction=0.75  restitution=0.0
    """

    def __post_init__(self):
        super().__post_init__()

        # --- Timing: match Newton's 10 × 2ms substep structure exactly ---------
        # Newton (generate_sequences.py): _N_SUBSTEPS=10 at 2ms each
        #   → outer step = 20ms → 50 Hz control rate
        # PhysX mirror: sim.dt=2ms, decimation=10 → same outer rate & same number
        #   of sub-steps, giving the closest possible contact-integration match.
        self.sim.dt = 0.002          # 2 ms substep  (Newton: _N_SUBSTEPS × 2ms)
        self.decimation = 10         # 10 substeps per action → 20 ms outer step
        self.sim.render_interval = self.decimation

        # --- Newton solver (activate when NewtonCfg lands on develop) ----------
        # All parameters taken verbatim from the validated standalone simulation
        # (scripts/generate_sequences.py, scripts/_common/sequence_schema.py).
        #
        # TODO: uncomment and replace the PhysxCfg already set by super().__post_init__()
        # once isaaclab_newton.physics exposes NewtonCfg / MJWarpSolverCfg:
        #
        #   from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
        #   self.sim.physics = NewtonCfg(
        #       solver_cfg=MJWarpSolverCfg(
        #           solver="newton",          # MuJoCo Newton solver
        #           integrator="implicitfast",
        #           iterations=20,            # solver iterations per substep
        #           ls_parallel=True,         # parallel linesearch
        #           ls_iterations=100,
        #           cone="elliptic",          # friction cone model
        #           impratio=1000.0,          # impedance ratio (stiff grasping)
        #           njmax=1000,               # max contacts per world (per env)
        #           nconmax=512,
        #       ),
        #       num_substeps=10,              # inner substeps (CUDA-graph captured)
        #       use_cuda_graph=True,
        #   )
        #
        # Robot PD gains / armature / effort limits are baked into the URDF via
        # build_robot_builder() in generate_sequences.py.  In the Manager-Based
        # RL env the Franka actuators are controlled through the ImplicitActuator
        # / JointPositionAction stack; no extra tuning is needed here.
        #
        # Gravity compensation is handled per-joint in Newton via
        # mujoco:jnt_actgravcomp; in IsaacLab it is approximated by the
        # articulation's gravity-compensation flag (enabled by default in URDF).


@configclass
class FrankaCubePickNewtonEnvCfg_PLAY(FrankaCubePickNewtonEnvCfg):
    """Smaller Newton scene for interactive play / debugging."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
