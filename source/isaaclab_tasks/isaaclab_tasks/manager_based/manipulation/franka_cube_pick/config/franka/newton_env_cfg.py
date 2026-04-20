# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-physics env cfg for the Franka cube pick task.

Physics parameters match the validated standalone Newton simulation exactly:
  scripts/generate_sequences.py  (SolverMuJoCo kwargs + ShapeConfig)
  scripts/replay_sequences.py    (validation cross-check)

Solver (MJWarpSolverCfg)
------------------------
  solver        = "newton"          (MuJoCo Newton solver)
  integrator    = "implicitfast"    (semi-implicit, stable for stiff contacts)
  iterations    = 20                (solver iterations per substep)
  ls_parallel   = True              (parallel linesearch, matches generate_sequences)
  ls_iterations = 100               (linesearch budget)
  cone          = "elliptic"        (elliptic friction cone)
  impratio      = 1000.0            (impedance ratio — stiff grasping)

Substeps / timing
-----------------
  num_substeps  = 10                (matches _N_SUBSTEPS = 10)
  sim.dt        = 0.002 s           (2 ms per substep)
  decimation    = 10                → 20 ms outer step → 50 Hz control rate

Contact material (cube)
-----------------------
  static_friction  = dynamic_friction = 0.75   (_CONTACT_MU  = 0.75)
  restitution      = 0.0                        (fully inelastic)
  compliant_contact_stiffness = 5e4            (_CONTACT_KE  = 5e4 N/m)
  compliant_contact_damping   = 5e2            (_CONTACT_KD  = 5e2 N·s/m)

Training command (Newton kit required):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Isaac-Pick-Cube-Franka-Newton-v0 \\
        --num_envs 4096 \\
        --headless \\
        --experience apps/isaacsim_5/isaaclab.python.headless.newton.kit
"""

from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from .joint_pos_env_cfg import FrankaCubePickEnvCfg


@configclass
class FrankaCubePickNewtonEnvCfg(FrankaCubePickEnvCfg):
    """Franka cube-pick env using the Newton physics backend.

    Inherits robot, cube, actions, observations, rewards, and terminations from
    the joint-position env cfg.  Replaces the PhysX backend with Newton and
    overrides simulation timing and contact parameters to match the validated
    standalone Newton simulation (generate_sequences.py / replay_sequences.py).
    """

    def __post_init__(self):
        super().__post_init__()

        # --- Timing: 10 × 2 ms substeps = 20 ms outer step = 50 Hz control ----
        # Matches generate_sequences.py: _N_SUBSTEPS=10, dt=0.002 s.
        self.sim.dt = 0.002       # 2 ms per Newton substep
        self.decimation = 10      # 10 substeps per RL action step
        self.sim.render_interval = self.decimation

        # --- Newton physics backend -------------------------------------------
        # All solver kwargs taken verbatim from the validated SolverMuJoCo call
        # in generate_sequences.py (build_batched_model / solver construction).
        #
        # njmax / nconmax: per-env contact budget.
        # Franka + 1 cube geometry is simple; 150 constraints and 40 contacts
        # per env is generous headroom.  Increase if Newton logs overflow warnings.
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",          # Newton conjugate-gradient solver
                integrator="implicitfast",# semi-implicit fast integrator
                iterations=20,            # solver iterations per substep
                ls_parallel=True,         # parallel linesearch (matches standalone)
                ls_iterations=100,        # linesearch iterations
                cone="elliptic",          # elliptic friction cone
                impratio=1000.0,          # impedance ratio (stiff normal constraint)
                njmax=150,                # max constraints per env
                nconmax=40,               # max contacts per env
                use_mujoco_contacts=True, # MuJoCo internal contact detection
            ),
            num_substeps=10,  # 10 inner substeps per sim.dt outer step
            use_cuda_graph=True,
        )

        # --- Cube contact material: match validated Newton ShapeConfig ---------
        # generate_sequences.py sets per-shape: ke=5e4, kd=5e2, kf=1e3, mu=0.75.
        # In IsaacLab Newton, ke/kd come from compliantContactStiffness/Damping
        # (SchemaResolverPhysx maps physxMaterial attrs → Newton shape params).
        # mu=0.75 and restitution=0.0 are already in the inherited cube config;
        # we add ke and kd here.
        self.scene.object.spawn.physics_material = RigidBodyMaterialCfg(
            static_friction=0.75,            # _CONTACT_MU  = 0.75
            dynamic_friction=0.75,
            restitution=0.0,                 # fully inelastic
            compliant_contact_stiffness=5e4, # _CONTACT_KE  = 5e4 N/m
            compliant_contact_damping=5e2,   # _CONTACT_KD  = 5e2 N·s/m
        )


@configclass
class FrankaCubePickNewtonEnvCfg_PLAY(FrankaCubePickNewtonEnvCfg):
    """Smaller Newton scene for interactive play / debugging."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
