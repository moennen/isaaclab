# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base environment configuration for the Franka VBD Cube Pick task.

Task description
----------------
A Franka Panda arm is mounted on the ground.  A deformable cube is randomly
spawned on the ground plane.  The cube is simulated via VBD (Vertex Block
Descent) physics using an extended Newton manager; it has no rigid-asset entry
in the Isaac Lab scene.

The robot must:
  - If the cube is REACHABLE:   pick it up and bring it to the success position.
  - If the cube is UNREACHABLE: move the end-effector to the signal position to
    communicate that the object cannot be reached.

Reachability is defined geometrically from the robot base position and the cube
CoM position.  It is not provided as an explicit observation.

This task is **Newton-only** (no PhysX equivalent for VBD deformable bodies).
Select Newton with ``presets=newton`` (required, not optional).

Geometry constants
------------------
All positions are in the world frame with the robot base at [0, 0, 0].

  CUBE_SPAWN_X_RANGE     = (0.0, 0.8)   m  — forward from robot base
  CUBE_SPAWN_Y_RANGE     = (-0.6, 0.6)  m  — lateral

  REACHABLE_RADIUS_MIN   = 0.22  m
  REACHABLE_RADIUS_MAX   = 0.65  m

  SUCCESS_EE_POSITION    = [0.5, 0.0, 0.5]
  SIGNAL_EE_POSITION     = [0.0, 0.0, 0.8]
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from . import mdp
from .physics.vbd_newton_cfg import FrankaVbdCubePickNewtonCfg

##
# Scene
##


@configclass
class GroundSceneCfg(InteractiveSceneCfg):
    """Scene with robot on ground.  No 'object' asset — cube is VBD soft body."""

    robot: ArticulationCfg = MISSING

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP components
##


@configclass
class ActionsCfg:
    """Action specifications — populated by robot-specific config."""

    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observations for the VBD pick-or-signal task — state-based, 47 dimensions total.

    Observation vector layout:
      cube_pos   (3): cube CoM XYZ in robot root frame [m]
      cube_quat  (4): cube Kabsch orientation wxyz in robot root frame
      ee_state  (13): EE pos(3) + quat(4) + lin_vel(3) + ang_vel(3) in robot root frame
      joint_pos  (9): arm (7) + finger (2) positions [rad or m]
      joint_vel  (9): arm (7) + finger (2) velocities [rad/s or m/s]
      actions    (9): last joint-position action
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # Cube CoM pose in robot root frame (from VBD manager obs cache)
        cube_pos = ObsTerm(
            func=mdp.cube_pos_b,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        cube_quat = ObsTerm(
            func=mdp.cube_quat_b,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )

        # EE body state in robot root frame
        ee_state = ObsTerm(
            func=mdp.ee_state_b,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )

        # Joint proprioception
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # Last action
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset and randomisation events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_cube_position = EventTerm(
        func=mdp.reset_cube_pose_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.8),
                "y": (-0.6, 0.6),
            },
            "cube_size": 0.05,  # must match FrankaVbdCubePickNewtonCfg.cube_size
        },
    )


@configclass
class RewardsCfg:
    """Two symmetric branches: pick when reachable, signal when not.

    Reachable branch:
      approach_cube   (1.0)  dense — EE closes on cube CoM
      grip_cube       (5.0)  binary — gripper closed + cube CoM > 5 cm
      lift_cube      (10.0)  binary — cube CoM > 0.5 m
      cube_at_success(15.0)  dense — EE at success pos while lifted

    Unreachable branch:
      go_to_signal    (1.0)  dense shaping
      signal_reached (10.0)  binary — EE within 5 cm of signal pos

    Penalties:
      action_rate (-1e-4), joint_vel (-1e-4)
    """

    approach_cube = RewTerm(
        func=mdp.approach_cube_reachable,
        params={"std": 0.1, "ee_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])},
        weight=1.0,
    )
    grip_cube = RewTerm(
        func=mdp.grip_cube_reachable,
        params={"grip_height": 0.05, "closed_threshold": 0.06},
        weight=5.0,
    )
    lift_cube = RewTerm(
        func=mdp.lift_cube_reachable,
        params={"lift_height": 0.5},
        weight=10.0,
    )
    cube_at_success = RewTerm(
        func=mdp.cube_at_success_position,
        params={
            "std": 0.1,
            "lift_height": 0.5,
            "ee_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
        },
        weight=15.0,
    )
    go_to_signal = RewTerm(
        func=mdp.go_to_signal_position,
        params={"std": 0.1, "ee_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])},
        weight=1.0,
    )
    signal_reached = RewTerm(
        func=mdp.signal_reached_unreachable,
        params={
            "signal_threshold": 0.05,
            "ee_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
        },
        weight=10.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Top-level environment config
##


@configclass
class FrankaVbdCubePickEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based env config for the Franka VBD cube pick + unreachable-signal task.

    The deformable cube is managed entirely by :class:`FrankaVbdCubePickNewtonManager`.
    There is no ``scene.object`` entry — the cube is not an Isaac Lab asset.

    Geometry constants are read by mdp functions so derived configs can override
    them without touching reward/observation logic.
    """

    # Reachability geometry (horizontal distance from robot base, metres)
    reachable_radius_min: float = 0.22
    reachable_radius_max: float = 0.65

    # Target EE positions (world frame, robot base at origin)
    success_ee_position: tuple = (0.5, 0.0, 0.5)
    signal_ee_position:  tuple = (0.0, 0.0, 0.8)

    # Lift threshold to consider cube "picked"
    lift_height: float = 0.5

    # --- Managers ---
    scene:        GroundSceneCfg  = GroundSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions:      ActionsCfg      = ActionsCfg()
    rewards:      RewardsCfg      = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events:       EventCfg        = EventCfg()

    def __post_init__(self):
        # 10 × 2 ms substeps = 20 ms/action (50 Hz control rate).
        # This task is Newton-only (VBD requires Newton backend).
        # Run with:  ./isaaclab.sh -p train.py --task Isaac-Pick-VBD-Cube-Franka-v0 presets=newton
        self.decimation = 10
        self.episode_length_s = 8.0
        self.sim.dt = 0.002
        self.sim.render_interval = 10
        # VBD cube physics — Newton only.
        self.sim.physics = FrankaVbdCubePickNewtonCfg(
            num_substeps=3,
            use_cuda_graph=True,
            vbd_iterations=8,
        )
