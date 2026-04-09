# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base environment configuration for the Franka Cube Pick task.

Task description
----------------
A Franka Panda arm is mounted on the ground. A rigid cube is randomly spawned
on the ground plane within a wide area that covers both reachable and unreachable
zones relative to the robot.

The robot must:
  - If the cube is REACHABLE:  pick it up and bring it to the success position.
  - If the cube is UNREACHABLE: move the end-effector to the signal position to
    communicate that the object cannot be reached.

Reachability is defined geometrically from the robot base position and the cube
position on the ground. It is not provided as an explicit observation — the robot
must infer it.

Proposed geometry (validate before finalising)
----------------------------------------------
All positions are in the world frame with the robot base at [0, 0, 0].

  CUBE_SPAWN_X_RANGE     = (0.0, 0.8)   m  — forward from robot base
  CUBE_SPAWN_Y_RANGE     = (-0.6, 0.6)  m  — lateral
  CUBE_SPAWN_Z           = 0.025        m  — cube half-height resting on ground

  REACHABLE_RADIUS_MIN   = 0.22  m  — min horizontal dist for ground-level grasp
  REACHABLE_RADIUS_MAX   = 0.65  m  — max horizontal dist for ground-level grasp

  SUCCESS_EE_POSITION    = [0.5, 0.0, 0.5]  — EE target when cube grasped & lifted
  SIGNAL_EE_POSITION     = [0.0, 0.0, 0.8]  — EE target when cube unreachable
                                               (arm pointing straight up)

These values are class-level constants on FrankaCubePickEnvCfg so that derived
configs and physics-specific extensions can override them without touching the
reward/observation logic.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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

##
# Scene
##


@configclass
class GroundSceneCfg(InteractiveSceneCfg):
    """Scene with robot on ground and a cube spawned on the ground plane.

    Robot, end-effector frame, and cube asset are left MISSING so that
    robot-specific derived configs (e.g. Franka joint-pos) can fill them in.
    This follows the same MISSING pattern as the standard lift task.
    """

    # Populated by robot-specific config
    robot: ArticulationCfg = MISSING
    object: RigidObjectCfg = MISSING

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=GroundPlaneCfg(),
    )

    # Lighting
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
    """Observations for the pick-or-signal task — dexsuite-style, state-based.

    All terms read body state directly from asset data (no FrameTransformer),
    so this works on both PhysX and Newton.

    Observation vector layout (47 dimensions total):
      cube_pos   (3): cube XYZ in robot root frame [m]
      cube_quat  (4): cube orientation wxyz in robot root frame
      ee_state  (13): EE pos(3) + quat(4) + lin_vel(3) + ang_vel(3) in robot root frame
      joint_pos  (9): arm (7) + finger (2) positions [rad or m]
      joint_vel  (9): arm (7) + finger (2) velocities [rad/s or m/s]
      actions    (9): last joint-position action
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # Cube pose in robot root frame
        cube_pos = ObsTerm(
            func=mdp.cube_pos_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        cube_quat = ObsTerm(
            func=mdp.cube_quat_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        # End-effector body state in robot root frame (pos + quat + lin_vel + ang_vel)
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

    # Spawn cube randomly on the ground across the full reachable+unreachable range.
    # x/y ranges intentionally cover both zones so the robot sees both cases.
    # z offset = 0.0 because the cube init_state already places it at half-height.
    reset_cube_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.8),    # PROPOSED — forward from robot base
                "y": (-0.6, 0.6),   # PROPOSED — lateral
                "z": (0.0, 0.0),    # zero offset (cube init_state handles height)
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Two-branch reward: pick path when reachable, signal path when not.

    PROPOSED reward structure (validate with scripted sequences before RL):

    Reachable branch (active when cube horizontal dist from base is in [R_MIN, R_MAX]):
      - approach_cube:      1 - tanh(||ee - cube|| / 0.1)       weight=1.0
      - lift_cube:          indicator(cube_z > 0.5)               weight=10.0  (mid-Franka height)
      - cube_at_success:    lift * (1 - tanh(||cube - success|| / 0.1))  weight=15.0

    Unreachable branch (active when cube is outside reachable zone):
      - go_to_signal:       1 - tanh(||ee - signal|| / 0.1)     weight=10.0

    Penalties (always active):
      - action_rate:        -||delta_action||²                   weight=-1e-4
      - joint_vel:          -||joint_vel||²                      weight=-1e-4
    """

    # --- Reachable branch ---
    approach_cube = RewTerm(
        func=mdp.approach_cube_reachable,
        params={
            "std": 0.1,
            "ee_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
        },
        weight=1.0,
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

    # --- Unreachable branch ---
    go_to_signal = RewTerm(
        func=mdp.go_to_signal_position,
        params={
            "std": 0.1,
            "ee_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
        },
        weight=10.0,
    )

    # --- Penalties ---
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

    # End early if cube falls off the world (shouldn't happen on flat ground)
    cube_fallen = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.1, "asset_cfg": SceneEntityCfg("object")},
    )


##
# Top-level environment config
##


@configclass
class FrankaCubePickEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based env config for the Franka cube pick + unreachable-signal task.

    Geometry constants
    ------------------
    These are read by the custom mdp functions (rewards, observations) so that
    derived configs and physics extensions can override them in one place.

    PROPOSED values — replace after user validation.
    """

    # Reachability geometry (horizontal distance from robot base, metres)
    reachable_radius_min: float = 0.22
    reachable_radius_max: float = 0.65

    # Target EE positions (world frame, robot base at origin)
    success_ee_position: tuple = (0.5, 0.0, 0.5)   # PROPOSED
    signal_ee_position: tuple = (0.0, 0.0, 0.8)    # PROPOSED

    # Lift threshold to consider cube "picked" — mid-Franka height (~1.0m robot, so 0.5m)
    lift_height: float = 0.5  # metres above ground

    # --- Managers ---
    scene: GroundSceneCfg = GroundSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0        # longer than lift — needs pick + transport
        self.sim.dt = 0.01                 # 100 Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
