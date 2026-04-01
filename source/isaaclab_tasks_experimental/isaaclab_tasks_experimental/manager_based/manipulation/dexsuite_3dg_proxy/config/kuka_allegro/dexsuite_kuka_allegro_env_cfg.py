# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from .physic.newton import (
    Dexsuite3dgProxyNewtonCfg,
    VbdObjectAdapter,
    VbdObjectAdapterCfg,
    contact_count_vbd,
    contacts_vbd,
    fingers_contact_force_b_vbd,
    object_ee_distance_vbd,
    object_point_cloud_b_vbd,
    orientation_command_error_tanh_vbd,
    position_command_error_tanh_vbd,
    success_reward_vbd,
)

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

from ... import dexsuite_env_cfg as dexsuite
from ...dexsuite_env_cfg import ObjectCfg
from ... import mdp
from .camera_cfg import (
    BaseTiledCameraCfg,
    DuoCameraObservationsCfg,
    SingleCameraObservationsCfg,
    StateObservationCfg,
    WristTiledCameraCfg,
)

FINGERTIP_LIST = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
THUMB_SENSOR = "thumb_link_3_object_s"
FINGER_SENSORS = [f"{name}_object_s" for name in FINGERTIP_LIST if name != "thumb_link_3"]


@configclass
class KukaAllegroPhysicsCfg(PresetCfg):
    default = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_max_rigid_patch_count=4 * 5 * 2**15,
        gpu_found_lost_pairs_capacity=2**26,
        gpu_found_lost_aggregate_pairs_capacity=2**29,
        gpu_total_aggregate_pairs_capacity=2**25,
    )
    newton = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=70,
            impratio=50.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=100,
            ls_iterations=15,
            ls_parallel=False,
            use_mujoco_contacts=True,
            ccd_iterations=200,
        ),
        num_substeps=2,
        debug_mode=False,
    )
    deformable = Dexsuite3dgProxyNewtonCfg(
        vbd_enabled=True,
        tet_mesh_path="/mnt/dev/isaac-newton3/assets/blueHairRagdollLR.msh",
        num_substeps=4,
        debug_mode=False,
    )
    physx = default


@configclass
class KukaAllegroSceneCfg(PresetCfg):
    @configclass
    class KukaAllegroSceneCfg(dexsuite.SceneCfg):
        """Kuka Allegro participant scene for Dexsuite Lifting/Reorientation"""

        robot: ArticulationCfg = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        base_camera: TiledCameraCfg | None = None

        wrist_camera: TiledCameraCfg | None = None

        def __post_init__(self: dexsuite.SceneCfg):
            super().__post_init__()
            for link_name in FINGERTIP_LIST:
                setattr(
                    self,
                    f"{link_name}_object_s",
                    ContactSensorCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + link_name,
                        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                    ),
                )

    default = KukaAllegroSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)
    mesh = default.replace(object=default.object.replace(spawn=ObjectCfg().mesh))
    single_camera = default.replace(base_camera=BaseTiledCameraCfg())
    duo_camera = default.replace(base_camera=BaseTiledCameraCfg(), wrist_camera=WristTiledCameraCfg())

    # VBD deformable: no rigid Object prim, no PhysX contact sensors.
    # The object is VBD particles; VbdObjectAdapter exposes their CoM as the
    # asset pose.  Contact detection is done via particle proximity in VBD
    # helpers (fingers_contact_force_b_vbd / contacts_vbd / contact_count_vbd).
    @configclass
    class _DeformableSceneCfg(KukaAllegroSceneCfg):
        def __post_init__(self):
            super().__post_init__()
            # Null out all PhysX fingertip-object contact sensors.
            # With no rigid Object body in Newton, filter_prim_paths_expr has no
            # match and force_matrix_w would be None → observation crash.
            for link_name in FINGERTIP_LIST:
                setattr(self, f"{link_name}_object_s", None)

    deformable = _DeformableSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True).replace(
        object=VbdObjectAdapterCfg(
            prim_path=default.object.prim_path,
            spawn=None,  # No prim — object is VBD particles only.
            init_state=default.object.init_state,
            class_type=VbdObjectAdapter,
        )
    )


@configclass
class KukaAllegroRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class KukaAllegroReorientRewardCfg(dexsuite.RewardsCfg):
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,
        params={"threshold": 0.1, "thumb_name": THUMB_SENSOR, "finger_names": FINGER_SENSORS},
    )

    contact_count = RewTerm(
        func=mdp.contact_count,
        weight=1.0,
        params={
            "threshold": 0.01,
            "sensor_names": FINGER_SENSORS + [THUMB_SENSOR],
        },
    )

    def __post_init__(self: dexsuite.RewardsCfg):
        super().__post_init__()
        self.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])
        self.fingers_to_object.params["thumb_name"] = THUMB_SENSOR
        self.fingers_to_object.params["finger_names"] = FINGER_SENSORS
        self.position_tracking.params["thumb_name"] = THUMB_SENSOR
        self.position_tracking.params["finger_names"] = FINGER_SENSORS
        if self.orientation_tracking:
            self.orientation_tracking.params["thumb_name"] = THUMB_SENSOR
            self.orientation_tracking.params["finger_names"] = FINGER_SENSORS
        self.success.params["thumb_name"] = THUMB_SENSOR
        self.success.params["finger_names"] = FINGER_SENSORS


@configclass
class KukaAllegroObservationCfg(PresetCfg):
    state = StateObservationCfg()
    single_camera = SingleCameraObservationsCfg()
    duo_camera = DuoCameraObservationsCfg()
    default = state


@configclass
class KukaAllegroEventCfg(PresetCfg):
    @configclass
    class KukaAllegroPhysxEventCfg(dexsuite.StartupEventCfg, dexsuite.EventCfg):
        pass

    @configclass
    class DeformableEventCfg(dexsuite.EventCfg):
        """Events for deformable (VBD) mode.

        Uses full gravity from episode 0 so the soft body falls correctly
        from the start.  During training the curriculum scheduler updates
        ``variable_gravity.params`` to a wider range; this just sets the
        initial value to something physically meaningful.
        """

        def __post_init__(self):
            super().__post_init__()
            # Full gravity from the start — the base EventCfg initialises to
            # ([0,0,0],[0,0,0]) for a zero-gravity curriculum warm-up, but that
            # means the ragdoll floats until the curriculum kicks in.  Deformable
            # environments start with full gravity so the physics is visible.
            self.variable_gravity.params["gravity_distribution_params"] = (
                [0.0, 0.0, -9.81],
                [0.0, 0.0, -9.81],
            )

    default = KukaAllegroPhysxEventCfg()
    newton = dexsuite.EventCfg()
    physx = default
    deformable = DeformableEventCfg()


@configclass
class KukaAllegroMixinCfg:
    scene: KukaAllegroSceneCfg = KukaAllegroSceneCfg()
    rewards: KukaAllegroReorientRewardCfg = KukaAllegroReorientRewardCfg()
    observations: KukaAllegroObservationCfg = KukaAllegroObservationCfg()
    events: KukaAllegroEventCfg = KukaAllegroEventCfg()
    actions: KukaAllegroRelJointPosActionCfg = KukaAllegroRelJointPosActionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = KukaAllegroPhysicsCfg()


@configclass
class Dexsuite3dgProxyKukaAllegroReorientEnvCfg(KukaAllegroMixinCfg, dexsuite.Dexsuite3dgProxyReorientEnvCfg):
    pass


@configclass
class Dexsuite3dgProxyKukaAllegroReorientEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.Dexsuite3dgProxyReorientEnvCfg_PLAY):
    pass


@configclass
class Dexsuite3dgProxyKukaAllegroLiftEnvCfg(KukaAllegroMixinCfg, dexsuite.Dexsuite3dgProxyLiftEnvCfg):
    pass


@configclass
class Dexsuite3dgProxyKukaAllegroLiftEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.Dexsuite3dgProxyLiftEnvCfg_PLAY):
    pass


# ---------------------------------------------------------------------------
# Deformable (VBD soft body) variants
# ---------------------------------------------------------------------------

# Contact threshold for particle-proximity contact detection.
# Must match particle_radius used in Dexsuite3dgProxyNewtonCfg (default 0.005 m).
_VBD_CONTACT_THRESHOLD = 0.01  # 2 × particle_radius


@configclass
class KukaAllegroDeformableObservationCfg(KukaAllegroObservationCfg):
    """Observations for VBD deformable mode.

    Replaces USD-prim-dependent observations with VBD-native equivalents:
    - ``proprio.contact``: particle-proximity instead of PhysX contact sensors.
    - ``perception.object_point_cloud``: particle positions instead of USD mesh sampling.
    """

    @configclass
    class _DeformableStateObsCfg(StateObservationCfg):
        def __post_init__(self):
            super().__post_init__()
            # Replace PhysX sensor-based contact with VBD particle-proximity.
            self.proprio.contact = ObsTerm(
                func=fingers_contact_force_b_vbd,
                params={
                    "fingertip_names": FINGERTIP_LIST,
                    "contact_threshold": _VBD_CONTACT_THRESHOLD,
                    "signal_magnitude": 1.0,
                },
                clip=(-20.0, 20.0),
            )
            # Replace USD prim surface sampling with VBD particle positions.
            self.perception.object_point_cloud = ObsTerm(
                func=object_point_cloud_b_vbd,
                noise=Unoise(n_min=0.0, n_max=0.0),  # starts at zero; ADR curriculum ramps it up
                clip=(-2.0, 2.0),
                params={"num_points": 64, "flatten": True},
            )

    state = _DeformableStateObsCfg()
    default = state


_VBD_THUMB = "thumb_link_3"
_VBD_FINGERS = ["index_link_3", "middle_link_3", "ring_link_3"]


@configclass
class KukaAllegroDeformableRewardCfg(KukaAllegroReorientRewardCfg):
    """Rewards for VBD deformable mode.

    All contact-gated rewards use particle-proximity instead of PhysX sensors.
    """

    def __post_init__(self):
        super().__post_init__()

        # Contact density rewards
        self.good_finger_contact = RewTerm(
            func=contacts_vbd,
            weight=0.5,
            params={
                "threshold": 0.5,
                "thumb_name": _VBD_THUMB,
                "finger_names": _VBD_FINGERS,
                "contact_threshold": _VBD_CONTACT_THRESHOLD,
            },
        )
        self.contact_count = RewTerm(
            func=contact_count_vbd,
            weight=1.0,
            params={
                "threshold": 0.5,
                "fingertip_names": FINGERTIP_LIST,
                "contact_threshold": _VBD_CONTACT_THRESHOLD,
            },
        )

        # Reach + contact gated rewards — use link names instead of sensor names
        self.fingers_to_object = RewTerm(
            func=object_ee_distance_vbd,
            weight=self.fingers_to_object.weight,
            params={
                "std": 0.4,
                "thumb_name": _VBD_THUMB,
                "finger_names": _VBD_FINGERS,
                "contact_threshold": _VBD_CONTACT_THRESHOLD,
                "asset_cfg": SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"]),
            },
        )
        self.position_tracking = RewTerm(
            func=position_command_error_tanh_vbd,
            weight=self.position_tracking.weight,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "std": 0.2,
                "command_name": "object_pose",
                "align_asset_cfg": SceneEntityCfg("object"),
                "thumb_name": _VBD_THUMB,
                "finger_names": _VBD_FINGERS,
                "contact_threshold": _VBD_CONTACT_THRESHOLD,
            },
        )
        if self.orientation_tracking is not None:
            self.orientation_tracking = RewTerm(
                func=orientation_command_error_tanh_vbd,
                weight=self.orientation_tracking.weight,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "std": 1.5,
                    "command_name": "object_pose",
                    "align_asset_cfg": SceneEntityCfg("object"),
                    "thumb_name": _VBD_THUMB,
                    "finger_names": _VBD_FINGERS,
                    "contact_threshold": _VBD_CONTACT_THRESHOLD,
                },
            )
        self.success = RewTerm(
            func=success_reward_vbd,
            weight=self.success.weight,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pos_std": 0.1,
                "rot_std": 0.5,
                "command_name": "object_pose",
                "align_asset_cfg": SceneEntityCfg("object"),
                "thumb_name": _VBD_THUMB,
                "finger_names": _VBD_FINGERS,
                "contact_threshold": _VBD_CONTACT_THRESHOLD,
            },
        )


@configclass
class KukaAllegroDeformableMixinCfg(KukaAllegroMixinCfg):
    """Mixin that selects the deformable physics + scene + obs + rewards presets."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = KukaAllegroPhysicsCfg().deformable
        self.scene = KukaAllegroSceneCfg().deformable
        # Full gravity from episode 0 for VBD soft body; no PhysX startup events.
        self.events = KukaAllegroEventCfg().deformable
        # VBD-native contact observation and rewards (no PhysX sensors).
        self.observations = KukaAllegroDeformableObservationCfg()
        self.rewards = KukaAllegroDeformableRewardCfg()
        # ADR curriculum: start gravity at full -9.81 (not zero) so the soft body
        # falls from episode 1.  During training the curriculum can widen the range.
        if self.curriculum is not None:
            self.curriculum.gravity_adr.params["modify_params"]["initial_value"] = (
                (0.0, 0.0, -9.81),
                (0.0, 0.0, -9.81),
            )


@configclass
class Dexsuite3dgProxyKukaAllegroDeformableReorientEnvCfg(
    KukaAllegroDeformableMixinCfg, dexsuite.Dexsuite3dgProxyReorientEnvCfg
):
    pass


@configclass
class Dexsuite3dgProxyKukaAllegroDeformableReorientEnvCfg_PLAY(
    KukaAllegroDeformableMixinCfg, dexsuite.Dexsuite3dgProxyReorientEnvCfg_PLAY
):
    pass


@configclass
class Dexsuite3dgProxyKukaAllegroDeformableLiftEnvCfg(
    KukaAllegroDeformableMixinCfg, dexsuite.Dexsuite3dgProxyLiftEnvCfg
):
    pass


@configclass
class Dexsuite3dgProxyKukaAllegroDeformableLiftEnvCfg_PLAY(
    KukaAllegroDeformableMixinCfg, dexsuite.Dexsuite3dgProxyLiftEnvCfg_PLAY
):
    pass
