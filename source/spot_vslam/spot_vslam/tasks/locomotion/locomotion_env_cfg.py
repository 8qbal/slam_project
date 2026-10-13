"""Depth-aware Spot locomotion (the low-level policy): walk at a commanded forward speed and yaw rate while seeing
the scene ahead through a downsampled depth image.

The observation is 48 proprioceptive dims + pooled depth + 5 depth statistics: 353 dims with the 60x80 depth camera
used here (paper Table 2). Timing follows paper Table 1: sim dt 0.002 s, decimation 20 (25 Hz control).
Robot/version specific configs (camera resolution, rewards, arenas) live in ``config/spot``.
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab_tasks.contrib.velocity.config.spot.mdp as spot_mdp

from isaaclab_assets.robots.spot import SPOT_CFG

from spot_vslam.slam import Ros2ManagerCfg

from . import mdp

##
# Scene
##

TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.7),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.01, 0.03), noise_step=0.01, border_width=0.25
        ),
    },
)
"""5x5 tiles of 8 m: 70 % flat, 30 % slightly rough."""

GROUND_MATERIAL_CFG = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=1.0,
    dynamic_friction=1.0,
)

CAMERA_PINHOLE_CFG = sim_utils.PinholeCameraCfg(
    focal_length=10.0, focus_distance=400.0, horizontal_aperture=22.0, clipping_range=(0.1, 10.0)
)
CAMERA_OFFSET_CFG = CameraCfg.OffsetCfg(pos=(0.4, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros")
"""Both cameras sit 0.4 m ahead of the body origin, looking forward (the stock spot.usd has no camera prim)."""

DEPTH_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/body/train_camera",
    update_period=0.04,  # Table 1: depth update period
    height=60,
    width=80,
    data_types=["distance_to_image_plane"],
    spawn=CAMERA_PINHOLE_CFG,
    offset=CAMERA_OFFSET_CFG,
)
"""Depth camera of the policy observation. The paper renders 320x240 and resizes to 60x80; rendering 80x60 directly
gives the same field of view and observation layout at a fraction of the render cost.

In Isaac Lab 3.0 a depth-only camera in the scene makes every camera's RGB output black, so whenever the RGB-D camera
(``SLAM_CAMERA_CFG``) is in the scene this camera must also render RGB: use ``DEPTH_CAMERA_WITH_RGB_CFG``."""

DEPTH_CAMERA_WITH_RGB_CFG = DEPTH_CAMERA_CFG.replace(data_types=["rgb", "distance_to_image_plane"])

SLAM_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/body/MyCamera",
    update_period=0.1,
    height=240,
    width=320,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=CAMERA_PINHOLE_CFG,
    offset=CAMERA_OFFSET_CFG,
)
"""320x240 RGB-D camera for ORB-SLAM3 (via ``Ros2Manager``) and TSDF mapping. The policy never reads it."""

STEREO_BASELINE = 0.12
"""Distance between the left (``SLAM_CAMERA_CFG``) and right SLAM camera in metres, as on a ZED 2."""

SLAM_RIGHT_CAMERA_CFG = SLAM_CAMERA_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot/body/SlamRightCamera",
    data_types=["rgb"],
    offset=CAMERA_OFFSET_CFG.replace(pos=(0.4, -STEREO_BASELINE, 0.0)),  # body y points left
)
"""Right camera of the stereo pair: ``SLAM_CAMERA_CFG`` is the left one, this one sits ``STEREO_BASELINE`` to its
right with the same intrinsics, orientation and update period, so the pair is rectified."""


@configclass
class SpotSceneCfg(InteractiveSceneCfg):
    """Spot on generated terrain, with the policy's depth camera and an optional SLAM camera."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TERRAIN_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=GROUND_MATERIAL_CFG,
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.75, 0.9),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = SPOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot", init_state=SPOT_CFG.init_state.replace(pos=(0.0, 2.0, 0.63))
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    train_camera: CameraCfg = DEPTH_CAMERA_CFG
    tilted_camera: CameraCfg | None = None
    right_camera: CameraCfg | None = None


##
# MDP
##


@configclass
class CommandsCfg:
    """Forward speed 0-1 m/s and yaw rate +-0.4 rad/s, the high-level policy's action range."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.4, 0.4), heading=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Order matters: trained policies depend on it. ``actions`` and ``velocity_commands`` keep these names so the
        navigation task's action term can feed them."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        camera_depth = ObsTerm(func=mdp.camera_depth_obs, params={"sensor_cfg": SceneEntityCfg("train_camera")})
        camera_depth_stats = ObsTerm(func=mdp.camera_depth_stats, params={"sensor_cfg": SceneEntityCfg("train_camera")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-2.5, 2.5),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-1.5, 1.5),
                "y": (-1.0, 1.0),
                "z": (-0.5, 0.5),
                "roll": (-0.7, 0.7),
                "pitch": (-0.7, 0.7),
                "yaw": (-1.0, 1.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"asset_cfg": SceneEntityCfg("robot"), "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Isaac Lab's Spot locomotion rewards plus two depth terms (the rl_vslam_v0 reward)."""

    # -- task
    air_time = RewTerm(
        func=spot_mdp.air_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    base_angular_velocity = RewTerm(
        func=spot_mdp.base_angular_velocity_reward,
        weight=5.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewTerm(
        func=spot_mdp.base_linear_velocity_reward,
        weight=5.0,
        params={"std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_clearance = RewTerm(
        func=spot_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    visual_wall_penalty = RewTerm(
        func=mdp.visual_wall_penalty,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("train_camera"), "safe_distance": 0.6},
    )
    open_space_seeker = RewTerm(
        func=mdp.open_space_seeker_reward, weight=0.5, params={"sensor_cfg": SceneEntityCfg("train_camera")}
    )
    gait = RewTerm(
        func=spot_mdp.GaitReward,
        weight=12.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # -- penalties
    action_smoothness = RewTerm(func=spot_mdp.action_smoothness_penalty, weight=-1.0)
    air_time_variance = RewTerm(
        func=spot_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    base_motion = RewTerm(func=spot_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")})
    base_orientation = RewTerm(
        func=spot_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    foot_slip = RewTerm(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    joint_acc = RewTerm(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )
    joint_pos = RewTerm(
        func=spot_mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )
    joint_torques = RewTerm(
        func=spot_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewTerm(
        func=spot_mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body"]), "threshold": 1.0},
    )
    leg_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*leg"]), "threshold": 1.0},
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment
##


@configclass
class LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    scene: SpotSceneCfg = SpotSceneCfg(num_envs=4096, env_spacing=0.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    ros2: Ros2ManagerCfg | None = None
    """Publish ``scene.tilted_camera`` to ROS 2 / ORB-SLAM3 (see ``spot_vslam.envs.ManagerBasedRLEnv``)."""

    def __post_init__(self):
        self.decimation = 20
        self.episode_length_s = 20.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = GROUND_MATERIAL_CFG
        self.sim.physics = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(10.5, 10.5, 0.3), lookat=(0.0, 0.0, 0.0))
        # The trained policies were trained with these sensor periods (set from an earlier dt = 0.005 s, decimation 4),
        # not with one step: keep them.
        self.scene.height_scanner.update_period = 0.02
        self.scene.contact_forces.update_period = 0.005


@configclass
class LocomotionEnvCfg_PLAY(LocomotionEnvCfg):
    """Few envs, no observation noise, endless episodes, SLAM camera published to ROS 2."""

    ros2: Ros2ManagerCfg | None = Ros2ManagerCfg(camera_name="tilted_camera", topic_prefix="/spot")

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.episode_length_s = 10000.0
        self.observations.policy.enable_corruption = False
        self.scene.train_camera = DEPTH_CAMERA_WITH_RGB_CFG
        self.scene.tilted_camera = SLAM_CAMERA_CFG
