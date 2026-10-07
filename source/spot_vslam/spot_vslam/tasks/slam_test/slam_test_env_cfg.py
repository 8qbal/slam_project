"""SLAM test env: Spot standing in the warehouse with a 640x480 RGB-D camera published to ROS 2 for ORB-SLAM3.

The policy group is the 48-dim proprioception of a blind Isaac Lab Spot policy; the scripts drive the robot.
``slam_info`` reports the ORB-SLAM3 state for logging.
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab.utils.noise as noise_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from spot_vslam.assets import WAREHOUSE_USD_PATH
from spot_vslam.assets.spot import SPOT_CFG
from spot_vslam.slam import OrbSlamSubscriberCfg, Ros2ManagerCfg

from . import mdp


@configclass
class SlamTestSceneCfg(InteractiveSceneCfg):
    num_envs = 1
    env_spacing = 5.0

    # a generated flat floor: GroundPlaneCfg needs IsaacLab/Environments/Grid/default_ground_plane.usda, which is not
    # in the local asset pack
    ground = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen.TerrainGeneratorCfg(
            size=(25.0, 25.0),
            border_width=0.0,
            num_rows=1,
            num_cols=1,
            use_cache=False,
            sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
        ),
        debug_vis=False,
    )
    env_walls = AssetBaseCfg(
        prim_path="/World/EnvWalls",
        spawn=sim_utils.UsdFileCfg(usd_path=WAREHOUSE_USD_PATH, copy_from_source=True, visible=True),
    )
    robot: ArticulationCfg = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/MyCamera",
        update_period=0.05,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0, focus_distance=400.0, horizontal_aperture=22.0, clipping_range=(0.1, 10.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.4, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """48 dims, the layout of Isaac Lab's flat-terrain Spot policy."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=noise_utils.UniformNoiseCfg(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=noise_utils.UniformNoiseCfg(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=noise_utils.UniformNoiseCfg(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=noise_utils.UniformNoiseCfg(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=noise_utils.UniformNoiseCfg(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SlamInfoCfg(ObsGroup):
        slam_tracking_state = ObsTerm(func=mdp.get_slam_tracking_state)
        slam_local_pc_count = ObsTerm(func=mdp.get_slam_local_pc_count)
        slam_pos_error_m = ObsTerm(func=mdp.get_slam_camera_error)
        base_yaw_deg = ObsTerm(func=mdp.base_yaw_deg)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    slam_info: SlamInfoCfg = SlamInfoCfg()


@configclass
class CommandsCfg:
    """A zero velocity command that never resamples: the scripts drive the robot."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=".*", scale=0.5, use_default_offset=True)


@configclass
class RewardsCfg:
    """No rewards. Isaac Lab 3.0 crashes on ``rewards = None``, so this stays an empty config."""


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class SlamTestEnvCfg(ManagerBasedRLEnvCfg):
    scene: SlamTestSceneCfg = SlamTestSceneCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    ros2: Ros2ManagerCfg = Ros2ManagerCfg(camera_name="camera", topic_prefix="/spot")
    slam_subscriber: OrbSlamSubscriberCfg = OrbSlamSubscriberCfg()

    def __post_init__(self):
        self.episode_length_s = 99999.0
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.camera.update_period = self.decimation * self.sim.dt
