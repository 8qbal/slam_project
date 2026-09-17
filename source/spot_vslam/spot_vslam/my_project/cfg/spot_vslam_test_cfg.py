from dataclasses import MISSING
import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

# [重要] 正確的 Manager Import
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

# [重要] MDP 函數庫
import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.terminations import time_out

# [重要] 雜訊工具
import isaaclab.utils.noise as noise_utils

# 您的自定義模組
from ..manager.ros2_manager import Ros2ManagerCfg
from ..manager.orb_slam_subscriber_manager import OrbSlamSubscriberCfg
import spot_vslam.my_project.custom_mdp as custom_mdp
from ..spot_new.spot_with_camera import SPOT_CFG

# --- 計算誤差函式 ---
def get_slam_camera_error(env):
    gt_cam_pos = env.scene.sensors["camera"].data.pos_w
    if hasattr(env, "slam_subscriber_manager") and env.slam_subscriber_manager is not None:
        slam_pos = env.slam_subscriber_manager.pose_buffer[:, 0:3]
    else:
        slam_pos = torch.zeros_like(gt_cam_pos)
    dist_error = torch.norm(gt_cam_pos - slam_pos, dim=-1, keepdim=True)
    return dist_error

# =============================================================================
# 場景設定 (Scene)
# =============================================================================
@configclass
class SlamTestSceneCfg(InteractiveSceneCfg):
    num_envs = 1
    env_spacing = 5.0

    # 1. 地板
    ground = AssetBaseCfg(
        prim_path="/World/groundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            size=(25.0, 25.0), # 設定足夠大
            color=(1.0, 1.0, 1.0) # 預設白色底
        ),
    )

    # 2. 牆壁 (您的 USD)
    env_walls = AssetBaseCfg(
        prim_path="/World/EnvWalls",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/bernie/Isaac_lab/project/spot_vslam/source/spot_vslam/spot_vslam/my_project/spot_new/Collected_spot/env_walls.usd",
            copy_from_source=True,
            visible=True,
        ),
    )

    # 3. 機器人
    robot: ArticulationCfg = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 4. 相機
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/MyCamera",
        update_period=0.05, # 20Hz
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=None,
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), convention="ros"),
    )

    # 5. 燈光
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

# =============================================================================
# 觀測設定 (Observations)
# =============================================================================
@configclass
class SlamTestObservationsCfg:
    
    # [Group 1] Policy (大腦) - 48 維
    # 這是為了匹配您訓練時用的 flat_env_cfg.py
    @configclass
    class PolicyObs(ObsGroup):
        # 1. 線性速度 (3)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=noise_utils.UniformNoiseCfg(n_min=-0.1, n_max=0.1),
        )
        # 2. 角速度 (3)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=noise_utils.UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        )
        # 3. 重力投影 (3)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=noise_utils.UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
        # 4. 速度指令 (3)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        # 5. 關節位置 (12)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=noise_utils.UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        )
        # 6. 關節速度 (12)
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=noise_utils.UniformNoiseCfg(n_min=-1.5, n_max=1.5),
        )
        # 7. 上一次動作 (12)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True  # 必須設為 True 才能變成 48維 Tensor

    # [Group 2] SLAM Info (給您看的)
    @configclass
    class SlamInfoGroup(ObsGroup):
        slam_tracking_state = ObsTerm(func=custom_mdp.get_slam_tracking_state)
        slam_local_pc_count = ObsTerm(func=custom_mdp.get_slam_local_pc_count)
        slam_pos_error_m = ObsTerm(func=get_slam_camera_error)
        base_yaw_deg = ObsTerm(
            func=lambda env: torch.rad2deg(torch.atan2(
                2.0 * (env.scene["robot"].data.root_quat_w[:, 0] * env.scene["robot"].data.root_quat_w[:, 3] +
                       env.scene["robot"].data.root_quat_w[:, 1] * env.scene["robot"].data.root_quat_w[:, 2]),
                1.0 - 2.0 * (env.scene["robot"].data.root_quat_w[:, 2]**2 + env.scene["robot"].data.root_quat_w[:, 3]**2)
            )),
            scale=1.0
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False # 保持字典結構，方便讀取

    policy: PolicyObs = PolicyObs()
    slam_info: SlamInfoGroup = SlamInfoGroup()

# =============================================================================
# 指令設定 (Commands) - 防止 KeyError
# =============================================================================
@configclass
class SlamTestCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )

# =============================================================================
# Action / Termination
# =============================================================================
@configclass
class SlamTestActionsCfg:
    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=".*",
        scale=0.5,
        use_default_offset=True,
    )

@configclass
class SlamTestTerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)

# =============================================================================
# 主環境設定類別
# =============================================================================
@configclass
class SpotSlamTestEnvCfg(ManagerBasedRLEnvCfg):
    scene: SlamTestSceneCfg = SlamTestSceneCfg()
    observations: SlamTestObservationsCfg = SlamTestObservationsCfg()
    actions: SlamTestActionsCfg = SlamTestActionsCfg()
    
    # [關鍵] 加入 commands
    commands: SlamTestCommandsCfg = SlamTestCommandsCfg()
    
    terminations: SlamTestTerminationsCfg = SlamTestTerminationsCfg()

    events = None
    curriculum = None
    rewards = None

    ros2: Ros2ManagerCfg = Ros2ManagerCfg(
        camera_name="camera",
        topic_prefix="/spot"
    )

    slam_subscriber: OrbSlamSubscriberCfg = OrbSlamSubscriberCfg(
        topic_prefix_template="/spot{i}/orbslam",
        state_topic_suffix="/tracking_state",
        local_pc_topic_suffix="/local_pc",
        pose_topic_suffix="/robot_pose", 
    )

    def __post_init__(self):
        self.episode_length_s = 99999.0
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        if self.scene.camera is not None:
            self.scene.camera.update_period = self.decimation * self.sim.dt