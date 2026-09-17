import math
import torch
import numpy as np
from dataclasses import MISSING
from typing import List, Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ViewerCfg, ManagerBasedRLEnvCfg
from spot_vslam.my_project.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, CameraCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensor

import isaaclab.terrains as terrain_gen # [新增] 地形生成函式庫
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_spot_env_cfg import LocomotionVelocityRoughEnvCfg

from .spot_new.spot_with_camera import SPOT_CFG

# ▼▼▼ 導入自定義管理器與 ROS2 邏輯 ▼▼▼
from spot_vslam.my_project.manager.ros2_manager import Ros2Manager, Ros2ManagerCfg

# ==========================================================
# 定義鵝卵石地形 (COBBLESTONE_ROAD_CFG)
# ==========================================================
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
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
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),
    },
)

# ==========================================================
# 自訂 MDP 函數區 (加入策略一：上帝視角造假法)
# ==========================================================

def get_orb_slam_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """智慧切換：訓練時用物理引擎作弊，Play時才讀取真實 SLAM"""
    if env.num_envs > 4:
        # 1. 抓取上帝視角的真實座標與姿態
        pos = env.scene["robot"].data.root_pos_w.clone()
        quat = env.scene["robot"].data.root_quat_w.clone()
        
        # 2. 加入微小的高斯雜訊，模擬 SLAM 飄移
        pos += torch.randn_like(pos) * 0.02 
        
        return torch.cat([pos, quat], dim=-1) # Shape: (num_envs, 7)
    else:
        if hasattr(env, "orb_slam_res"):
            return env.orb_slam_res["pose"]
        return torch.zeros((env.num_envs, 7), device=env.device)

def get_orb_slam_status(env: ManagerBasedRLEnv) -> torch.Tensor:
    """智慧切換：訓練時假裝 SLAM 狀態永遠完美 (1)"""
    if env.num_envs > 4:
        return torch.ones((env.num_envs, 1), device=env.device)
    else:
        if hasattr(env, "orb_slam_res"):
            return env.orb_slam_res["status"]
        return torch.zeros((env.num_envs, 1), device=env.device)

def forward_depth_scan(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """讓機器人真正「看見」前方的牆壁距離"""
    sensor = env.scene.sensors[sensor_cfg.name]
    hits_w = sensor.data.ray_hits_w.clone()
    robot_pos = env.scene["robot"].data.root_pos_w.unsqueeze(1)
    
    distances = torch.norm(hits_w - robot_pos, dim=-1)
    distances = torch.nan_to_num(distances, nan=5.0, posinf=5.0, neginf=5.0)
    distances = torch.clamp(distances, min=0.0, max=5.0)
    
    normalized_depth = (distances / 5.0) * 2.0 - 1.0
    return normalized_depth

# ==========================================================
# 原有的地圖管理與感測器函數
# ==========================================================

class VisualCoverageManager:
    def __init__(self, map_size=40.0, resolution=0.2, device="cuda"):
        self.map_size = map_size
        self.resolution = resolution
        self.grid_dim = int(map_size / resolution)
        self.device = device
        self.maps = None 
        self.initialized = False

    def init_maps(self, num_envs):
        if not self.initialized or self.maps.shape[0] != num_envs:
            self.maps = torch.zeros((num_envs, self.grid_dim, self.grid_dim), dtype=torch.bool, device=self.device)
            self.initialized = True

    def reset_idx(self, env_ids):
        if self.initialized and len(env_ids) > 0:
            self.maps[env_ids] = False

    def compute_reward(self, env, sensor_cfg: SceneEntityCfg):
        if not self.initialized: self.init_maps(env.num_envs)
        reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0: self.reset_idx(reset_env_ids)
        sensor = env.scene.sensors[sensor_cfg.name]
        hits = sensor.data.ray_hits_w.clone()
        half_size = self.map_size / 2.0
        grid_indices = ((hits[..., :2] + half_size) / self.resolution).long()
        x_idx, y_idx = grid_indices[..., 0], grid_indices[..., 1]
        valid_mask = (x_idx >= 0) & (x_idx < self.grid_dim) & (y_idx >= 0) & (y_idx < self.grid_dim)
        num_envs, num_rays = x_idx.shape
        batch_indices = torch.arange(num_envs, device=self.device).unsqueeze(1).expand(-1, num_rays)
        valid_batch, valid_x, valid_y = batch_indices[valid_mask], x_idx[valid_mask], y_idx[valid_mask]
        if len(valid_batch) == 0: return torch.zeros(num_envs, device=self.device)
        is_visited = self.maps[valid_batch, valid_x, valid_y]
        new_visits = ~is_visited
        self.maps[valid_batch[new_visits], valid_x[new_visits], valid_y[new_visits]] = True
        rewards = torch.zeros(num_envs, device=self.device)
        new_visit_envs = valid_batch[new_visits]
        ones = torch.ones_like(new_visit_envs, dtype=torch.float)
        rewards.scatter_add_(0, new_visit_envs, ones)
        return rewards

COVERAGE_MANAGER = VisualCoverageManager(map_size=20.0, resolution=0.2)
def visual_coverage_reward(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return COVERAGE_MANAGER.compute_reward(env, sensor_cfg)

# ==========================================================
# 環境配置 (整合 SLAM 與 ROS2)
# ==========================================================

@configclass
class SpotObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
        
        # RL 看牆壁用的深度射線
        forward_depth = ObsTerm(func=forward_depth_scan, params={"sensor_cfg": SceneEntityCfg("camera_frustum")})
        
        # [移除] 已經把 height_scan (底部射線) 拔掉了
        
        # SLAM 回傳 Observations
        orb_pose = ObsTerm(func=get_orb_slam_pose)
        orb_status = ObsTerm(func=get_orb_slam_status)

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

from .simulate_vslam_env_cfg import SpotActionsCfg, SpotCommandsCfg, SpotTerminationsCfg, SpotEventCfg

# ==========================================================
# 自訂 Reward 函數區
# ==========================================================

def forward_moving_reward(env) -> torch.Tensor:
    """只有在「走直線」時，前進才有高分！轉彎會沒收前進獎勵"""
    base_vel = env.scene["robot"].data.root_lin_vel_b
    yaw_vel = torch.abs(env.scene["robot"].data.root_ang_vel_b[:, 2])
    
    fwd_vel = torch.clamp(base_vel[:, 0], min=0.0, max=0.5)
    
    # [關鍵魔法] 轉速越快，直線倍率越低。
    # 如果轉速是 0，倍率是 1.0 (拿滿分)。如果一直轉彎，倍率會趨近於 0 (做白工)。
    straight_factor = torch.exp(-3.0 * yaw_vel)
    
    return fwd_vel * straight_factor

def stand_still_penalty_no_cmd(env, threshold: float) -> torch.Tensor:
    """怠惰懲罰：嚴格要求必須『往前走』，原地扭動一律視為偷懶！"""
    base_vel = env.scene["robot"].data.root_lin_vel_b
    fwd_vel = base_vel[:, 0]
    
    yaw_vel = torch.abs(env.scene["robot"].data.root_ang_vel_b[:, 2])
    # 拔除 yaw_vel 的漏洞，只要前進速度不達標，直接開罰
    # is_lazy = (fwd_vel < threshold) & (yaw_vel < 0.2)
    is_lazy = (fwd_vel < threshold)
    return is_lazy.float()

def close_to_wall_penalty(env, sensor_cfg: SceneEntityCfg, safe_distance: float) -> torch.Tensor:
    """防撞力場：拔除平方，改用直接線性痛覺，一靠近就讓牠痛不欲生"""
    sensor = env.scene.sensors[sensor_cfg.name]
    hits_w = sensor.data.ray_hits_w.clone()
    robot_pos = env.scene["robot"].data.root_pos_w.unsqueeze(1)
    
    distances = torch.norm(hits_w - robot_pos, dim=-1)
    distances = torch.nan_to_num(distances, nan=5.0, posinf=5.0, neginf=5.0)
    min_dist, _ = torch.min(distances, dim=1)
    
    # 計算侵入安全距離的程度
    violation = torch.clamp(safe_distance - min_dist, min=0.0)
    
    # [修改] 拔除 ** 2，直接回傳 violation。配合底下調高的權重，效果會非常顯著！
    return violation

def no_fly(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    懲罰四腳騰空 (Four legs off ground)。
    如果四隻腳同時沒有接觸力，回傳 1.0。
    """
    # 1. 取得接觸感測器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 2. 取得最近一步的接觸力
    # net_forces_w_history: (env, history, bodies, 3) -> 取最後一幀 (env, bodies, 3)
    current_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    
    # 3. 計算每隻腳的受力大小
    forces_norm = torch.norm(current_forces, dim=-1)
    
    # 4. 判斷是否懸空 (受力 < 1.0)
    is_air = forces_norm < 1.0
    
    # 5. 如果 "所有" (all) 腳都在空中，就是違規
    all_legs_in_air = torch.all(is_air, dim=1)
    
    return all_legs_in_air.float()

def yaw_rate_penalty(env) -> torch.Tensor:
    """懲罰無意義的打轉，迫使機器人走直線"""
    # 取得狗狗身體的 Z 軸旋轉速度 (Yaw Rate)
    yaw_vel = env.scene["robot"].data.root_ang_vel_b[:, 2]
    # 轉得越快，懲罰越大
    return torch.abs(yaw_vel)

@configclass
class SpotRewardsCfg:
    # ============================================================
    # 1. 核心驅動力 (Exploration Drivers)
    # ============================================================
    
    # [核心] 視覺覆蓋率獎勵 (這是終極目標)
    visual_coverage = RewTerm(
        func=visual_coverage_reward, 
        weight=8.0, # 權重加大！讓看見新世界的誘惑力超過一切
        params={"sensor_cfg": SceneEntityCfg("camera_frustum")}
    )

    # [關鍵修正] 探索的「引擎」！
    # 你的 command 會隨機產生速度指令。我們給予追蹤指令的獎勵，
    # 讓牠有「隨機亂逛」的基礎動力，這樣牠才會不小心看到新地圖並獲得 visual_coverage 的巨大獎勵！
    # track_lin_vel_xy_exp = RewTerm( 
    #     func=mdp.track_lin_vel_xy_exp, 
    #     weight=1.5, # 權重比 coverage 小，讓牠知道「探索大於盲目亂衝」
    #     params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )

    keep_moving = RewTerm(
        func=forward_moving_reward, 
        weight=3.0, # 權重不能大於 visual_coverage
    )

    # ▼▼▼ [關鍵新增] 打轉懲罰 ▼▼▼
    stop_spinning = RewTerm(
        func=yaw_rate_penalty,
        # 給予 -0.5 的懲罰。
        # 這樣牠平常會乖乖走直線；但遇到牆壁時，為了躲避 -5.0 的防撞懲罰，牠還是會心甘情願地轉彎！
        weight=-1.0, 
    )

    # ⚠️ [已經刪除] 絕對不能給 action_magnitude 正分！那會導致原地抽搐

    # ============================================================
    # 2. 穩定性與姿態 (Stability & Posture)
    # ============================================================
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0, 
        params={"target_height": 0.55, "asset_cfg": SceneEntityCfg("robot")}
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-1.0)

    # ============================================================
    # 3. 步態與越野能力 (Locomotion Quality)
    # ============================================================
    foot_clearance = RewTerm(
        func=spot_mdp.foot_clearance_reward, 
        weight=3.0, 
        params={
            "std": 0.05,
            "target_height": 0.18, 
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "tanh_mult": 2.0,
        },
    )
    gait = RewardTermCfg(
        func=spot_mdp.GaitReward,
        weight=2.0, 
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.25,
    #     },
    # )

    # ============================================================
    # 4. 懲罰機制 (Penalties)
    # ============================================================
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0, 
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]), "threshold": 1.0},
    )
    # ▼▼▼ 關鍵修正 2：新增防撞雷達力場 ▼▼▼
    wall_repulsion = RewTerm(
        func=close_to_wall_penalty,
        weight=-5.0, # 權重調高，一靠近牆壁就感到劇痛
        params={"sensor_cfg": SceneEntityCfg("camera_frustum"), "safe_distance": 1.0} # 0.6米內開始警告
    )
    stand_still = RewTerm(
        func=stand_still_penalty_no_cmd, 
        weight=-3.0,
        params={"threshold": 0.2}
    )
    no_fly_penalty = RewTerm(
        func=no_fly, weight=-1.0, 
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.1)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)

@configclass
class SpotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    # Basic settings
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()
    commands: SpotCommandsCfg = SpotCommandsCfg()

    # MDP setting
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()

    viewer = ViewerCfg(eye=(10.5, 10.5, 0.3), origin_type="world", env_index=0, asset_name="robot")
    
    # 訓練時不需要 ROS2，設為 None 避免報錯
    ros2: Ros2ManagerCfg = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 0.0
        self.decimation = 10
        self.episode_length_s = 20.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation

        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.5)
        # [新增] 採用指定的鵝卵石地形
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=True,
        )

        # 迷宮配置
        self.scene.maze = AssetBaseCfg(
            prim_path="/World/Maze", 
            spawn=sim_utils.UsdFileCfg(usd_path="/home/bernie/Isaac_lab/project/spot_vslam/source/spot_vslam/spot_vslam/my_project/spot_new/Collected_spot/flat_maze.usd"),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )

        # 軌道 1: 給 RL 避障用的低解析度 RayCaster (前方)
        self.scene.camera_frustum = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body",
            offset=RayCasterCfg.OffsetCfg(pos=(0.5, 0.0, 0.0)),
            ray_alignment="yaw", 
            pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[4.0, 3.0], direction=(1.0, 0.0, 0.0)),
            max_distance=5.0,
            debug_vis=True,
            mesh_prim_paths=["/World/Maze/walls"],
        )
        
        # [移除] 已拔掉底部的 height_scanner RayCaster

# ==========================================================
# 播放/測試用配置 (Play) - 裝備真實相機與 ROS2
# ==========================================================
@configclass
class SpotRoughEnvCfg_Play(SpotRoughEnvCfg):
    
    # 只有 Play 模式才有 ROS2 配置
    ros2: Ros2ManagerCfg = Ros2ManagerCfg(camera_name="tilted_camera", topic_prefix="/spot")

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 0.0
        self.episode_length_s = 10000.0
        self.observations.policy.enable_corruption = False
        
        # [新增] 只有在 Play 模式，狗狗頭上才裝上吃資源的真實相機給 SLAM
        self.scene.tilted_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body/MyCamera",
            update_period=0.033, # 30Hz
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"], 
            spawn=None,
            offset=CameraCfg.OffsetCfg(pos=(0.4, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        )