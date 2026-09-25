from isaaclab.utils.configclass import configclass
from spot_vslam.assets import SPOT_VSLAM_USD_DIR

from spot_vslam.tasks.spot_vslam.velocity_spot_env_cfg import LocomotionVelocityRoughEnvCfg
# from .cfg.velocity_spot_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from spot_vslam.assets.spot_with_camera import SPOT_CFG
# from  isaaclab_assets.robots.spot import SPOT_CFG

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ViewerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
import torch
from isaaclab.sensors import ContactSensor
from isaaclab.sensors import RayCaster

import isaaclab_tasks.core.velocity.mdp as mdp
import isaaclab_tasks.contrib.velocity.config.spot.mdp as spot_mdp
from isaaclab.sensors import CameraCfg
from isaaclab.assets import AssetBaseCfg
# from spot_vslam.managers.ros2_manager import Ros2Manager, Ros2ManagerCfg
# from spot_vslam.managers.orb_slam_subscriber_manager import OrbSlamSubscriberCfg
# import spot_vslam.mdp.custom_mdp as custom_mdp

from isaaclab.sim import UsdFileCfg
from typing import List
import numpy as np


##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
import isaaclab.terrains as terrain_gen

##
# Scene definition
##


class VisualCoverageManager:
    """
    管理所有環境的柵格地圖 (Grid Map) 並計算覆蓋率獎勵。
    """
    def __init__(self, map_size=40.0, resolution=0.2, device="cuda"):
        self.map_size = map_size # 地圖邊長 (米)
        self.resolution = resolution # 格子大小 (米)
        self.grid_dim = int(map_size / resolution) # 格子數量 (例如 100x100)
        self.device = device
        
        # [num_envs, grid_dim, grid_dim]
        # 0 = 未知, 1 = 已知
        self.maps = None 
        self.initialized = False

    def init_maps(self, num_envs):
        if not self.initialized or self.maps.shape[0] != num_envs:
            self.maps = torch.zeros((num_envs, self.grid_dim, self.grid_dim), 
                                  dtype=torch.bool, device=self.device)
            self.initialized = True
            print(f"[CoverageManager] Initialized {num_envs} maps of size {self.grid_dim}x{self.grid_dim}")

    def reset_idx(self, env_ids):
        """重置指定環境的地圖 (當機器人死掉或重來時)"""
        if self.initialized and len(env_ids) > 0:
            self.maps[env_ids] = False

    def compute_reward(self, env, sensor_cfg: SceneEntityCfg):
        """
        核心邏輯：計算有多少射線打到了「未知區域」
        """
        # 1. 確保地圖已初始化
        if not self.initialized:
            self.init_maps(env.num_envs)
            
        # 2. 處理 Reset (檢查哪些環境剛被重置)
        # reset_buf 是 Isaac Lab 用來標記哪些環境剛重置的 buffer
        reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # 3. 取得 RayCaster 擊中點
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        # ray_hits_w: [num_envs, num_rays, 3]
        hits = sensor.data.ray_hits_w.clone()
        
        # 4. 座標轉換 (World -> Grid)
        # 假設機器人出生點附近的區域是地圖範圍
        # 為了簡單，我們假設地圖中心跟隨機器人的出生點 (env_origins)
        # 或者是固定的 World Origin，這裡使用 World Origin 比較適合建圖任務
        
        # 將座標平移，使 (0,0) 位於地圖中心
        # grid_x = (x + size/2) / res
        half_size = self.map_size / 2.0
        
        # 我們只關心 X, Y 平面
        grid_indices = ((hits[..., :2] + half_size) / self.resolution).long()
        
        # 5. 過濾出界點
        x_idx = grid_indices[..., 0]
        y_idx = grid_indices[..., 1]
        valid_mask = (x_idx >= 0) & (x_idx < self.grid_dim) & \
                     (y_idx >= 0) & (y_idx < self.grid_dim)
        
        # 6. 計算獎勵 (向量化操作)
        # 我們要把 hits 展平成 [total_rays, 2] 來處理
        num_envs, num_rays = x_idx.shape
        
        # 為了利用 PyTorch 的高效率，我們使用 flatten 來索引
        batch_indices = torch.arange(num_envs, device=self.device).unsqueeze(1).expand(-1, num_rays)
        
        # 只取有效的點
        valid_batch = batch_indices[valid_mask]
        valid_x = x_idx[valid_mask]
        valid_y = y_idx[valid_mask]
        
        if len(valid_batch) == 0:
            return torch.zeros(num_envs, device=self.device)

        # 檢查這些點是否已經被訪問過
        # is_visited: [num_valid_hits]
        is_visited = self.maps[valid_batch, valid_x, valid_y]
        
        # 只有「未訪問 (False)」的點才給分
        new_visits = ~is_visited
        
        # 更新地圖
        # 注意：這裡可能會有同一個 step 多條射線打到同一個新格子的情況
        # 但為了效能，我們暫時允許重複計算 (或者你可以用 unique)
        self.maps[valid_batch[new_visits], valid_x[new_visits], valid_y[new_visits]] = True
        
        # 7. 聚合獎勵回每個環境
        # 我們需要計算每個 env 有多少個 new_visits
        rewards = torch.zeros(num_envs, device=self.device)
        
        # 使用 scatter_add 或 index_add 計算每個環境的新發現數
        # 此處 new_visits 是一個 mask，我們先找出這些新發現屬於哪個 env
        new_visit_envs = valid_batch[new_visits]
        
        # 每個新格子給 1.0 分 (之後在 config 裡調整 weight)
        ones = torch.ones_like(new_visit_envs, dtype=torch.float)
        rewards.scatter_add_(0, new_visit_envs, ones)
        
        # 正規化獎勵：除以射線總數，避免獎勵過大
        # 或是直接輸出數量，由 weight 控制
        return rewards

# 實例化一個全域管理器
COVERAGE_MANAGER = VisualCoverageManager(map_size=20.0, resolution=0.2)

# 定義給 Config 呼叫的 Wrapper 函數
def visual_coverage_reward(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return COVERAGE_MANAGER.compute_reward(env, sensor_cfg)

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


@configclass
class SpotActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True)


@configclass
class SpotCommandsCfg:
    """廢除外部指令：讓機器人完全自主決定速度和方向"""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,  # <--- 100% 處於無指令狀態
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), # 全部歸零
            lin_vel_y=(0.0, 0.0), 
            ang_vel_z=(0.0, 0.0), 
            heading=(0.0, 0.0)
        ),
    )


def forward_depth_scan(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """讓機器人真正「看見」前方的牆壁距離，並過濾致命的 Inf/NaN 毒藥"""
    sensor = env.scene.sensors[sensor_cfg.name]
    
    # 取得射線擊中點的世界座標 [num_envs, num_rays, 3]
    hits_w = sensor.data.ray_hits_w.clone()
    
    # 取得機器人身體的位置 [num_envs, 1, 3]
    robot_pos = env.scene["robot"].data.root_pos_w.unsqueeze(1)
    # ====================================================================
    # print("env count:", hits_w.shape[0])
    # print("hit count per env:", torch.isfinite(hits_w[...,0]).sum(dim=1))
    # ====================================================================

    # 計算每條射線打到障礙物的距離
    distances = torch.norm(hits_w - robot_pos, dim=-1)
    
    # ▼▼▼ [防爆防禦機制] ▼▼▼
    # 當射線沒有打到任何東西 (Miss) 時，Isaac Lab 會給予 NaN 或 Inf
    # 我們把這些「看入虛空」的射線，全部強制設定為最大視距 5.0 米
    distances = torch.nan_to_num(distances, nan=5.0, posinf=5.0, neginf=5.0)
    
    # 安全裁切，確保距離絕對在 0.0 到 5.0 之間
    distances = torch.clamp(distances, min=0.0, max=5.0)
    # ▲▲▲ ================= ▲▲▲
    
    # 正規化：將距離映射到 (-1.0 ~ 1.0) 幫助神經網路完美收斂
    normalized_depth = (distances / 5.0) * 2.0 - 1.0
    
    return normalized_depth

@configclass
class SpotObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        forward_depth = ObsTerm(
            func=forward_depth_scan,
            params={"sensor_cfg": SceneEntityCfg("camera_frustum")},
        )
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class SpotEventCfg:
    """Configuration for randomization."""

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
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

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

# 請把這段加在檔案最上面，或是 mdp 導入區
def stand_still_penalty(env, command_name: str, threshold: float) -> torch.Tensor:
    """
    懲罰懶惰：當命令速度大於 0.1，但實際速度小於 0.1 時，給予懲罰。
    """
    # 1. 取得指令 (Command)
    commands = env.command_manager.get_command(command_name)
    cmd_lin_vel_xy = commands[:, :2] # 取前兩個維度 (vx, vy)
    
    # 2. 取得實際速度 (Root Velocity)
    root_vel_w = env.scene["robot"].data.root_lin_vel_w
    root_vel_xy = root_vel_w[:, :2]

    # 3. 計算大小
    cmd_norm = torch.norm(cmd_lin_vel_xy, dim=1)
    vel_norm = torch.norm(root_vel_xy, dim=1)

    # 4. 判斷是否偷懶 (命令很大，速度很小)
    is_lazy = (cmd_norm > threshold) & (vel_norm < threshold)
    
    return is_lazy.float()

# ==========================================================
# 自訂 Reward 函數區
# ==========================================================

def forward_moving_reward(env) -> torch.Tensor:
    """引擎進化：獎勵往前走，『也獎勵原地轉向探路』"""
    base_vel = env.scene["robot"].data.root_lin_vel_b
    fwd_vel = torch.clamp(base_vel[:, 0], min=0.0)
    
    # 取得原地轉向 (Yaw) 的速度
    yaw_vel = torch.abs(env.scene["robot"].data.root_ang_vel_b[:, 2])
    
    # 往前走給 1 倍分，原地轉向找路給 0.5 倍分
    # return fwd_vel + 0.01 * yaw_vel
    return fwd_vel

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
    """防撞力場：距離牆壁太近時給予指數級別的懲罰，逼迫提早轉彎"""
    sensor = env.scene.sensors[sensor_cfg.name]
    
    # 取得射線擊中點的世界座標
    hits_w = sensor.data.ray_hits_w.clone()
    # 取得機器人位置
    robot_pos = env.scene["robot"].data.root_pos_w.unsqueeze(1)
    
    # 計算每條射線打到障礙物的距離
    distances = torch.norm(hits_w - robot_pos, dim=-1)
    
    # 過濾未擊中的射線 (Miss)，將它們視為絕對安全的 5.0 米
    distances = torch.nan_to_num(distances, nan=5.0, posinf=5.0, neginf=5.0)
    
    # 找出每隻狗正前方「最近」的障礙物距離
    min_dist, _ = torch.min(distances, dim=1)
    
    # 如果最近距離小於安全距離 (例如 0.5)，就產生懲罰倍率
    # 離得越近，懲罰值越大 (例如 0.5 - 0.1 = 0.4)
    violation = torch.clamp(safe_distance - min_dist, min=0.0)
    
    return violation


@configclass
class SpotRewardsCfg:
    # ============================================================
    # 1. 核心驅動力 (Exploration Drivers)
    # ============================================================
    
    # [核心] 視覺覆蓋率獎勵 (這是終極目標)
    visual_coverage = RewTerm(
        func=visual_coverage_reward, 
        weight=5.0, # 權重加大！讓看見新世界的誘惑力超過一切
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
        weight=1.0, # 權重不能大於 visual_coverage
    )

    # ⚠️ [已經刪除] 絕對不能給 action_magnitude 正分！那會導致原地抽搐

    # ============================================================
    # 2. 穩定性與姿態 (Stability & Posture)
    # ============================================================
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0, 
        params={"target_height": 0.55, "asset_cfg": SceneEntityCfg("robot")}
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)

    # ============================================================
    # 3. 步態與越野能力 (Locomotion Quality)
    # ============================================================
    foot_clearance = RewTerm(
        func=spot_mdp.foot_clearance_reward, 
        weight=1.0, 
        params={
            "std": 0.05,
            "target_height": 0.12, 
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "tanh_mult": 2.0,
        },
    )
    gait = RewardTermCfg(
        func=spot_mdp.GaitReward,
        weight=1.0, 
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
        weight=-0.1, # 權重調高，一靠近牆壁就感到劇痛
        params={"sensor_cfg": SceneEntityCfg("camera_frustum"), "safe_distance": 0.6} # 0.6米內開始警告
    )
    stand_still = RewTerm(
        func=stand_still_penalty_no_cmd, 
        weight=-2.0,
        params={"threshold": 0.2}
    )
    no_fly_penalty = RewTerm(
        func=no_fly, weight=-1.0, 
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.1)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1)

@configclass
class SpotTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]), "threshold": 1.0}, 
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class SpotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the Spot robot in a flat environment."""

    # Basic settings
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()
    commands: SpotCommandsCfg = SpotCommandsCfg()

    # MDP setting
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()

    # Viewer
    viewer = ViewerCfg(eye=(10.5, 10.5, 0.3), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.env_spacing = 0.0
        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002  # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt

        # switch robot to Spot-d
        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # terrain
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
        self.scene.maze = AssetBaseCfg(
            prim_path="/World/Maze", 
            spawn=sim_utils.UsdFileCfg(
                # [⚠️重要] 請務必把這裡改成你的 USD 檔案的絕對路徑！
                usd_path=f"{SPOT_VSLAM_USD_DIR}/flat_maze.usd", 
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0), # 如果迷宮沒對齊，可以在這裡調整 XYZ 偏移量
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )


        # no height scan
        # self.scene.height_scanner = None
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        # [修正] 模擬相機 (使用朝前的 GridPattern 繞過 Pinhole Bug)
        self.scene.camera_frustum = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body",
            offset=RayCasterCfg.OffsetCfg(pos=(0.5, 0.0, 0.0)), # 裝在狗頭上
            ray_alignment="yaw", 
            # ▼▼▼ 關鍵修正：改用 GridPattern，並設定 direction 朝前 ▼▼▼
            pattern_cfg=patterns.GridPatternCfg(
                resolution=0.2,            # 每 20cm 發射一條射線 (兼顧效能與密度)
                size=[4.0, 3.0],           # 掃描視野寬度 4m, 高度 3m
                direction=(1.0, 0.0, 0.0)  # [核心] 1.0 在 X 軸，代表朝機器人正前方發射
            ),
            max_distance=5.0, # 視線最遠看 5 米
            debug_vis=False,   # 建議開啟，你會看到機器人前面推著一堵紅色的射線牆
            mesh_prim_paths=["/World/Maze/walls"],
        )


class SpotRoughEnvCfg_Play(SpotRoughEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 64
        self.scene.env_spacing = 0.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event