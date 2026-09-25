import math
from spot_vslam.assets import SPOT_VSLAM_USD_DIR
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import MISSING
from typing import List, Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ViewerCfg, ManagerBasedRLEnvCfg
from spot_vslam.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, CameraCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.sensors import ContactSensor

import isaaclab.terrains as terrain_gen
import isaaclab_tasks.core.velocity.mdp as mdp
import isaaclab_tasks.contrib.velocity.config.spot.mdp as spot_mdp
from spot_vslam.tasks.spot_vslam.velocity_spot_env_cfg import LocomotionVelocityRoughEnvCfg

# from spot_vslam.assets.spot_with_camera import SPOT_CFG
from isaaclab_assets.robots.spot import SPOT_CFG 
from spot_vslam.managers.ros2_manager import Ros2Manager, Ros2ManagerCfg

# ==========================================================
# 訓練用地形：先以平地為主，讓 policy 先學會穩定走
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
        # 先讓大多數環境都是平地
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.7),

        # 之後真的穩了再打開，例如 proportion=0.1
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(0.01, 0.03),
            noise_step=0.01,
            border_width=0.25,
        ),
    },
)

@configclass
class SpotCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.1, 0.6),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.2, 0.2),
            heading=(0.0, 0.0),
        ),
    )

# ==========================================================
# 視覺 / SLAM 輔助函式
# ==========================================================

def get_orb_slam_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """訓練時用 GT + 小噪聲假裝 SLAM；Play 時讀真實 SLAM。"""
    if env.num_envs > 4:
        pos = env.scene["robot"].data.root_pos_w.clone()
        quat = env.scene["robot"].data.root_quat_w.clone()
        pos += torch.randn_like(pos) * 0.01
        return torch.cat([pos, quat], dim=-1)
    else:
        if hasattr(env, "orb_slam_res"):
            return env.orb_slam_res["pose"]
        return torch.zeros((env.num_envs, 7), device=env.device)

def get_orb_slam_status(env: ManagerBasedRLEnv) -> torch.Tensor:
    if env.num_envs > 4:
        return torch.ones((env.num_envs, 1), device=env.device)
    else:
        if hasattr(env, "orb_slam_res"):
            return env.orb_slam_res["status"]
        return torch.zeros((env.num_envs, 1), device=env.device)

def _get_clean_depth(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """回傳乾淨的深度圖，shape = (N, H, W)"""
    sensor = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output["distance_to_image_plane"].clone()
    depth = torch.nan_to_num(depth, nan=5.0, posinf=5.0, neginf=5.0)
    depth = torch.clamp(depth, min=0.0, max=5.0)
    depth = depth.squeeze(-1)  # (N, H, W)
    return depth

def camera_depth_obs(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """把 60x80 深度圖壓到 15x20，再攤平成 300 維。"""
    depth = _get_clean_depth(env, sensor_cfg)  # (N, H, W)
    depth = depth.unsqueeze(1)                 # (N, 1, H, W)
    pooled_depth = F.avg_pool2d(depth, kernel_size=4, stride=4)  # (N, 1, 15, 20)
    normalized_depth = (pooled_depth.reshape(env.num_envs, -1) / 5.0) * 2.0 - 1.0
    return normalized_depth

def camera_depth_stats(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """額外提供幾個容易學的 summary feature。"""
    depth = _get_clean_depth(env, sensor_cfg)  # (N, H, W)

    left = depth[:, 20:50, 5:25].mean(dim=(1, 2))
    center = depth[:, 20:50, 25:55].mean(dim=(1, 2))
    right = depth[:, 20:50, 55:75].mean(dim=(1, 2))

    front_patch = depth[:, 20:50, 25:55].reshape(env.num_envs, -1)
    front_min = front_patch.min(dim=1).values
    lr_balance = left - right

    stats = torch.stack([left, center, right, front_min, lr_balance], dim=1)

    # 前四個是深度，做 0~5m normalization；最後一個左右差值也壓一下
    stats[:, :4] = (stats[:, :4] / 5.0) * 2.0 - 1.0
    stats[:, 4] = torch.clamp(stats[:, 4] / 5.0, min=-1.0, max=1.0)
    return stats

def slow_down_near_obstacle_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    safe_distance: float,
    speed_scale: float,
    max_body_tilt: float,
) -> torch.Tensor:
    depth = _get_clean_depth(env, sensor_cfg)  # (N, H, W)

    near_region = depth[:, 30:47, 22:42].reshape(env.num_envs, -1)

    k = max(1, int(near_region.shape[1] * 0.1))
    low_k, _ = torch.topk(near_region, k=k, dim=1, largest=False)
    near_dist = low_k.mean(dim=1)

    obstacle_violation = torch.clamp(safe_distance - near_dist, min=0.0)

    vx = torch.clamp(env.scene["robot"].data.root_lin_vel_b[:, 0], min=0.0)

    # 用 projected gravity 判斷身體是否傾斜太大
    gravity = env.scene["robot"].data.projected_gravity_b
    body_tilt = torch.norm(gravity[:, :2], dim=1)

    stable_mask = (body_tilt < max_body_tilt).float()

    return obstacle_violation * (vx / speed_scale) * stable_mask

# ==========================================================
# Reward 函式
# ==========================================================

def target_forward_speed_reward(env: ManagerBasedRLEnv, target_speed: float, sigma: float, target_height: float) -> torch.Tensor:
    """
    鼓勵『接近目標前進速度』，而不是越快越好。
    這會比直接獎勵 vx 更穩，不容易暴衝。
    """
    vx = env.scene["robot"].data.root_lin_vel_b[:, 0]
    yaw_rate = torch.abs(env.scene["robot"].data.root_ang_vel_b[:, 2])
    base_height = env.scene["robot"].data.root_pos_w[:, 2]

    speed_reward = torch.exp(-torch.square(vx - target_speed) / sigma)
    yaw_factor = torch.exp(-2.0 * yaw_rate)
    height_factor = torch.exp(-40.0 * torch.square(base_height - target_height))

    return speed_reward * yaw_factor * height_factor

def visual_wall_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, safe_distance: float) -> torch.Tensor:
    depth = _get_clean_depth(env, sensor_cfg)  # (N, H, W)

    left_front = depth[:, 20:50, 8:28].reshape(env.num_envs, -1)
    center_front = depth[:, 20:50, 22:42].reshape(env.num_envs, -1)
    right_front = depth[:, 20:50, 36:56].reshape(env.num_envs, -1)

    def robust_near_dist(region: torch.Tensor) -> torch.Tensor:
        k = max(1, int(region.shape[1] * 0.1))
        low_k, _ = torch.topk(region, k=k, dim=1, largest=False)
        return low_k.mean(dim=1)

    left_dist = robust_near_dist(left_front)
    center_dist = robust_near_dist(center_front)
    right_dist = robust_near_dist(right_front)

    near_dist = torch.minimum(torch.minimum(left_dist, center_dist), right_dist)

    violation = torch.clamp(safe_distance - near_dist, min=0.0)
    return violation ** 2

def open_space_seeker_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    小幅鼓勵往前方開闊區走。
    權重不宜太大，不然會重新鼓勵暴衝。
    """
    depth = _get_clean_depth(env, sensor_cfg)  # (N, H, W)

    front = depth[:, 20:50, 25:55]
    front_mean = front.mean(dim=(1, 2)) / 5.0

    fwd_vel = torch.clamp(env.scene["robot"].data.root_lin_vel_b[:, 0], min=0.0)
    return fwd_vel * front_mean

def no_fly(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """四隻腳都沒有接觸力時懲罰。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    forces_norm = torch.norm(current_forces, dim=-1)
    is_air = forces_norm < 1.0
    all_legs_in_air = torch.all(is_air, dim=1)
    return all_legs_in_air.float()

def yaw_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    yaw_vel = env.scene["robot"].data.root_ang_vel_b[:, 2]
    return torch.abs(yaw_vel)

def stand_still_penalty_no_cmd(env: ManagerBasedRLEnv, threshold: float, max_curriculum_steps: int) -> torch.Tensor:
    """
    沒有明確 command 時的怠惰懲罰。
    讓前期先別罰太重，後期再慢慢增加。
    """
    fwd_vel = env.scene["robot"].data.root_lin_vel_b[:, 0]
    shortfall = torch.clamp(threshold - fwd_vel, min=0.0)
    penalty_ratio = shortfall / threshold
    progress = min(env.common_step_counter / max_curriculum_steps, 1.0)
    return penalty_ratio * progress

def forward_progress_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    很小的前進鼓勵，避免完全不動。
    但上限壓低，不讓它把這條當成爆衝來源。
    """
    vx = env.scene["robot"].data.root_lin_vel_b[:, 0]
    yaw_rate = torch.abs(env.scene["robot"].data.root_ang_vel_b[:, 2])

    forward = torch.clamp(vx, min=0.0, max=0.35)
    straight_factor = torch.exp(-2.5 * yaw_rate)
    return forward * straight_factor

def body_height_reward(env: ManagerBasedRLEnv, target_height: float, sigma: float = 0.03) -> torch.Tensor:
    """鼓勵 body 維持在合理高度，避免一直蹲坐。"""
    base_height = env.scene["robot"].data.root_pos_w[:, 2]
    return torch.exp(-torch.square(base_height - target_height) / sigma)

def body_contact_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """只要 body 碰地就懲罰。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    forces_norm = torch.norm(current_forces, dim=-1)
    is_contact = forces_norm > threshold
    return torch.any(is_contact, dim=1).float()

def forward_speed_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    vx = env.scene["robot"].data.root_lin_vel_b[:, 0]
    return torch.clamp(vx, min=0.0)

def exploration_reward(env):
    pos = env.scene["robot"].data.root_pos_w[:, :2]
    return torch.norm(pos, dim=1)

def stand_body_height_reward(env: ManagerBasedRLEnv, target_height: float, speed_threshold: float) -> torch.Tensor:
    base_height = env.scene["robot"].data.root_pos_w[:, 2]
    speed = torch.norm(env.scene["robot"].data.root_lin_vel_b[:, :2], dim=1)

    height_reward = torch.exp(-torch.square(base_height - target_height) / 0.02)
    stand_mask = (speed < speed_threshold).float()
    return height_reward * stand_mask

# ==========================================================
# Observation 配置
# ==========================================================

@configclass
class SpotObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Proprioception
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # Vision
        camera_depth = ObsTerm(func=camera_depth_obs, params={"sensor_cfg": SceneEntityCfg("train_camera")})
        camera_depth_stats = ObsTerm(func=camera_depth_stats, params={"sensor_cfg": SceneEntityCfg("train_camera")})

        # SLAM
        # orb_pose = ObsTerm(func=get_orb_slam_pose)
        # orb_status = ObsTerm(func=get_orb_slam_status)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

# ==========================================================
# Reward 配置
# ==========================================================

@configclass
class SpotRewardsCfg:
    stand_height = RewTerm(
        func=stand_body_height_reward,
        weight=1.0,
        params={"target_height": 0.52, "speed_threshold": 0.15},
    )

    keep_body_height = RewTerm(
        func=body_height_reward,
        weight=2.0,
        params={"target_height": 0.47, "sigma": 0.05},
    )

    # 主目標：追蹤高層給的速度命令
    base_linear_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward,
        weight=10.0,
        params={
            "std": 0.2,
            "ramp_rate": 0.5,
            "ramp_at_vel": 0.8,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 壓制亂轉
    # stop_spinning = RewTerm(
    #     func=yaw_rate_penalty,
    #     weight=-0.8,
    # )
    base_angular_velocity = RewardTermCfg(
        func=spot_mdp.base_angular_velocity_reward,
        weight=6.0,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("robot")},
    )
    gait = RewardTermCfg(
        func=spot_mdp.GaitReward,
        weight=3.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.15,
            "synced_feet_pair_names": (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    foot_clearance = RewardTermCfg(
        func=spot_mdp.foot_clearance_reward,
        weight=0.8,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.10,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

    body_contact = RewTerm(
        func=body_contact_penalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body"]),
            "threshold": 1.0,
        },
    )

    base_orientation = RewardTermCfg(
        func=spot_mdp.base_orientation_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )

    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-1.0,
    )

    action_smoothness = RewardTermCfg(
        func=spot_mdp.action_smoothness_penalty,
        weight=-0.05,
    )

    joint_pos = RewardTermCfg(
        func=spot_mdp.joint_position_penalty,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )

    # 視覺避障先保留很弱，避免低層完全撞牆
    # visual_wall_penalty = RewTerm(
    #     func=visual_wall_penalty,
    #     weight=-0.15,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("train_camera"),
    #         "safe_distance": 0.55,
    #     },
    # )
# ==========================================================
# 核心環境配置
# ==========================================================

from .simulate_vslam_env_cfg import SpotActionsCfg, SpotEventCfg

@configclass
class SpotTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body"]),
            "threshold": 1.0,
        },
    )

    leg_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*leg"]),
            "threshold": 1.0,
        },
    )

    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )

@configclass
class SpotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()
    commands: SpotCommandsCfg = SpotCommandsCfg()
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()

    viewer = ViewerCfg(eye=(10.5, 10.5, 0.3), origin_type="world", env_index=0, asset_name="robot")
    ros2: Ros2ManagerCfg = None

    def __post_init__(self):
        super().__post_init__()

        self.scene.env_spacing = 0.0
        self.decimation = 20
        self.episode_length_s = 20.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation

        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.53)

        # --------------------------------------------------
        # 地形：先用平地訓練
        # --------------------------------------------------
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level=0,
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
            debug_vis=False,
        )

        # --------------------------------------------------
        # Maze：先關掉，等穩了再打開
        # --------------------------------------------------
        # self.scene.maze = AssetBaseCfg(
        #     prim_path="/World/Maze",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path=f"{SPOT_VSLAM_USD_DIR}/flat_maze.usd"
        #     ),
        #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        # )
        # self.scene.fourwalls = AssetBaseCfg(
        #     prim_path="/World/walls",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path=f"{SPOT_VSLAM_USD_DIR}/four_big_walls.usd"
        #     ),
        #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        # )

        # --------------------------------------------------
        # 訓練用深度相機
        # --------------------------------------------------
        self.scene.train_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body/train_camera",
            update_period=0.04,
            height=48,
            width=64,
            data_types=["distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0,
                focus_distance=400.0,
                horizontal_aperture=22.0,
                clipping_range=(0.1, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.4, 0.0, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )
        self.scene.tilted_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body/MyCamera",
            update_period=0.1,
            height=240,
            width=320,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0,
                focus_distance=400.0,
                horizontal_aperture=22.0,
                clipping_range=(0.1, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.4, 0.0, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )

# ==========================================================
# Play / 測試用配置
# ==========================================================

@configclass
class SpotRoughEnvCfg_Play(SpotRoughEnvCfg):
    ros2: Ros2ManagerCfg = Ros2ManagerCfg(camera_name="tilted_camera", topic_prefix="/spot")

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 0.0
        self.episode_length_s = 10000.0
        self.observations.policy.enable_corruption = False

        self.scene.tilted_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body/MyCamera",
            update_period=0.1,
            height=240,
            width=320,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0,
                focus_distance=400.0,
                horizontal_aperture=22.0,
                clipping_range=(0.1, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.4, 0.0, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )
        self.scene.maze = AssetBaseCfg(
            prim_path="/World/Maze",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{SPOT_VSLAM_USD_DIR}/flat_maze.usd"
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )
        # self.scene.fourwalls = AssetBaseCfg(
        #     prim_path="/World/walls",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path=f"{SPOT_VSLAM_USD_DIR}/four_big_walls.usd"
        #     ),
        #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        # )

# ==========================================================
# Base high-level train cfg
# ==========================================================
@configclass
class SpotHighLevelTrainEnvCfg(SpotRoughEnvCfg):
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()
    commands: SpotCommandsCfg = SpotCommandsCfg()
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()

    viewer = ViewerCfg(eye=(10.5, 10.5, 0.3), origin_type="world", env_index=0, asset_name="robot")
    ros2: Ros2ManagerCfg = Ros2ManagerCfg(camera_name="tilted_camera", topic_prefix="/spot")

    # 明確控制 SLAM 模式
    use_fake_slam_for_training: bool = False

    def __post_init__(self):
        super().__post_init__()

        # 保持跟 low-level locomotion checkpoint 一致
        self.scene.env_spacing = 0.0
        self.decimation = 20
        self.episode_length_s = 20.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation

        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.53)

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level=0,
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
            debug_vis=False,
        )

        # high-level 訓練要的迷宮/障礙場景
        self.scene.maze = AssetBaseCfg(
            prim_path="/World/Maze",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{SPOT_VSLAM_USD_DIR}/flat_maze.usd"
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )

        # 訓練用 depth camera
        self.scene.train_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body/train_camera",
            update_period=0.04,  # 25 Hz
            height=48,
            width=64,
            data_types=["distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0,
                focus_distance=400.0,
                horizontal_aperture=22.0,
                clipping_range=(0.1, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.4, 0.0, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )
        self.scene.tilted_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body/MyCamera",
            update_period=0.1,
            height=240,
            width=320,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0,
                focus_distance=400.0,
                horizontal_aperture=22.0,
                clipping_range=(0.1, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.4, 0.0, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )


# ==========================================================
# High-level play/eval cfg
# ==========================================================
@configclass
class SpotHighLevelTrainEnvCfg_Play(SpotHighLevelTrainEnvCfg):
    ros2: Ros2ManagerCfg = Ros2ManagerCfg(camera_name="tilted_camera", topic_prefix="/spot")
    use_fake_slam_for_training: bool = False

    # 新增：dense map 開關
    enable_dense_mapping: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()

        # Play / eval 少量 env
        self.scene.num_envs = 1
        self.scene.env_spacing = 0.0
        self.episode_length_s = 10000.0
        self.observations.policy.enable_corruption = False

        # 不要改成 40，保持跟 low-level checkpoint 接近
        self.decimation = 20
        self.sim.render_interval = self.decimation

        # 給 ORB-SLAM / ROS2 用的較高解析度相機
        self.scene.tilted_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body/MyCamera",
            update_period=0.1,  # 10 Hz
            height=240,
            width=320,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0,
                focus_distance=400.0,
                horizontal_aperture=22.0,
                clipping_range=(0.1, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.4, 0.0, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )
