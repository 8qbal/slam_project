import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
import numpy as np
import torch


# =========================================================
# 小工具
# =========================================================

def quat_to_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """
    quat format: (w, x, y, z)
    return shape: (N,)
    """
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def get_camera_depth_stats_from_env(env_unwrapped, sensor_name: str = "train_camera") -> torch.Tensor:
    """
    回傳 shape = (N, 5)
    [left, center, right, front_min, lr_balance]
    已 normalize 到大約 [-1, 1]
    """
    sensor = env_unwrapped.scene.sensors[sensor_name]
    depth = sensor.data.output["distance_to_image_plane"].clone()
    depth = torch.nan_to_num(depth, nan=5.0, posinf=5.0, neginf=5.0)
    depth = torch.clamp(depth, min=0.0, max=5.0)
    depth = depth.squeeze(-1)  # (N, H, W)

    left = depth[:, 20:50, 5:25].mean(dim=(1, 2))
    center = depth[:, 20:50, 25:55].mean(dim=(1, 2))
    right = depth[:, 20:50, 55:75].mean(dim=(1, 2))

    front_patch = depth[:, 20:50, 25:55].reshape(env_unwrapped.num_envs, -1)
    front_min = front_patch.min(dim=1).values
    lr_balance = left - right

    stats = torch.stack([left, center, right, front_min, lr_balance], dim=1)
    stats[:, :4] = (stats[:, :4] / 5.0) * 2.0 - 1.0
    stats[:, 4] = torch.clamp(stats[:, 4] / 5.0, min=-1.0, max=1.0)
    return stats


def get_orb_pose_xyyaw_from_env(env_unwrapped) -> torch.Tensor:
    """
    優先讀真 ORB-SLAM3：
      env_unwrapped.orb_slam_res["pose_xyyaw"]  shape (N, 3)

    若沒有，fallback 到 GT + noise
    """
    if (
        hasattr(env_unwrapped, "orb_slam_res")
        and isinstance(env_unwrapped.orb_slam_res, dict)
        and "pose_xyyaw" in env_unwrapped.orb_slam_res
    ):
        pose_xyyaw = env_unwrapped.orb_slam_res["pose_xyyaw"]
        if pose_xyyaw is not None:
            return pose_xyyaw

    pos_xy = env_unwrapped.scene["robot"].data.root_pos_w[:, :2].clone()
    quat = env_unwrapped.scene["robot"].data.root_quat_w.clone()
    yaw = quat_to_yaw(quat).unsqueeze(-1)

    pose = torch.cat([pos_xy, yaw], dim=-1)
    noise = torch.zeros_like(pose)
    noise[:, 0] = torch.randn_like(pose[:, 0]) * 0.01
    noise[:, 1] = torch.randn_like(pose[:, 1]) * 0.01
    noise[:, 2] = torch.randn_like(pose[:, 2]) * 0.01
    return pose + noise


def get_orb_status_from_env(env_unwrapped) -> torch.Tensor:
    """
    優先讀真 ORB-SLAM3：
      env_unwrapped.orb_slam_res["status"] shape (N, 1)

    若沒有，fallback = 全部 tracking ok
    """
    if (
        hasattr(env_unwrapped, "orb_slam_res")
        and isinstance(env_unwrapped.orb_slam_res, dict)
        and "status" in env_unwrapped.orb_slam_res
    ):
        status = env_unwrapped.orb_slam_res["status"]
        if status is not None:
            return status

    return torch.ones((env_unwrapped.num_envs, 1), device=env_unwrapped.device, dtype=torch.float32)


# =========================================================
# 可選：高層 command 平滑
# =========================================================

class CommandFilter:
    def __init__(self, num_envs, device, alpha=0.25):
        self.alpha = alpha
        self.cmd = torch.zeros((num_envs, 3), device=device)

    def reset(self):
        self.cmd.zero_()

    def update(self, new_cmd):
        self.cmd = (1.0 - self.alpha) * self.cmd + self.alpha * new_cmd
        return self.cmd


# =========================================================
# 凍結 low-level policy adapter
# =========================================================

class FrozenLowLevelAgentAdapter:
    def __init__(self, agent):
        self.agent = agent

    @torch.inference_mode()
    def act(self, obs):
        obs_dict = self.agent.obs_to_torch(obs)
        action = self.agent.get_action(obs_dict, is_deterministic=True)

        if isinstance(action, np.ndarray):
            action = torch.tensor(action, device=next(self.agent.model.parameters()).device)

        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        return action

    def reset_rnn_if_needed(self, dones):
        if self.agent.is_rnn and self.agent.states is not None and len(dones) > 0:
            for s in self.agent.states:
                s[:, dones, :] = 0.0


# =========================================================
# 高層 config
# =========================================================

@dataclass
class HighLevelEnvCfg:
    hl_decimation: int = 5
    cmd_smoothing_alpha: float = 0.25

    vx_min: float = 0.0
    vx_max: float = 0.5
    wz_min: float = -0.25
    wz_max: float = 0.25

    success_progress_scale: float = 2.0
    tracking_bonus: float = 0.2
    tracking_lost_penalty: float = 1.5
    stuck_penalty: float = 0.8
    body_contact_penalty: float = 1.5
    excessive_turn_penalty: float = 0.05

    stuck_dist_threshold: float = 0.03
    max_episode_hl_steps: int = 400


# =========================================================
# High-level RL Env
# =========================================================

class HighLevelRLEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, low_level_env, low_level_agent, cfg: Optional[HighLevelEnvCfg] = None):
        super().__init__()
        self.env = low_level_env
        self.env_unwrapped = low_level_env.unwrapped
        self.low_level_agent = FrozenLowLevelAgentAdapter(low_level_agent)
        self.cfg = cfg or HighLevelEnvCfg()
        self._last_low_obs = None

        self.num_envs = self.env_unwrapped.num_envs
        self.device = self.env_unwrapped.device

        self.cmd_filter = CommandFilter(
            num_envs=self.num_envs,
            device=self.device,
            alpha=self.cfg.cmd_smoothing_alpha,
        )

        self.obs_dim = 11
        self.act_dim = 2

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.act_dim,),
            dtype=np.float32,
        )

        self.last_high_action = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.prev_pos_xy = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.hl_step_count = 0

        # ===== 新增：episode 統計 =====
        self.ep_reward_sum = 0.0
        self.ep_length = 0
        self.ep_start_time = time.time()

    # -----------------------------------------------------
    # command mapping
    # -----------------------------------------------------
    def _action_to_command(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.clamp(action, -1.0, 1.0)

        vx = (action[:, 0] + 1.0) * 0.5 * (self.cfg.vx_max - self.cfg.vx_min) + self.cfg.vx_min
        wz = (action[:, 1] + 1.0) * 0.5 * (self.cfg.wz_max - self.cfg.wz_min) + self.cfg.wz_min

        cmd = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        cmd[:, 0] = vx
        cmd[:, 1] = 0.0
        cmd[:, 2] = wz
        return cmd

    def _set_base_velocity_command(self, cmd_tensor: torch.Tensor):
        if hasattr(self.env_unwrapped.command_manager, "set_command"):
            self.env_unwrapped.command_manager.set_command("base_velocity", cmd_tensor)
        else:
            self.env_unwrapped.command_manager._terms["base_velocity"].command[:] = cmd_tensor

    # -----------------------------------------------------
    # observation
    # -----------------------------------------------------
    def _get_obs_tensor(self) -> torch.Tensor:
        orb_pose = get_orb_pose_xyyaw_from_env(self.env_unwrapped)
        orb_status = get_orb_status_from_env(self.env_unwrapped)
        depth_stats = get_camera_depth_stats_from_env(self.env_unwrapped)

        obs = torch.cat(
            [
                orb_pose,
                orb_status,
                depth_stats,
                self.last_high_action,
            ],
            dim=-1,
        )
        return obs

    def _get_obs_numpy(self) -> np.ndarray:
        obs = self._get_obs_tensor().detach().cpu().numpy().astype(np.float32)
        return obs

    # -----------------------------------------------------
    # reward
    # -----------------------------------------------------
    def _body_contact_flag(self) -> torch.Tensor:
        contact_sensor = self.env_unwrapped.scene.sensors["contact_forces"]

        sensor_cfg_body_ids = None
        try:
            from isaaclab.managers import SceneEntityCfg
            sensor_cfg_body_ids = SceneEntityCfg("contact_forces", body_names=["body"]).body_ids
        except Exception:
            pass

        if sensor_cfg_body_ids is None:
            return torch.zeros((self.num_envs,), device=self.device)

        current_forces = contact_sensor.data.net_forces_w[:, sensor_cfg_body_ids, :]
        forces_norm = torch.norm(current_forces, dim=-1)
        body_contact = torch.any(forces_norm > 1.0, dim=1).float()
        return body_contact

    def _compute_reward(self, action_cmd: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        curr_pos_xy = self.env_unwrapped.scene["robot"].data.root_pos_w[:, :2]
        delta = curr_pos_xy - self.prev_pos_xy
        progress = torch.norm(delta, dim=1)

        orb_status = get_orb_status_from_env(self.env_unwrapped).squeeze(-1)
        tracking_ok = orb_status
        tracking_bad = 1.0 - orb_status

        stuck = (progress < self.cfg.stuck_dist_threshold).float()
        body_contact = self._body_contact_flag()
        wz_abs = torch.abs(action_cmd[:, 2])

        reward = (
            self.cfg.success_progress_scale * progress
            + self.cfg.tracking_bonus * tracking_ok
            - self.cfg.tracking_lost_penalty * tracking_bad
            - self.cfg.stuck_penalty * stuck
            - self.cfg.body_contact_penalty * body_contact
            - self.cfg.excessive_turn_penalty * wz_abs
        )

        info = {
            "progress": float(progress.mean().item()),
            "tracking_ok": float(tracking_ok.mean().item()),
            "tracking_bad": float(tracking_bad.mean().item()),
            "stuck": float(stuck.mean().item()),
            "body_contact": float(body_contact.mean().item()),
            "wz_abs": float(wz_abs.mean().item()),
            "reward_mean": float(reward.mean().item()),
        }
        return reward, info

    # -----------------------------------------------------
    # gym API
    # -----------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        obs = self.env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]

        self._last_low_obs = obs
        self.last_high_action.zero_()
        self.cmd_filter.reset()
        self.prev_pos_xy = self.env_unwrapped.scene["robot"].data.root_pos_w[:, :2].clone()
        self.hl_step_count = 0

        # ===== 新增：reset episode 統計 =====
        self.ep_reward_sum = 0.0
        self.ep_length = 0
        self.ep_start_time = time.time()

        if hasattr(self.low_level_agent.agent, "is_rnn") and self.low_level_agent.agent.is_rnn:
            self.low_level_agent.agent.init_rnn()

        obs_np = self._get_obs_numpy()
        info = {}
        return obs_np, info

    def step(self, action):
        self.hl_step_count += 1

        action_t = torch.tensor(action, device=self.device, dtype=torch.float32)
        if action_t.ndim == 1:
            action_t = action_t.unsqueeze(0)

        if action_t.shape != (self.num_envs, self.act_dim):
            raise ValueError(
                f"Expected action shape ({self.num_envs}, {self.act_dim}), got {tuple(action_t.shape)}"
            )

        self.last_high_action[:] = action_t

        raw_cmd = self._action_to_command(action_t)
        cmd = self.cmd_filter.update(raw_cmd)

        low_level_done_any = False
        low_level_info = {}

        for _ in range(self.cfg.hl_decimation):
            self._set_base_velocity_command(cmd)

            low_action = self.low_level_agent.act(self._last_low_obs)
            out = self.env.step(low_action)

            if len(out) == 4:
                obs, _, dones, infos = out
                self._last_low_obs = obs
            else:
                raise RuntimeError("Unexpected low-level env.step output format")

            if hasattr(self.env_unwrapped, "ros2_manager") and self.env_unwrapped.ros2_manager is not None:
                self.env_unwrapped.ros2_manager.update(dt=self.env_unwrapped.step_dt)

            if len(dones) > 0:
                self.low_level_agent.reset_rnn_if_needed(dones)
                if torch.any(dones).item():
                    low_level_done_any = True
                    low_level_info = infos
                    break

        reward_t, reward_info = self._compute_reward(cmd)
        reward_scalar = float(reward_t.mean().item())
        obs_np = self._get_obs_numpy()

        self.prev_pos_xy = self.env_unwrapped.scene["robot"].data.root_pos_w[:, :2].clone()

        terminated = bool(low_level_done_any)
        truncated = bool(self.hl_step_count >= self.cfg.max_episode_hl_steps)

        # ===== 新增：episode 累積 =====
        self.ep_reward_sum += reward_scalar
        self.ep_length += 1

        info = dict(reward_info)
        info["cmd_vx"] = float(cmd[:, 0].mean().item())
        info["cmd_wz"] = float(cmd[:, 2].mean().item())

        # 可選：把 low-level 資訊掛上去，但不要覆蓋 episode
        if low_level_info is not None:
            info["low_level_info"] = low_level_info

        # ===== 關鍵修正：done/truncated 時補標準 episode dict =====
        if terminated or truncated:
            info["episode"] = {
                "r": float(self.ep_reward_sum),
                "l": int(self.ep_length),
                "t": float(time.time() - self.ep_start_time),
            }

        return obs_np, reward_scalar, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass