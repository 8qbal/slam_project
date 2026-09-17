import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, Any

from isaaclab.managers import SceneEntityCfg

# 你自己的低層 env 類
from spot_vslam.my_project.envs import ManagerBasedRLEnv


# ==========================================================
# 小工具
# ==========================================================

def quat_to_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """quat format: (w, x, y, z)"""
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def get_pseudo_orb_pose_xyyaw(env: ManagerBasedRLEnv, pos_noise=0.01, yaw_noise=0.01) -> torch.Tensor:
    """
    訓練高層第一版：
    先用 GT + noise 假裝 ORB-SLAM3 輸出
    shape = (N, 3) => [x, y, yaw]
    """
    pos_xy = env.scene["robot"].data.root_pos_w[:, :2].clone()
    quat = env.scene["robot"].data.root_quat_w.clone()
    yaw = quat_to_yaw(quat).unsqueeze(-1)

    pose = torch.cat([pos_xy, yaw], dim=-1)

    noise = torch.zeros_like(pose)
    noise[:, :2] = torch.randn_like(pose[:, :2]) * pos_noise
    noise[:, 2] = torch.randn_like(pose[:, 2]) * yaw_noise
    return pose + noise


def get_pseudo_orb_status(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    第一版先假設 tracking 都正常
    shape = (N, 1)
    """
    return torch.ones((env.num_envs, 1), device=env.device)


def get_camera_depth_stats(env: ManagerBasedRLEnv, sensor_name: str = "train_camera") -> torch.Tensor:
    """
    跟你原本 camera_depth_stats 類似，但這裡獨立寫一版，避免互相依賴太深。
    output shape = (N, 5)
    """
    sensor = env.scene.sensors[sensor_name]
    depth = sensor.data.output["distance_to_image_plane"].clone()
    depth = torch.nan_to_num(depth, nan=5.0, posinf=5.0, neginf=5.0)
    depth = torch.clamp(depth, min=0.0, max=5.0)
    depth = depth.squeeze(-1)  # (N, H, W)

    left = depth[:, 20:50, 5:25].mean(dim=(1, 2))
    center = depth[:, 20:50, 25:55].mean(dim=(1, 2))
    right = depth[:, 20:50, 55:75].mean(dim=(1, 2))

    front_patch = depth[:, 20:50, 25:55].reshape(env.num_envs, -1)
    front_min = front_patch.min(dim=1).values
    lr_balance = left - right

    stats = torch.stack([left, center, right, front_min, lr_balance], dim=1)
    stats[:, :4] = (stats[:, :4] / 5.0) * 2.0 - 1.0
    stats[:, 4] = torch.clamp(stats[:, 4] / 5.0, min=-1.0, max=1.0)
    return stats


# ==========================================================
# 高層 command buffer
# ==========================================================

@dataclass
class HighLevelCommandCfg:
    vx_min: float = 0.0
    vx_max: float = 0.6
    wz_min: float = -0.3
    wz_max: float = 0.3


class HighLevelCommandBuffer:
    def __init__(self, num_envs: int, device: str, cfg: HighLevelCommandCfg):
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg
        self.commands = torch.zeros((num_envs, 3), device=device)  # [vx, vy, wz]

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            self.commands.zero_()
        else:
            self.commands[env_ids] = 0.0

    def set_from_action(self, action: torch.Tensor):
        """
        action in [-1, 1], shape = (N, 2)
        maps to vx, wz
        """
        vx = (action[:, 0] + 1.0) * 0.5 * (self.cfg.vx_max - self.cfg.vx_min) + self.cfg.vx_min
        wz = (action[:, 1] + 1.0) * 0.5 * (self.cfg.wz_max - self.cfg.wz_min) + self.cfg.wz_min

        self.commands[:, 0] = vx
        self.commands[:, 1] = 0.0
        self.commands[:, 2] = wz

    def get(self) -> torch.Tensor:
        return self.commands


# ==========================================================
# 凍結低層 policy
# ==========================================================

class FrozenLowLevelPolicy(nn.Module):
    def __init__(self, actor: nn.Module):
        super().__init__()
        self.actor = actor.eval()
        for p in self.actor.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.actor(obs)
        if isinstance(out, dict):
            if "mus" in out:
                return out["mus"]
            if "actions" in out:
                return out["actions"]
        return out


# ==========================================================
# 高層 env wrapper
# ==========================================================

class HighLevelSpotEnv:
    """
    用一個已存在的低層 env + frozen low-level policy
    包成高層 env。

    高層每 step 一次
    內部連跑 hl_decimation 次低層
    """
    def __init__(
        self,
        low_level_env: ManagerBasedRLEnv,
        low_level_policy: FrozenLowLevelPolicy,
        hl_decimation: int = 5,
        command_cfg: HighLevelCommandCfg = HighLevelCommandCfg(),
    ):
        self.env = low_level_env
        self.low_level_policy = low_level_policy
        self.hl_decimation = hl_decimation

        self.num_envs = self.env.num_envs
        self.device = self.env.device

        self.cmd_buffer = HighLevelCommandBuffer(
            num_envs=self.num_envs,
            device=self.device,
            cfg=command_cfg,
        )

        self.last_high_action = torch.zeros((self.num_envs, 2), device=self.device)
        self.prev_pos_xy = torch.zeros((self.num_envs, 2), device=self.device)

        # obs dim:
        # orb_pose_xyyaw: 3
        # orb_status: 1
        # last_high_action: 2
        # depth_stats: 5
        # delta_from_start: 2
        self.obs_dim = 13
        self.act_dim = 2

    def _set_low_level_command(self):
        cmd = self.cmd_buffer.get()

        # 視你的 command manager 實作而定
        if hasattr(self.env.command_manager, "set_command"):
            self.env.command_manager.set_command("base_velocity", cmd)
        else:
            # IsaacLab 常見內部覆蓋方式
            self.env.command_manager._terms["base_velocity"].command[:] = cmd

    def _get_low_level_obs(self) -> torch.Tensor:
        """
        視你 env 的 obs 結構改
        """
        if hasattr(self.env, "obs_buf"):
            if isinstance(self.env.obs_buf, dict):
                return self.env.obs_buf["policy"]
            return self.env.obs_buf

        # 保底：直接調 observation manager
        return self.env.observation_manager.compute_group("policy")

    def _get_high_level_obs(self, start_pos_xy: torch.Tensor) -> torch.Tensor:
        orb_pose = get_pseudo_orb_pose_xyyaw(self.env)          # (N, 3)
        orb_status = get_pseudo_orb_status(self.env)            # (N, 1)
        depth_stats = get_camera_depth_stats(self.env)          # (N, 5)

        curr_xy = self.env.scene["robot"].data.root_pos_w[:, :2]
        delta_xy = curr_xy - start_pos_xy                       # (N, 2)

        obs = torch.cat(
            [
                orb_pose,
                orb_status,
                self.last_high_action,
                depth_stats,
                delta_xy,
            ],
            dim=-1,
        )
        return obs

    def _compute_high_level_reward(
        self,
        start_pos_xy: torch.Tensor,
        orb_status: torch.Tensor,
    ) -> torch.Tensor:
        curr_xy = self.env.scene["robot"].data.root_pos_w[:, :2]
        delta_xy = curr_xy - start_pos_xy
        progress = torch.norm(delta_xy, dim=1)

        # tracking 正常 bonus
        tracking_bonus = orb_status.squeeze(-1) * 0.2

        # tracking lost penalty
        tracking_lost_penalty = (1.0 - orb_status.squeeze(-1)) * 1.0

        # stuck penalty
        stuck_flag = (progress < 0.03).float()
        stuck_penalty = stuck_flag * 0.5

        # body contact penalty
        contact_sensor = self.env.scene.sensors["contact_forces"]
        body_ids = SceneEntityCfg("contact_forces", body_names=["body"]).body_ids
        current_forces = contact_sensor.data.net_forces_w[:, body_ids, :]
        forces_norm = torch.norm(current_forces, dim=-1)
        body_contact = torch.any(forces_norm > 1.0, dim=1).float()
        body_contact_penalty = body_contact * 1.0

        reward = (
            2.0 * progress
            + 1.0 * tracking_bonus
            - 2.0 * tracking_lost_penalty
            - 1.0 * stuck_penalty
            - 2.0 * body_contact_penalty
        )
        return reward

    @torch.no_grad()
    def reset(self):
        reset_out = self.env.reset()
        if isinstance(reset_out, tuple):
            _, info = reset_out
        else:
            info = {}

        self.cmd_buffer.reset()
        self.last_high_action.zero_()
        self.prev_pos_xy = self.env.scene["robot"].data.root_pos_w[:, :2].clone()

        obs = self._get_high_level_obs(self.prev_pos_xy)
        return obs, info

    @torch.no_grad()
    def step(self, high_action: torch.Tensor):
        """
        high_action shape = (N, 2), assumed in [-1, 1]
        """
        self.last_high_action = high_action.clone()
        self.cmd_buffer.set_from_action(high_action)

        start_pos_xy = self.env.scene["robot"].data.root_pos_w[:, :2].clone()

        done_any = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        info_out: Dict[str, Any] = {}

        for _ in range(self.hl_decimation):
            self._set_low_level_command()

            low_obs = self._get_low_level_obs()
            low_action = self.low_level_policy(low_obs)

            step_out = self.env.step(low_action)
            if len(step_out) == 4:
                _, _, done, info = step_out
            else:
                # 視你的 env 回傳格式調整
                raise RuntimeError("Unexpected low-level env.step output format")

            done_any |= done
            info_out = info

            if torch.any(done_any):
                break

        orb_status = get_pseudo_orb_status(self.env)
        reward = self._compute_high_level_reward(start_pos_xy, orb_status)
        obs = self._get_high_level_obs(start_pos_xy)

        return obs, reward, done_any, info_out