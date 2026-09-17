import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn


def quat_to_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def get_camera_depth_stats_from_env(env_unwrapped, sensor_name: str = "train_camera") -> torch.Tensor:
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
    if (
        hasattr(env_unwrapped, "orb_slam_res")
        and isinstance(env_unwrapped.orb_slam_res, dict)
        and "status" in env_unwrapped.orb_slam_res
    ):
        status = env_unwrapped.orb_slam_res["status"]
        if status is not None:
            return status

    return torch.ones((env_unwrapped.num_envs, 1), device=env_unwrapped.device, dtype=torch.float32)


class CommandFilter:
    def __init__(self, num_envs, device, alpha=0.25):
        self.alpha = alpha
        self.cmd = torch.zeros((num_envs, 3), device=device)

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            self.cmd.zero_()
        else:
            self.cmd[env_ids] = 0.0

    def update(self, new_cmd):
        self.cmd = (1.0 - self.alpha) * self.cmd + self.alpha * new_cmd
        return self.cmd


class FrozenLowLevelAgentAdapter:
    def __init__(self, agent, num_envs: int):
        self.agent = agent
        self.num_envs = num_envs
        self.expected_obs_dim = self.agent.model.a2c_network.actor_mlp[0].in_features

    def _unwrap_obs(self, obs):
        while isinstance(obs, dict):
            if "obs" not in obs:
                raise ValueError(f"Unexpected obs dict keys: {list(obs.keys())}")
            obs = obs["obs"]
        return obs

    @torch.inference_mode()
    def act(self, obs):
        base_obs = self._unwrap_obs(obs)

        if isinstance(base_obs, np.ndarray):
            base_obs = torch.as_tensor(base_obs, dtype=torch.float32)
        elif not isinstance(base_obs, torch.Tensor):
            base_obs = torch.tensor(base_obs, dtype=torch.float32)

        if base_obs.ndim == 1 and base_obs.numel() == self.num_envs * self.expected_obs_dim:
            base_obs = base_obs.view(self.num_envs, self.expected_obs_dim)
        elif base_obs.ndim == 2 and base_obs.shape[0] == 1 and base_obs.shape[1] == self.num_envs * self.expected_obs_dim:
            base_obs = base_obs.view(self.num_envs, self.expected_obs_dim)
        elif base_obs.ndim == 2 and base_obs.shape == (self.num_envs, self.expected_obs_dim):
            pass
        else:
            raise ValueError(
                f"Unexpected low-level obs shape {tuple(base_obs.shape)}, "
                f"expected ({self.num_envs}, {self.expected_obs_dim}) "
                f"or flattened ({self.num_envs * self.expected_obs_dim},)"
            )

        device = next(self.agent.model.parameters()).device
        base_obs = base_obs.to(device)

        if self.agent.is_rnn:
            raise RuntimeError("目前這版 adapter 先不支援 RNN low-level policy，請先用非 RNN policy。")

        action_list = []
        for i in range(self.num_envs):
            single_obs = base_obs[i].unsqueeze(0)
            single_action = self.agent.get_action(single_obs, is_deterministic=True)

            if isinstance(single_action, np.ndarray):
                single_action = torch.as_tensor(single_action, dtype=torch.float32, device=device)
            elif not isinstance(single_action, torch.Tensor):
                single_action = torch.tensor(single_action, dtype=torch.float32, device=device)

            if single_action.ndim == 1:
                single_action = single_action.unsqueeze(0)

            action_list.append(single_action)

        action = torch.cat(action_list, dim=0)
        return action

    def reset_rnn_if_needed(self, done_ids):
        if self.agent.is_rnn and self.agent.states is not None and len(done_ids) > 0:
            done_ids_t = torch.as_tensor(done_ids, dtype=torch.long, device=self.agent.states[0].device)
            for s in self.agent.states:
                s[:, done_ids_t, :] = 0.0

    def init_rnn(self):
        if self.agent.is_rnn:
            self.agent.init_rnn()


@dataclass
class HighLevelEnvCfg:
    hl_decimation: int = 2
    cmd_smoothing_alpha: float = 0.25

    vx_min: float = 0.0
    vx_max: float = 0.50
    wz_min: float = -0.12
    wz_max: float = 0.12

    success_progress_scale: float = 6.0
    tracking_bonus: float = 0.0
    tracking_lost_penalty: float = 0.02
    stuck_penalty: float = 0.005
    body_contact_penalty: float = 0.2
    excessive_turn_penalty: float = 0.005

    stuck_dist_threshold: float = 0.001
    max_episode_hl_steps: int = 400


class HighLevelIsaacVecEnv(VecEnv):
    """
    SB3 VecEnv adapter for one Isaac env with internal num_envs = N.
    """

    def __init__(self, low_level_env, low_level_agent, cfg: Optional[HighLevelEnvCfg] = None):
        self.env = low_level_env
        self.env_unwrapped = low_level_env.unwrapped
        self.cfg = cfg or HighLevelEnvCfg()

        self.num_envs = self.env_unwrapped.num_envs
        self.device = self.env_unwrapped.device
        self.low_level_agent = FrozenLowLevelAgentAdapter(low_level_agent, self.num_envs)

        self.obs_dim = 11
        self.act_dim = 2

        observation_space = __import__("gymnasium").spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        action_space = __import__("gymnasium").spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.act_dim,),
            dtype=np.float32,
        )
        super().__init__(self.num_envs, observation_space, action_space)

        self.cmd_filter = CommandFilter(self.num_envs, self.device, self.cfg.cmd_smoothing_alpha)
        self.last_high_action = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.prev_pos_xy = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.hl_step_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.ep_rewards = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.ep_lengths = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.ep_start_times = np.zeros((self.num_envs,), dtype=np.float64)

        self._last_low_obs = None
        self._pending_actions = None
        self.reset_infos: List[Dict] = [{} for _ in range(self.num_envs)]

    def _action_to_command(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.clamp(action, -1.0, 1.0)
        vx = (action[:, 0] + 1.0) * 0.5 * (self.cfg.vx_max - self.cfg.vx_min) + self.cfg.vx_min
        wz = (action[:, 1] + 1.0) * 0.5 * (self.cfg.wz_max - self.cfg.wz_min) + self.cfg.wz_min

        cmd = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        cmd[:, 0] = vx
        cmd[:, 2] = wz
        return cmd

    def _set_base_velocity_command(self, cmd_tensor: torch.Tensor):
        if hasattr(self.env_unwrapped.command_manager, "set_command"):
            self.env_unwrapped.command_manager.set_command("base_velocity", cmd_tensor)
        else:
            self.env_unwrapped.command_manager._terms["base_velocity"].command[:] = cmd_tensor

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
        return self._get_obs_tensor().detach().cpu().numpy().astype(np.float32)

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
        return torch.any(forces_norm > 1.0, dim=1).float()



    # def _compute_reward(self, action_cmd: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    #     # 目前位置與前一個 high-level step 的位移
    #     curr_pos_xy = self.env_unwrapped.scene["robot"].data.root_pos_w[:, :2]
    #     delta_xy = curr_pos_xy - self.prev_pos_xy

    #     # 用 robot 當前朝向，計算「沿機器人前向」的真實位移
    #     quat = self.env_unwrapped.scene["robot"].data.root_quat_w
    #     yaw = quat_to_yaw(quat)

    #     forward_dir = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    #     progress = torch.sum(delta_xy * forward_dir, dim=-1)
    #     progress = torch.clamp(progress, min=0.0)

    #     # ORB / fake SLAM tracking 狀態
    #     orb_status = get_orb_status_from_env(self.env_unwrapped).squeeze(-1)
    #     tracking_ok = orb_status
    #     tracking_bad = 1.0 - orb_status

    #     # stuck 判定放寬
    #     stuck = (progress < self.cfg.stuck_dist_threshold).float()

    #     body_contact = self._body_contact_flag()
    #     wz_abs = torch.abs(action_cmd[:, 2])

    #     # progress_term = self.cfg.success_progress_scale * progress
    #     # tracking_bonus_term = self.cfg.tracking_bonus * tracking_ok
    #     # tracking_lost_term = -self.cfg.tracking_lost_penalty * tracking_bad
    #     # stuck_term = -self.cfg.stuck_penalty * stuck
    #     # body_contact_term = -self.cfg.body_contact_penalty * body_contact
    #     # turn_term = -self.cfg.excessive_turn_penalty * wz_abs

    #     # reward = (
    #     #     progress_term
    #     #     + tracking_bonus_term
    #     #     + tracking_lost_term
    #     #     + stuck_term
    #     #     + body_contact_term
    #     #     + turn_term
    #     # )
    #     # --- Forward progress (主要 reward) ---
    #     progress_term = 5.0 * progress

    #     # --- Stuck penalty (避免卡住) ---
    #     stuck_term = -0.05 * stuck

    #     # --- Body collision penalty ---
    #     body_contact_term = -0.05 * body_contact

    #     # --- 小幅度轉彎懲罰 (避免抖動) ---
    #     turn_term = -0.01 * wz_abs

    #     reward = (
    #         progress_term
    #         + stuck_term
    #         + body_contact_term
    #         + turn_term
    #     )

    #     info_mean = {
    #         "progress": float(progress.mean().item()),
    #         "tracking_ok": float(tracking_ok.mean().item()),
    #         "tracking_bad": float(tracking_bad.mean().item()),
    #         "stuck": float(stuck.mean().item()),
    #         "body_contact": float(body_contact.mean().item()),
    #         "wz_abs": float(wz_abs.mean().item()),
    #         "reward_step_mean": float(reward.mean().item()),
    #         "reward_progress_term": float(progress_term.mean().item()),
    #         "reward_tracking_bonus_term": float(tracking_bonus_term.mean().item()),
    #         "reward_tracking_lost_term": float(tracking_lost_term.mean().item()),
    #         "reward_stuck_term": float(stuck_term.mean().item()),
    #         "reward_body_contact_term": float(body_contact_term.mean().item()),
    #         "reward_turn_term": float(turn_term.mean().item()),
    #     }
    #     return reward, info_mean
    def _compute_reward(self, action_cmd: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        progress = torch.clamp(
            self.env_unwrapped.scene["robot"].data.root_lin_vel_b[:, 0],
            min=0.0,
        ) * self.env_unwrapped.step_dt * self.cfg.hl_decimation

        stuck = (progress < self.cfg.stuck_dist_threshold).float()
        body_contact = self._body_contact_flag()
        wz_abs = torch.abs(action_cmd[:, 2])

        # forward progress
        progress_term = 10.0 * progress

        # penalties
        stuck_term = -0.02 * stuck
        body_contact_term = -0.02 * body_contact
        turn_term = -0.005 * wz_abs

        reward = (
            progress_term
            + stuck_term
            + body_contact_term
            + turn_term
        )

        info_mean = {
            "progress": float(progress.mean().item()),
            "stuck": float(stuck.mean().item()),
            "body_contact": float(body_contact.mean().item()),
            "wz_abs": float(wz_abs.mean().item()),
            "reward_step_mean": float(reward.mean().item()),
            "reward_progress_term": float(progress_term.mean().item()),
            "reward_stuck_term": float(stuck_term.mean().item()),
            "reward_body_contact_term": float(body_contact_term.mean().item()),
            "reward_turn_term": float(turn_term.mean().item()),
        }
        return reward, info_mean

    def _reset_high_level_states(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        self.last_high_action[env_ids_t] = 0.0
        self.cmd_filter.reset(env_ids_t)
        self.prev_pos_xy[env_ids_t] = self.env_unwrapped.scene["robot"].data.root_pos_w[env_ids_t, :2].clone()
        self.hl_step_count[env_ids_t] = 0

        self.ep_rewards[env_ids_t] = 0.0
        self.ep_lengths[env_ids_t] = 0
        for i in env_ids:
            self.ep_start_times[i] = time.time()

        if hasattr(self.env_unwrapped, "ros2_manager") and self.env_unwrapped.ros2_manager is not None:
            self.env_unwrapped.ros2_manager.reset(env_ids_t)

        self.low_level_agent.reset_rnn_if_needed(env_ids)

    def reset(self) -> VecEnvObs:
        obs = self.env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        self._last_low_obs = obs

        self.last_high_action.zero_()
        self.cmd_filter.reset()
        self.prev_pos_xy = self.env_unwrapped.scene["robot"].data.root_pos_w[:, :2].clone()
        self.hl_step_count.zero_()

        self.ep_rewards.zero_()
        self.ep_lengths.zero_()
        self.ep_start_times[:] = time.time()

        self.low_level_agent.init_rnn()
        return self._get_obs_numpy()

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        if self._pending_actions is None:
            raise RuntimeError("step_async must be called before step_wait")

        action_t = torch.as_tensor(self._pending_actions, device=self.device, dtype=torch.float32)
        if action_t.ndim == 1:
            action_t = action_t.reshape(1, -1)

        if action_t.shape != (self.num_envs, self.act_dim):
            raise ValueError(
                f"Expected action shape ({self.num_envs}, {self.act_dim}), got {tuple(action_t.shape)}"
            )

        self.hl_step_count += 1
        self.last_high_action[:] = action_t

        raw_cmd = self._action_to_command(action_t)
        cmd = self.cmd_filter.update(raw_cmd)

        low_level_done = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        low_infos = [{} for _ in range(self.num_envs)]

        self._set_base_velocity_command(cmd)

        for _ in range(self.cfg.hl_decimation):
            low_action = self.low_level_agent.act(self._last_low_obs)
            obs, _, dones, infos = self.env.step(low_action)
            self._last_low_obs = obs

            if hasattr(self.env_unwrapped, "ros2_manager") and self.env_unwrapped.ros2_manager is not None:
                self.env_unwrapped.ros2_manager.update(dt=self.env_unwrapped.step_dt)

            if isinstance(dones, np.ndarray):
                dones_t = torch.as_tensor(dones, device=self.device, dtype=torch.bool)
            else:
                dones_t = dones.to(self.device).bool()

            low_level_done |= dones_t

            if isinstance(infos, list):
                low_infos = infos
            elif isinstance(infos, dict):
                low_infos = [infos for _ in range(self.num_envs)]

            if torch.any(dones_t):
                break

        reward_t, reward_mean_info = self._compute_reward(cmd)
        obs_np = self._get_obs_numpy()
        self.prev_pos_xy = self.env_unwrapped.scene["robot"].data.root_pos_w[:, :2].clone()

        self.ep_rewards += reward_t
        self.ep_lengths += 1

        truncated_t = self.hl_step_count >= self.cfg.max_episode_hl_steps
        done_t = low_level_done | truncated_t

        rewards = reward_t.detach().cpu().numpy().astype(np.float32)
        dones = done_t.detach().cpu().numpy().astype(bool)

        infos: List[Dict] = []
        for i in range(self.num_envs):
            info_i = {}

            if i < len(low_infos) and isinstance(low_infos[i], dict):
                low_info_i = dict(low_infos[i])
                low_info_i.pop("episode", None)
                info_i.update(low_info_i)

            # info_i.update({
            #     "progress": reward_mean_info["progress"],
            #     "tracking_ok": reward_mean_info["tracking_ok"],
            #     "tracking_bad": reward_mean_info["tracking_bad"],
            #     "stuck": reward_mean_info["stuck"],
            #     "body_contact": reward_mean_info["body_contact"],
            #     "wz_abs": reward_mean_info["wz_abs"],
            #     "reward_step_mean": reward_mean_info["reward_step_mean"],
            #     "reward_progress_term": reward_mean_info["reward_progress_term"],
            #     "reward_tracking_bonus_term": reward_mean_info["reward_tracking_bonus_term"],
            #     "reward_tracking_lost_term": reward_mean_info["reward_tracking_lost_term"],
            #     "reward_stuck_term": reward_mean_info["reward_stuck_term"],
            #     "reward_body_contact_term": reward_mean_info["reward_body_contact_term"],
            #     "reward_turn_term": reward_mean_info["reward_turn_term"],
            #     "cmd_vx": float(cmd[i, 0].item()),
            #     "cmd_wz": float(cmd[i, 2].item()),
            # })
            info_i.update({
                "progress": reward_mean_info["progress"],
                "stuck": reward_mean_info["stuck"],
                "body_contact": reward_mean_info["body_contact"],
                "wz_abs": reward_mean_info["wz_abs"],
                "reward_step_mean": reward_mean_info["reward_step_mean"],
                "reward_progress_term": reward_mean_info["reward_progress_term"],
                "reward_stuck_term": reward_mean_info["reward_stuck_term"],
                "reward_body_contact_term": reward_mean_info["reward_body_contact_term"],
                "reward_turn_term": reward_mean_info["reward_turn_term"],
                "cmd_vx": float(cmd[i, 0].item()),
                "cmd_wz": float(cmd[i, 2].item()),
            })

            if dones[i]:
                info_i["episode"] = {
                    "r": float(self.ep_rewards[i].item()),
                    "l": int(self.ep_lengths[i].item()),
                    "t": float(time.time() - self.ep_start_times[i]),
                }
                if truncated_t[i].item():
                    info_i["TimeLimit.truncated"] = True

            infos.append(info_i)

        done_ids = torch.nonzero(done_t).squeeze(-1).detach().cpu().numpy().tolist()
        if len(done_ids) > 0:
            self._reset_high_level_states(done_ids)

        self._pending_actions = None
        return obs_np, rewards, dones, infos

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def get_attr(self, attr_name: str, indices=None):
        return [getattr(self, attr_name) for _ in range(self.num_envs)]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        method = getattr(self, method_name)
        return [method(*method_args, **method_kwargs) for _ in range(self.num_envs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in range(self.num_envs)]

    def render(self, mode: Optional[str] = None):
        return None