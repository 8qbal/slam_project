"""Goal-conditioned high-level navigation env from the paper (references/main.pdf Sec. 5.2, Tables 5-6).

Observation (9, Table 5), all in the SLAM map frame:
    SLAM x, y, yaw | goal dx, dy | goal heading error | base vx, vy | base wz

ORB-SLAM3 (``orbslam3_rgbd_node``) expresses its pose relative to the first tracked camera pose, i.e. in the frame
of the robot's start pose. Without ORB-SLAM3 (training), the "SLAM" pose is the ground-truth pose expressed in the
start-pose frame plus small noise, so the policy sees the same frame in training and at play time. Goals are given in
the world frame and converted to the start-pose frame when an episode starts.

The paper does not give the high-level reward. This one is a plain goal-reaching reward: progress towards the goal,
a success bonus, a small per-step time penalty and a penalty when the low-level episode terminates (fall / illegal
contact). Distances for the reward use the ground-truth pose.

``HighLevelIsaacVecEnv`` (11-dim observation, no goal) is unchanged so the rl_vslam_v0 policy still runs.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from .high_level_vec_env import HighLevelEnvCfg, HighLevelIsaacVecEnv, quat_to_yaw


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


@dataclass
class PaperHighLevelEnvCfg(HighLevelEnvCfg):
    # Table 6
    hl_decimation: int = 2
    cmd_smoothing_alpha: float = 0.25
    vx_min: float = 0.0
    vx_max: float = 1.0
    wz_min: float = -0.4
    wz_max: float = 0.4
    max_episode_hl_steps: int = 400

    # Fixed goal of the paper map in map coordinates (paper_cfg.PAPER_MAPS), used in training and evaluation.
    goal_w: Optional[Tuple[float, float]] = None
    goal_radius: float = 1.0

    # Reward (not specified in the paper)
    progress_scale: float = 5.0
    success_bonus: float = 10.0
    time_penalty: float = 0.01
    fall_penalty: float = 5.0

    # Pose source: ORB-SLAM3 when Ros2Manager provides it, otherwise ground truth + noise (training).
    use_orb_pose: bool = True
    fake_slam_noise_std: float = 0.01


class PaperHighLevelIsaacVecEnv(HighLevelIsaacVecEnv):
    def __init__(self, low_level_env, low_level_agent, cfg: Optional[PaperHighLevelEnvCfg] = None):
        super().__init__(low_level_env, low_level_agent, cfg or PaperHighLevelEnvCfg())
        if self.cfg.goal_w is None:
            raise ValueError("PaperHighLevelEnvCfg.goal_w must be set (see paper_cfg.PAPER_MAPS)")

        self.obs_dim = 9
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        n, dev = self.num_envs, self.device
        self.start_xy = torch.zeros((n, 2), device=dev)
        self.start_yaw = torch.zeros((n,), device=dev)
        self.goal_w = torch.zeros((n, 2), device=dev)
        self.goal_s = torch.zeros((n, 2), device=dev)
        self.prev_dist = torch.zeros((n,), device=dev)

        contact_sensor = self.env_unwrapped.scene.sensors["contact_forces"]
        self._body_ids, _ = contact_sensor.find_bodies("body")

    # ------------------------------------------------------------------ poses
    def _gt_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        robot = self.env_unwrapped.scene["robot"].data
        return robot.root_pos_w.torch[:, :2].clone(), quat_to_yaw(robot.root_quat_w.torch)

    def _world_to_start(self, xy_w: torch.Tensor) -> torch.Tensor:
        d = xy_w - self.start_xy
        c, s = torch.cos(self.start_yaw), torch.sin(self.start_yaw)
        return torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], dim=-1)

    def _orb_pose(self) -> Optional[torch.Tensor]:
        res = getattr(self.env_unwrapped, "orb_slam_res", None)
        if not self.cfg.use_orb_pose or not isinstance(res, dict):
            return None
        return res.get("pose_xyyaw")

    def _slam_pose(self) -> torch.Tensor:
        orb = self._orb_pose()
        if orb is not None:
            return orb
        xy_w, yaw_w = self._gt_pose()
        pose = torch.cat([self._world_to_start(xy_w), wrap_to_pi(yaw_w - self.start_yaw).unsqueeze(-1)], dim=-1)
        return pose + torch.randn_like(pose) * self.cfg.fake_slam_noise_std

    # ------------------------------------------------------------------ episodes
    def _start_episodes(self, env_ids: torch.Tensor):
        xy_w, yaw_w = self._gt_pose()
        self.start_xy[env_ids] = xy_w[env_ids]
        self.start_yaw[env_ids] = yaw_w[env_ids]

        # the goal is given in map coordinates; every env has its own copy of the map at its env origin
        goal = torch.tensor(self.cfg.goal_w, device=self.device, dtype=torch.float32)
        self.goal_w[env_ids] = goal + self.env_unwrapped.scene.env_origins[env_ids, :2]

        self.goal_s[:] = self._world_to_start(self.goal_w)
        self.prev_dist[env_ids] = torch.norm(self.goal_w[env_ids] - xy_w[env_ids], dim=-1)

    def _reset_robots(self, env_ids: Sequence[int]):
        """Send robots back to the map start (the goal is fixed, so an episode must not start where the last ended).

        The low-level env only resets robots that fell or timed out; this is the same reset sequence it runs inside
        ``step()``, for episodes that ended by reaching the goal or by the 400-step high-level limit.
        """
        u = self.env_unwrapped
        u._reset_idx(torch.as_tensor(env_ids, dtype=torch.long, device=u.device))
        u.scene.write_data_to_sim()
        u.sim.forward()
        if u.has_rtx_sensors:
            for _ in range(max(1, u.cfg.num_rerenders_on_reset)):
                u.sim.render()
        self._last_low_obs = self.env._process_obs(u.observation_manager.compute(update_history=True))

    def _reset_high_level_states(self, env_ids: Sequence[int]):
        super()._reset_high_level_states(env_ids)
        if len(env_ids) > 0:
            self._start_episodes(torch.as_tensor(env_ids, dtype=torch.long, device=self.device))

    def reset(self) -> VecEnvObs:
        super().reset()
        self._start_episodes(torch.arange(self.num_envs, device=self.device))
        return self._get_obs_numpy()

    # ------------------------------------------------------------------ obs / reward
    def _get_obs_tensor(self) -> torch.Tensor:
        pose = self._slam_pose()
        delta = self.goal_s - pose[:, :2]
        heading_err = wrap_to_pi(torch.atan2(delta[:, 1], delta[:, 0]) - pose[:, 2]).unsqueeze(-1)
        robot = self.env_unwrapped.scene["robot"].data
        return torch.cat(
            [pose, delta, heading_err, robot.root_lin_vel_b.torch[:, :2], robot.root_ang_vel_b.torch[:, 2:3]], dim=-1
        )

    def _body_contact_flag(self) -> torch.Tensor:
        forces = self.env_unwrapped.scene.sensors["contact_forces"].data.net_forces_w.torch[:, self._body_ids, :]
        return torch.any(torch.norm(forces, dim=-1) > 1.0, dim=1).float()

    def _compute_goal_reward(self, low_done: torch.Tensor, fell: torch.Tensor):
        # For envs the low-level env just reset, the robot already stands at its new spawn pose: no progress / success.
        xy_w, _ = self._gt_pose()
        dist = torch.norm(self.goal_w - xy_w, dim=-1)
        progress = torch.where(low_done, torch.zeros_like(dist), self.prev_dist - dist)
        reached = (dist < self.cfg.goal_radius) & ~low_done
        self.prev_dist = dist

        reward = (
            self.cfg.progress_scale * progress
            - self.cfg.time_penalty
            + self.cfg.success_bonus * reached.float()
            - self.cfg.fall_penalty * fell.float()
        )
        info_mean = {
            "progress": float(progress.mean().item()),
            "dist_to_goal": float(dist.mean().item()),
            "goal_reached": float(reached.float().mean().item()),
            "fell": float(fell.float().mean().item()),
            "body_contact": float(self._body_contact_flag().mean().item()),
            "reward_step_mean": float(reward.mean().item()),
        }
        return reward, reached, info_mean

    # ------------------------------------------------------------------ step
    def step_wait(self) -> VecEnvStepReturn:
        if self._pending_actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        action_t = torch.as_tensor(self._pending_actions, device=self.device, dtype=torch.float32).reshape(
            self.num_envs, self.act_dim
        )

        self.hl_step_count += 1
        self.last_high_action[:] = action_t
        cmd = self.cmd_filter.update(self._action_to_command(action_t))
        self._set_base_velocity_command(cmd)

        low_done = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        low_timeout = torch.zeros_like(low_done)
        for _ in range(self.cfg.hl_decimation):
            obs, _, dones, infos = self.env.step(self.low_level_agent.act(self._last_low_obs))
            self._last_low_obs = obs
            if getattr(self.env_unwrapped, "ros2_manager", None) is not None:
                self.env_unwrapped.ros2_manager.update(dt=self.env_unwrapped.step_dt)

            dones_t = torch.as_tensor(dones, device=self.device).bool()
            low_done |= dones_t
            if isinstance(infos, dict) and "time_outs" in infos:
                low_timeout |= torch.as_tensor(infos["time_outs"], device=self.device).bool() & dones_t
            if torch.any(dones_t):
                break

        fell = low_done & ~low_timeout
        reward_t, reached, info_mean = self._compute_goal_reward(low_done, fell)
        truncated_t = ~reached & ~fell & ((self.hl_step_count >= self.cfg.max_episode_hl_steps) | low_timeout)
        done_t = reached | low_done | truncated_t

        self.ep_rewards += reward_t
        self.ep_lengths += 1
        obs_np = self._get_obs_numpy()

        rewards = reward_t.cpu().numpy().astype(np.float32)
        dones_np = done_t.cpu().numpy().astype(bool)
        infos_out: List[Dict] = []
        for i in range(self.num_envs):
            info_i = dict(info_mean, cmd_vx=float(cmd[i, 0].item()), cmd_wz=float(cmd[i, 2].item()))
            if dones_np[i]:
                info_i["terminal_observation"] = obs_np[i]
                info_i["is_success"] = bool(reached[i].item())
                info_i["episode"] = {
                    "r": float(self.ep_rewards[i].item()),
                    "l": int(self.ep_lengths[i].item()),
                    "t": float(time.time() - self.ep_start_times[i]),
                }
                if truncated_t[i].item():
                    info_i["TimeLimit.truncated"] = True
            infos_out.append(info_i)

        done_ids = torch.nonzero(done_t).squeeze(-1).cpu().tolist()
        if done_ids:
            not_reset_ids = torch.nonzero(done_t & ~low_done).squeeze(-1).cpu().tolist()
            if not_reset_ids:
                self._reset_robots(not_reset_ids)
            self._reset_high_level_states(done_ids)
            obs_np = self._get_obs_numpy()

        self._pending_actions = None
        return obs_np, rewards, dones_np, infos_out
