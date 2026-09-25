# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
ORB-aware rule-based high-level controller test with frozen low-level RL policy.

功能：
1. 載入已訓好的 low-level RL policy
2. high-level 先用 rule-based command，不用神經網路
3. ORB 資料第一版先用 GT + noise 假裝
4. 局部避障使用 train_camera depth stats
5. 驗證：
   ORB-like pose/status + depth -> high-level command -> low-level policy -> robot motion
"""

import argparse
import math
import os
import time
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

# -------------------------------------------------
# CLI
# -------------------------------------------------
parser = argparse.ArgumentParser(description="Test ORB-aware rule-based high-level wrapper with frozen low-level policy.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument(
    "--use_manual_command",
    action="store_true",
    default=False,
    help="Use manual scripted command instead of ORB-aware rule-based command.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------------------------------
# Imports after simulator launch
# -------------------------------------------------
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
import spot_vslam.tasks  # noqa: F401


# -------------------------------------------------
# Helper: overwrite low-level command
# -------------------------------------------------
def set_base_velocity_command(env_unwrapped, cmd_tensor: torch.Tensor):
    """
    cmd_tensor shape: (N, 3) => [vx, vy, wz]
    """
    if hasattr(env_unwrapped.command_manager, "set_command"):
        env_unwrapped.command_manager.set_command("base_velocity", cmd_tensor)
    else:
        env_unwrapped.command_manager._terms["base_velocity"].command[:] = cmd_tensor


# -------------------------------------------------
# Manual high-level command for sanity check
# -------------------------------------------------
def build_manual_high_level_command(num_envs, device, t):
    cmd = torch.zeros((num_envs, 3), device=device)

    if t < 100:
        cmd[:, 0] = 0.4
        cmd[:, 1] = 0.0
        cmd[:, 2] = 0.0
    elif t < 200:
        cmd[:, 0] = 0.25
        cmd[:, 1] = 0.0
        cmd[:, 2] = 0.2
    elif t < 300:
        cmd[:, 0] = 0.25
        cmd[:, 1] = 0.0
        cmd[:, 2] = -0.2
    else:
        cmd[:, 0] = 0.0
        cmd[:, 1] = 0.0
        cmd[:, 2] = 0.0

    return cmd


# -------------------------------------------------
# ORB helper
# -------------------------------------------------
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


def get_orb_pose_xyyaw_from_env(env_unwrapped):
    """
    第一版先用 GT + 小噪聲假裝 ORB-SLAM3
    回傳 shape = (N, 3): [x, y, yaw]
    """
    pos_xy = env_unwrapped.scene["robot"].data.root_pos_w[:, :2].clone()
    quat = env_unwrapped.scene["robot"].data.root_quat_w.clone()
    yaw = quat_to_yaw(quat).unsqueeze(-1)

    pose = torch.cat([pos_xy, yaw], dim=-1)

    noise = torch.zeros_like(pose)
    noise[:, 0] = torch.randn_like(pose[:, 0]) * 0.01
    noise[:, 1] = torch.randn_like(pose[:, 1]) * 0.01
    noise[:, 2] = torch.randn_like(pose[:, 2]) * 0.01
    return pose + noise


def get_orb_status_from_env(env_unwrapped):
    """
    第一版先假設 ORB tracking 正常
    shape = (N, 1)
    """
    return torch.ones((env_unwrapped.num_envs, 1), device=env_unwrapped.device)


# -------------------------------------------------
# Depth-based local perception
# -------------------------------------------------
def get_camera_depth_stats_from_env(env_unwrapped, sensor_name: str = "train_camera"):
    """
    從 train_camera 取出簡化深度特徵
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


# -------------------------------------------------
# Optional smoothing
# -------------------------------------------------
class CommandFilter:
    def __init__(self, num_envs, device, alpha=0.25):
        self.alpha = alpha
        self.cmd = torch.zeros((num_envs, 3), device=device)

    def reset(self):
        self.cmd.zero_()

    def update(self, new_cmd):
        self.cmd = (1.0 - self.alpha) * self.cmd + self.alpha * new_cmd
        return self.cmd


# -------------------------------------------------
# ORB-aware rule-based high-level command
# -------------------------------------------------
def build_orb_rule_based_high_level_command(env_unwrapped, device):
    """
    high-level 融合：
    - pseudo ORB pose / status
    - depth stats 局部避障

    邏輯：
    1. tracking lost -> 停下並慢慢旋轉
    2. tracking OK -> depth-based local navigation
    3. ORB yaw 暫時只做 observation/debug，不做強導引
       （之後接真 ORB / frontier 再做全局導航）
    """
    stats = get_camera_depth_stats_from_env(env_unwrapped)
    orb_pose = get_orb_pose_xyyaw_from_env(env_unwrapped)
    orb_status = get_orb_status_from_env(env_unwrapped)

    center = stats[:, 1]
    front_min = stats[:, 3]
    lr_balance = stats[:, 4]

    yaw = orb_pose[:, 2]
    tracking_ok = orb_status[:, 0] > 0.5

    cmd = torch.zeros((env_unwrapped.num_envs, 3), device=device)

    # depth thresholds
    front_clear = center > -0.05
    front_too_close = front_min < -0.70

    # 大 deadband，避免崎嶇路面小噪聲造成誤轉
    turn_left_strong = lr_balance > 0.25
    turn_right_strong = lr_balance < -0.25

    # 預設 vx
    cmd[:, 0] = torch.where(
        front_too_close,
        torch.tensor(0.15, device=device),
        torch.where(
            front_clear,
            torch.tensor(0.50, device=device),
            torch.tensor(0.42, device=device),
        ),
    )

    # vy 固定 0
    cmd[:, 1] = 0.0

    # wz 預設 0
    cmd[:, 2] = 0.0

    # 只有前方真的近，且左右差明顯時才轉
    cmd[:, 2] = torch.where(front_too_close & turn_left_strong, torch.tensor(0.20, device=device), cmd[:, 2])
    cmd[:, 2] = torch.where(front_too_close & turn_right_strong, torch.tensor(-0.20, device=device), cmd[:, 2])

    # 如果 tracking 壞掉，先停下並慢慢旋轉找回
    cmd[:, 0] = torch.where(~tracking_ok, torch.tensor(0.0, device=device), cmd[:, 0])
    cmd[:, 2] = torch.where(~tracking_ok, torch.tensor(0.15, device=device), cmd[:, 2])

    return cmd


def main():
    # -------------------------------------------------
    # Env config / agent config
    # -------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)

    # checkpoint path
    if args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)

    print(f"[INFO] Loading checkpoint from: {resume_path}")

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # -------------------------------------------------
    # Create env
    # -------------------------------------------------
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # -------------------------------------------------
    # Load frozen low-level agent
    # -------------------------------------------------
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)

    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt

    # -------------------------------------------------
    # Reset
    # -------------------------------------------------
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    cmd_filter = CommandFilter(env.unwrapped.num_envs, rl_device, alpha=0.25)
    cmd_filter.reset()

    # -------------------------------------------------
    # Main loop
    # -------------------------------------------------
    t = 0
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # 1. High-level command
            if args_cli.use_manual_command:
                raw_cmd = build_manual_high_level_command(env.unwrapped.num_envs, rl_device, t)
            else:
                raw_cmd = build_orb_rule_based_high_level_command(env.unwrapped, rl_device)

            # 2. Smooth the command a bit
            cmd = cmd_filter.update(raw_cmd)

            # 3. Overwrite low-level command
            set_base_velocity_command(env.unwrapped, cmd)

            # 4. Low-level frozen policy inference
            obs_dict = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_dict, is_deterministic=True)

            # 5. Step env
            obs, _, dones, infos = env.step(actions)

            # debug print
            if t % 20 == 0:
                stats = get_camera_depth_stats_from_env(env.unwrapped)
                orb_pose = get_orb_pose_xyyaw_from_env(env.unwrapped)
                orb_status = get_orb_status_from_env(env.unwrapped)

                print(
                    f"[{t:04d}] "
                    f"cmd(vx,wz)=({cmd[0,0].item():.2f}, {cmd[0,2].item():.2f}) | "
                    f"orb(x,y,yaw)=("
                    f"{orb_pose[0,0].item():.2f}, "
                    f"{orb_pose[0,1].item():.2f}, "
                    f"{orb_pose[0,2].item():.2f}) | "
                    f"status={orb_status[0,0].item():.0f} | "
                    f"depth(center,front_min,lr_balance)=("
                    f"{stats[0,1].item():.2f}, "
                    f"{stats[0,3].item():.2f}, "
                    f"{stats[0,4].item():.2f})"
                )

            # reset RNN states if needed
            if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0

        t += 1

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()