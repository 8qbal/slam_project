import argparse
import math
import os
import gymnasium as gym

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build high-level RL env on top of frozen low-level policy.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
import spot_vslam.tasks  # noqa: F401

from spot_vslam.my_project.manager.ros2_manager import Ros2Manager
from high_level_rl_env import HighLevelRLEnv, HighLevelEnvCfg


def build_low_level_env_and_agent():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)

    print(f"[INFO] Loading low-level checkpoint from: {resume_path}")

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)

    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    if getattr(env.unwrapped.cfg, "ros2", None) is not None:
        env.unwrapped.ros2_manager = Ros2Manager(env.unwrapped.cfg.ros2, env.unwrapped)

    return env, agent


def main():
    low_env, low_agent = build_low_level_env_and_agent()

    hl_cfg = HighLevelEnvCfg(
        hl_decimation=5,
        cmd_smoothing_alpha=0.25,
        vx_min=0.0,
        vx_max=0.5,
        wz_min=-0.25,
        wz_max=0.25,
        max_episode_hl_steps=400,
    )

    hl_env = HighLevelRLEnv(low_env, low_agent, hl_cfg)

    obs, info = hl_env.reset()
    print("[INFO] High-level env ready.")
    print("[INFO] obs shape:", obs.shape)
    print("[INFO] initial obs:", obs)

    # 先跑幾步 random action 驗證
    for i in range(20):
        action = hl_env.action_space.sample()
        obs, reward, terminated, truncated, info = hl_env.step(action)
        print(
            f"[{i:03d}] reward={reward:.3f}, "
            f"terminated={terminated}, truncated={truncated}, "
            f"cmd=({info['cmd_vx']:.3f}, {info['cmd_wz']:.3f}), "
            f"progress={info['progress']:.3f}, "
            f"tracking_ok={info['tracking_ok']:.1f}"
        )
        if terminated or truncated:
            obs, info = hl_env.reset()

    hl_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()