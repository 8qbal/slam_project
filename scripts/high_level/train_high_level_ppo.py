import argparse
import os
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train high-level PPO with one Isaac env and internal num_envs.")
parser.add_argument("--task", type=str, required=True, help="High-level env task")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--low_level_task", type=str, required=True, help="Low-level locomotion task for rl-games cfg")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import isaaclab_tasks  # noqa: F401
import spot_vslam.tasks  # noqa: F401

from spot_vslam.managers.ros2_manager import Ros2Manager
from spot_vslam.high_level.high_level_vec_env import (
    HighLevelEnvCfg,
    HighLevelIsaacVecEnv,
)


class HighLevelTensorboardCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        # 你想看的 high-level command
        cmd_vx_vals = [info["cmd_vx"] for info in infos if "cmd_vx" in info]
        cmd_wz_vals = [info["cmd_wz"] for info in infos if "cmd_wz" in info]

        if cmd_vx_vals:
            self.logger.record("env/cmd_vx", float(np.mean(cmd_vx_vals)))
        if cmd_wz_vals:
            self.logger.record("env/cmd_wz", float(np.mean(cmd_wz_vals)))

        # 順手把一些常用 env 指標也記下來，之後看圖會很方便
        extra_keys = [
            "progress",
            "tracking_ok",
            "tracking_bad",
            "stuck",
            "body_contact",
            "wz_abs",
            "reward_step_mean",
            "reward_progress_term",
            "reward_tracking_bonus_term",
            "reward_tracking_lost_term",
            "reward_stuck_term",
            "reward_body_contact_term",
            "reward_turn_term",
        ]

        for key in extra_keys:
            vals = [info[key] for info in infos if key in info]
            if vals:
                self.logger.record(f"env/{key}", float(np.mean(vals)))

        return True


def build_low_level():
    env_cfg = parse_env_cfg(
        args_cli.low_level_task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
    )

    agent_cfg = load_cfg_from_registry(args_cli.low_level_task, "rl_games_cfg_entry_point")
    log_root = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])

    if args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = args_cli.checkpoint

    print("[INFO] Loading low-level checkpoint:", resume_path)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))

    env = gym.make(args_cli.low_level_task, cfg=env_cfg)
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

    print("[INFO] low-level num_envs =", env.unwrapped.num_envs)
    return env, agent


def main():
    low_env, low_agent = build_low_level()

    hl_cfg = HighLevelEnvCfg(
        hl_decimation=2,
        cmd_smoothing_alpha=0.25,
        vx_min=0.0,
        vx_max=1.0,
        wz_min=-0.4,
        wz_max=0.4,
        max_episode_hl_steps=400,
    )

    vec_env = HighLevelIsaacVecEnv(low_env, low_agent, hl_cfg)
    vec_env = VecMonitor(vec_env)

    print("Observation space:", vec_env.observation_space)
    print("Action space:", vec_env.action_space)
    print("SB3 vec env num_envs:", vec_env.num_envs)

    model = PPO(
        "MlpPolicy",
        vec_env,
        device="cpu",
        learning_rate=3e-4,
        n_steps=256,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log="./tensorboard_high_level/",
    )

    os.makedirs("./checkpoints_high_level", exist_ok=True)

    tensorboard_callback = HighLevelTensorboardCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=4_000,
        save_path="./checkpoints_high_level",
        name_prefix="high_level_policy",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    callback = CallbackList([tensorboard_callback, checkpoint_callback])

    model.learn(
        total_timesteps=500_000,
        callback=callback,
        log_interval=1,
    )

    model.save("high_level_policy")
    print("Training finished")

    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()