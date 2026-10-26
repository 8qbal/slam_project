import argparse
import os

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train high-level PPO with one Isaac env and internal num_envs.")
parser.add_argument("--task", type=str, required=True, help="High-level env task")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--low_level_task", type=str, required=True, help="Low-level locomotion task for rl-games cfg")
parser.add_argument("--total_timesteps", type=int, default=500_000, help="Table 6: 500,000")
parser.add_argument(
    "--paper",
    action="store_true",
    help="Paper high-level policy (Tables 5-6): 9-dim goal observation, goal reward, [256, 128] ELU MLP",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry

import spot_vslam.tasks  # noqa: F401
from spot_vslam.assets import PAPER_MAPS
from spot_vslam.legacy.high_level.low_level_runner import build_low_level as build_low_level_policy
from spot_vslam.legacy.high_level.paper_vec_env import PaperHighLevelEnvCfg, PaperHighLevelIsaacVecEnv


class HighLevelTensorboardCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        # The high-level command
        cmd_vx_vals = [info["cmd_vx"] for info in infos if "cmd_vx" in info]
        cmd_wz_vals = [info["cmd_wz"] for info in infos if "cmd_wz" in info]

        if cmd_vx_vals:
            self.logger.record("env/cmd_vx", float(np.mean(cmd_vx_vals)))
        if cmd_wz_vals:
            self.logger.record("env/cmd_wz", float(np.mean(cmd_wz_vals)))

        # Also log some common env metrics, handy when reading the plots
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
            "dist_to_goal",
            "goal_reached",
            "fell",
        ]

        for key in extra_keys:
            vals = [info[key] for info in infos if key in info]
            if vals:
                self.logger.record(f"env/{key}", float(np.mean(vals)))

        return True


def build_low_level():
    if args_cli.checkpoint is None:
        agent_cfg = load_cfg_from_registry(args_cli.low_level_task, "rl_games_cfg_entry_point")
        log_root = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = args_cli.checkpoint
    return build_low_level_policy(args_cli.low_level_task, resume_path, args_cli.num_envs)


def main():
    if not args_cli.paper:
        raise SystemExit(
            "rl_vslam_v0 (the non-paper high level) was removed: add --paper (old code: git tag baseline/rl-vslam-v0)"
        )
    low_env, low_agent = build_low_level()

    paper_map = PAPER_MAPS[low_env.unwrapped.cfg.paper_map]
    hl_cfg = PaperHighLevelEnvCfg(goal_w=paper_map.goal, goal_radius=paper_map.goal_radius, use_orb_pose=False)
    print(f"[INFO] Paper map {low_env.unwrapped.cfg.paper_map}: start {paper_map.start} -> goal {paper_map.goal}")
    vec_env = PaperHighLevelIsaacVecEnv(low_env, low_agent, hl_cfg)
    # Sec. 5.2: same MLP type as the low-level policy (ELU, orthogonal init) with hidden layers [256, 128]
    policy_kwargs = dict(net_arch=dict(pi=[256, 128], vf=[256, 128]), activation_fn=nn.ELU)
    out_suffix = "_paper"
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
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"./output/tensorboard/high_level{out_suffix}/",
    )

    ckpt_dir = f"./output/checkpoints/high_level{out_suffix}"
    os.makedirs(ckpt_dir, exist_ok=True)

    tensorboard_callback = HighLevelTensorboardCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=4_000,
        save_path=ckpt_dir,
        name_prefix="high_level_policy",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    callback = CallbackList([tensorboard_callback, checkpoint_callback])

    model.learn(
        total_timesteps=args_cli.total_timesteps,
        callback=callback,
        log_interval=1,
    )

    model.save(os.path.join(ckpt_dir, "high_level_policy"))
    print("Training finished")

    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
