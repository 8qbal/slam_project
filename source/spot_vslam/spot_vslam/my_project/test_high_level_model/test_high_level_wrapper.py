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
parser = argparse.ArgumentParser(description="Test high-level wrapper with frozen low-level policy.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--real-time", action="store_true", default=False)
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
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
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


def build_manual_high_level_command(num_envs, device, t):
    """
    手動測試高層 command
    前 100 step 前進
    接著 100 step 左轉
    接著 100 step 右轉
    最後停住
    """
    cmd = torch.zeros((num_envs, 3), device=device)

    if t < 100:
        # 前進
        cmd[:, 0] = 0.4
        cmd[:, 1] = 0.0
        cmd[:, 2] = 0.0
    elif t < 200:
        # 前進 + 左轉
        cmd[:, 0] = 0.25
        cmd[:, 1] = 0.0
        cmd[:, 2] = 0.2
    elif t < 300:
        # 前進 + 右轉
        cmd[:, 0] = 0.25
        cmd[:, 1] = 0.0
        cmd[:, 2] = -0.2
    else:
        # 停住
        cmd[:, 0] = 0.0
        cmd[:, 1] = 0.0
        cmd[:, 2] = 0.0

    return cmd


def main():
    task_name = args_cli.task.split(":")[-1]

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

    # -------------------------------------------------
    # Main loop
    # -------------------------------------------------
    t = 0
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # 1. 手動高層 command
            cmd = build_manual_high_level_command(env.unwrapped.num_envs, rl_device, t)

            # 2. 強制覆蓋 low-level env 的 base_velocity command
            set_base_velocity_command(env.unwrapped, cmd)

            # 3. low-level frozen agent 依 observation + command 產生 joint action
            obs_dict = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_dict, is_deterministic=True)

            # 4. step env
            obs, _, dones, infos = env.step(actions)

            # print debug
            if t % 20 == 0:
                print(
                    f"[{t:04d}] "
                    f"cmd(vx,wz)=({cmd[0,0].item():.2f}, {cmd[0,2].item():.2f})"
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