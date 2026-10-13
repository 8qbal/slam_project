"""Load a frozen rl_games low-level policy and its env, for the SB3 high-level scripts
(``scripts/high_level/train_high_level_ppo.py``, ``load_high_level.py``). Import only after the app is launched."""

import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

from spot_vslam.slam.ros2_manager import Ros2Manager


def build_low_level(task: str, checkpoint: str, num_envs: int, seed: int | None = None, ros2: bool = True):
    """Make ``task``'s env wrapped for rl_games and a player restored from ``checkpoint``; returns (env, agent).

    With ``ros2`` (default), a task cfg that sets ``ros2`` gets its ``Ros2Manager`` attached to the env.
    """
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=num_envs)
    if seed is not None:
        env_cfg.seed = seed
    agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    print("[INFO] Loading low-level checkpoint:", checkpoint)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))

    env = gym.make(task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)

    agent: BasePlayer = runner.create_player()
    agent.restore(checkpoint)
    agent.reset()

    if ros2 and getattr(env.unwrapped.cfg, "ros2", None) is not None:
        env.unwrapped.ros2_manager = Ros2Manager(env.unwrapped.cfg.ros2, env.unwrapped)

    print("[INFO] low-level num_envs =", env.unwrapped.num_envs)
    return env, agent
