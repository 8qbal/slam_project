"""Spot high-level navigation tasks on the paper maps."""

import gymnasium as gym

from . import agents

for _map, _suffix in (("A", ""), ("B", "_MAP_B")):
    for _variant, _cfg in (
        ("", ""),
        ("-Play", "_PLAY"),  # ORB-SLAM3 RGB-D
        ("-Stereo-Play", "_STEREO_PLAY"),  # ORB-SLAM3 stereo
        ("-GT-Play", "_GT_PLAY"),
        ("-cuVSLAM-Play", "_CUVSLAM_PLAY"),  # cuVSLAM stereo
        ("-cuVSLAM-Quality-Play", "_CUVSLAM_QUALITY_PLAY"),  # cuVSLAM + quality inputs (condition C, 12-dim)
    ):
        gym.register(
            id=f"Spot-Nav-Map{_map}{_variant}-v0",
            entry_point="spot_vslam.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:SpotNavigationEnvCfg{_suffix}{_cfg}",
                "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
            },
        )

gym.register(
    id="Spot-Nav-Maze-v0",  # random mazes, random start and goal; evaluate on the paper maps
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:SpotNavigationMazeEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Spot-Nav-Hall-v0",  # random mazes + halls like Maps A / B (wall rows, shelves); evaluate on the paper maps
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:SpotNavigationHallEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Spot-Nav-HallRacks-v0",  # Hall + open racks in the doorways (walk under them, like Map B's shelves)
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:SpotNavigationHallRacksEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Spot-Nav-HallRacks-cuVSLAM-v0",  # HallRacks with the cuVSLAM pose in the observation (condition B)
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:SpotNavigationHallRacksCuVslamEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Spot-Nav-HallRacks-cuVSLAM-Quality-v0",  # condition C: cuVSLAM pose + quality inputs (12-dim observation)
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:SpotNavigationHallRacksCuVslamQualityEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
