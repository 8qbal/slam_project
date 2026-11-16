"""Spot high-level navigation tasks on the paper maps."""

import os

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

for _variant, _cfg in (
    ("-GT-Play", "GTPlay"),
    ("-cuVSLAM-Play", "CuVslamPlay"),
    ("-Play", "Play"),  # ORB-SLAM3 RGB-D
    ("-Stereo-Play", "StereoPlay"),  # ORB-SLAM3 stereo
):
    gym.register(
        id=f"Spot-Nav-SmallMaze{_variant}-v0",  # one small bright maze, fixed layout, goal at the far corner
        entry_point="spot_vslam.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:SpotNavigationSmallMaze{_cfg}EnvCfg",
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

for _id, _cfg in (
    ("Spot-Nav-HallRacks-Depth-v0", "SpotNavigationHallRacksDepthEnvCfg"),  # GT pose + 5 depth sectors (14-dim)
    ("Spot-Nav-WarehouseMaze-Depth-v0", "SpotNavigationWarehouseMazeDepthEnvCfg"),  # path reward, warehouse + maze
    ("Spot-Nav-WarehouseMaze-cuVSLAM-Depth-v0", "SpotNavigationWarehouseMazeCuVslamDepthEnvCfg"),
    ("Spot-Nav-WarehouseMaze-cuVSLAM-QualityDepth-v0", "SpotNavigationWarehouseMazeCuVslamQualityDepthEnvCfg"),
    ("Spot-Nav-HallRacks-cuVSLAM-Depth-v0", "SpotNavigationHallRacksCuVslamDepthEnvCfg"),  # cuVSLAM + depth (14)
    ("Spot-Nav-HallRacks-cuVSLAM-QualityDepth-v0", "SpotNavigationHallRacksCuVslamQualityDepthEnvCfg"),  # (17)
    ("Spot-Nav-SmallMaze-GT-Depth-Play-v0", "SpotNavigationSmallMazeGTDepthPlayEnvCfg"),
    ("Spot-Nav-SmallMaze-cuVSLAM-Depth-Play-v0", "SpotNavigationSmallMazeCuVslamDepthPlayEnvCfg"),
    ("Spot-Nav-SmallMaze-cuVSLAM-QualityDepth-Play-v0", "SpotNavigationSmallMazeCuVslamQualityDepthPlayEnvCfg"),
    # warehouse: rack rows and aisles, layout from SPOT_VSLAM_WAREHOUSE_SEED (evaluation world)
    ("Spot-Nav-Warehouse-GT-Play-v0", "SpotNavigationWarehouseGTPlayEnvCfg"),
    ("Spot-Nav-Warehouse-cuVSLAM-Play-v0", "SpotNavigationWarehouseCuVslamPlayEnvCfg"),
    ("Spot-Nav-Warehouse-GT-Depth-Play-v0", "SpotNavigationWarehouseGTDepthPlayEnvCfg"),
    ("Spot-Nav-Warehouse-cuVSLAM-Depth-Play-v0", "SpotNavigationWarehouseCuVslamDepthPlayEnvCfg"),
    ("Spot-Nav-Warehouse-cuVSLAM-QualityDepth-Play-v0", "SpotNavigationWarehouseCuVslamQualityDepthPlayEnvCfg"),
):
    gym.register(
        id=_id,
        entry_point="spot_vslam.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.navigation_env_cfg:{_cfg}",
            # the training tasks use the PPO file with the smaller initial std and bounded mean action
            "rl_games_cfg_entry_point": f"{agents.__name__}:"
            + (
                "rl_games_ppo_pathreward_lstm_cfg.yaml"  # recurrent policy; training AND evaluation need the same value
                if os.environ.get("SPOT_VSLAM_HL_LSTM") == "1"
                else "rl_games_ppo_pathreward_cfg.yaml" if "WarehouseMaze" in _id else "rl_games_ppo_cfg.yaml"
            ),
        },
    )
