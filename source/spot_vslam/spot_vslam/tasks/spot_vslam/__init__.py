import gymnasium as gym

from . import agents

# Training
gym.register(
    id="Spot-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg_v0_rough:SpotRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
    },
)

# Validating
gym.register(
    id="Spot-Play-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg_v0_rough:SpotRoughEnvCfg_Play",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
    },
)

# Student Trainning Model
gym.register(
    id="Spot-Vslam-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simulate_vslam_env_cfg:SpotRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:vslam_cfg.yaml",
    },
)

# Student Test
gym.register(
    id="Spot-Vslam-Play-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.simulate_vslam_env_cfg:SpotRoughEnvCfg_Play",
        "rl_games_cfg_entry_point": f"{agents.__name__}:vslam_cfg.yaml",
    },
)

# Deploy Train
gym.register(
    id="Spot-Vslam-Deploy-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_model_run_on_vslam:SpotRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
    },
)

# Deploy Test
gym.register(
    id="Spot-Vslam-Deploy-Play-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_model_run_on_vslam:SpotRoughEnvCfg_Play",
        "rl_games_cfg_entry_point": f"{agents.__name__}:vslam_cfg.yaml",
    },
)

# Use depth Training
gym.register(
    id="Spot-Vslam-Depth-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.use_depth_training:SpotRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
    },
)

# Use depth Playing
gym.register(
    id="Spot-Vslam-Depth-Play-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.use_depth_training:SpotRoughEnvCfg_Play",
        "rl_games_cfg_entry_point": f"{agents.__name__}:vslam_cfg.yaml",
    },
)

# Use depth 2 Training
gym.register(
    id="Spot-Vslam-Depth-2-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.use_depth_training_2:SpotRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
    },
)

# Use depth 2 Testing
gym.register(
    id="Spot-Vslam-Depth-2-Play-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.use_depth_training_2:SpotRoughEnvCfg_Play",
        "rl_games_cfg_entry_point": f"{agents.__name__}:vslam_cfg.yaml",
    },
)

# Use depth 2-2 Training
gym.register(
    id="Spot-Vslam-Depth-3-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vslam_training:SpotRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
    },
)

# Use depth 2-2 Testing
gym.register(
    id="Spot-Vslam-Depth-3-Play-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vslam_training:SpotRoughEnvCfg_Play",
        "rl_games_cfg_entry_point": f"{agents.__name__}:vslam_cfg.yaml",
    },
)

# Use depth high level model Training
gym.register(
    id="Spot-Vslam-high-level-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.use_depth_training:SpotHighLevelTrainEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
    },
)

# Use depth high level model Training
gym.register(
    id="Spot-Vslam-high-level-Play-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.use_depth_training:SpotHighLevelTrainEnvCfg_Play",
        "rl_games_cfg_entry_point": f"{agents.__name__}:vslam_cfg.yaml",
    },
)

# High-level play/eval (rl_vslam_v0) in Simple_Warehouse/warehouse.usd = paper Map A, spawn (0, 0)
gym.register(
    id="Spot-Vslam-high-level-Warehouse-Play-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.use_depth_training:SpotHighLevelTrainEnvCfg_PlayWarehouse",
        "rl_games_cfg_entry_point": f"{agents.__name__}:vslam_cfg.yaml",
    },
)

# Paper policies (references/main.pdf Sec. 5): 353-dim low-level obs, goal-conditioned high level. See paper_cfg.py.
gym.register(
    id="Spot-Paper-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.paper_cfg:SpotPaperEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_paper_ppo_cfg.yaml",
    },
)

gym.register(
    id="Spot-Paper-Play-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.paper_cfg:SpotPaperEnvCfg_Play",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_paper_ppo_cfg.yaml",
    },
)

# High-level paper tasks: one per paper map (A = Room, B = Warehouse), fixed start and goal (paper_cfg.PAPER_MAPS)
gym.register(
    id="Spot-Paper-high-level-MapA-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.paper_cfg:SpotPaperHighLevelEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_paper_ppo_cfg.yaml",
    },
)

gym.register(
    id="Spot-Paper-high-level-MapA-Play-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.paper_cfg:SpotPaperHighLevelEnvCfg_Play",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_paper_ppo_cfg.yaml",
    },
)

gym.register(
    id="Spot-Paper-high-level-MapB-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.paper_cfg:SpotPaperHighLevelEnvCfg_MapB",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_paper_ppo_cfg.yaml",
    },
)

gym.register(
    id="Spot-Paper-high-level-MapB-Play-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.paper_cfg:SpotPaperHighLevelEnvCfg_MapB_Play",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_paper_ppo_cfg.yaml",
    },
)

# Other Testing
gym.register(
    id="Spot-test180-v0",
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.spot_vslam_test_cfg:SpotSlamTestEnvCfg"},
)

gym.register(
    id="Isaac-Velocity-Rough-Spot-v0",           
    entry_point="spot_vslam.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.spot_vslam_test_cfg:SpotSlamTestEnvCfg"},
)