# source/spot_vslam/spot_vslam/my_project/envs/spiral_config.py

import os


class SpiralTestConfig:
    # --- Model path ---
    # Low-level walking policy: from $SPOT_VSLAM_LOW_LEVEL_CKPT or the script's --checkpoint
    MODEL_PATH = os.environ.get("SPOT_VSLAM_LOW_LEVEL_CKPT", "")

    # --- Simulator ---
    ENABLE_RATE_LIMIT = True
    RATE_LIMIT_FPS = 30.0

    # --- Robot and camera ---
    START_HEIGHT = 0.55  # Spot standing height
    CAMERA_OFFSET_X = 0.4

    # --- Phase 1: mapping (spin in place) ---
    SPIN_LOOPS = 3

    # SPIN_SPEED: spin rate of the Phase 1 floating motion
    SPIN_SPEED = 0.4  # rad/s (floating mode)

    # SPIN_VEL_Z: limits the Phase 2 RL yaw-rate command
    SPIN_VEL_Z = 0.5  # rad/s (max yaw rate when walking)

    # --- Phase 2: square spiral ---
    SPIRAL_LENGTHS = [1.0, 1.0, 2.0, 2.0, 3.0]
    GOAL_TOLERANCE = 0.3  # goal tolerance when walking

    # --- Logging ---
    LOG_INTERVAL = 5

    # --- PPO agent config (network structure for loading the model) ---
    AGENT_CFG = {
        "seed": 42,
        "device": "cuda:0",
        "num_steps_per_env": 24,
        "max_iterations": 1500,
        "save_interval": 50,
        "empirical_normalization": False,
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 0.5,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.0025,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }
