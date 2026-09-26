# source/spot_vslam/spot_vslam/my_project/envs/spiral_config.py

import os

class SpiralTestConfig:
    # --- 模型路徑 ---
    # 低階走路策略 (.pt)：由環境變數 SPOT_VSLAM_LOW_LEVEL_CKPT 或腳本的 --checkpoint 指定
    MODEL_PATH = os.environ.get("SPOT_VSLAM_LOW_LEVEL_CKPT", "")
    
    # --- 模擬器設定 ---
    ENABLE_RATE_LIMIT = True
    RATE_LIMIT_FPS = 30.0

    # --- 機器人與相機 ---
    START_HEIGHT = 0.55         # Spot 站立高度
    CAMERA_OFFSET_X = 0.4
    
    # --- 階段 1: 建圖 (原地轉圈) ---
    SPIN_LOOPS = 3
    
    # [關鍵修正] 補回 SPIN_SPEED (用於 Phase 1 漂浮計算)
    SPIN_SPEED = 0.4            # rad/s (漂浮模式的轉速)
    
    # [保留] SPIN_VEL_Z (用於 Phase 2 RL 指令限制)
    SPIN_VEL_Z = 0.5            # rad/s (走路模式的最大角速度)
    
    # --- 階段 2: 方形螺旋 ---
    SPIRAL_LENGTHS = [1.0, 1.0, 2.0, 2.0, 3.0]
    GOAL_TOLERANCE = 0.3        # 走路誤差容許值
    
    # --- 數據記錄 ---
    LOG_INTERVAL = 5
    
    # --- PPO Agent Config (載入模型結構用) ---
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
        }
    }