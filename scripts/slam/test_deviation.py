import argparse
import torch
import os
import sys
import math
import numpy as np
from isaaclab.app import AppLauncher

# --- 1. 啟動器 ---
parser = argparse.ArgumentParser(description="VSLAM Accuracy Test")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=os.environ.get("SPOT_VSLAM_LOW_LEVEL_CKPT"),
    help="Low-level rsl_rl walking policy (.pt). Defaults to $SPOT_VSLAM_LOW_LEVEL_CKPT.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 2. 匯入庫 ---
import rclpy
from rclpy.node import Node
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner

# 匯入您的模組
from spot_vslam.tasks.spot_vslam.spot_vslam_test_cfg import SpotSlamTestEnvCfg
from spot_vslam.assets.spot_with_camera import SPOT_CFG
from spot_vslam.managers.ros2_manager import Ros2Manager
from spot_vslam.managers.orb_slam_subscriber_manager import OrbSlamSubscriberManager

# ==========================================
# 設定區
# ==========================================
MODEL_PATH = args_cli.checkpoint
if not MODEL_PATH or not os.path.isfile(MODEL_PATH):
    print(f"[ERROR] Low-level checkpoint not found: {MODEL_PATH!r}. Pass --checkpoint or set SPOT_VSLAM_LOW_LEVEL_CKPT.")
    sys.exit(1)
TEST_DURATION_SEC = 20.0  # 測試總秒數
CMD_VEL_X = 0.4           # 前進速度 (m/s)
CMD_YAW   = 0.0           # 轉向速度 (rad/s)

# ==========================================
# 簡單的 Wrapper (與之前相同)
# ==========================================
class RslRlWrapper:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        obs_dict, _ = self.env.reset()
        self.num_obs = obs_dict["policy"].shape[1]
    
    def get_observations(self):
        obs_dict = self.env.observation_manager.compute()
        return obs_dict["policy"], {"observations": obs_dict}

    def reset(self):
        obs_dict, extras = self.env.reset()
        extras["observations"] = obs_dict
        return obs_dict["policy"], extras

    def step(self, actions):
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        extras["observations"] = obs_dict
        return obs_dict["policy"], rew, terminated | truncated, extras

def main():
    rclpy.init()
    
    # 1. 建立環境
    env_cfg = SpotSlamTestEnvCfg()
    env_cfg.scene.robot = SPOT_CFG.copy()
    env_cfg.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
    
    # 確保地板有摩擦力，不然機器人會滑
    # env_cfg.scene.terrain.terrain_type = "plane" 
    
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 2. 啟動 Manager
    ros2_mgr = Ros2Manager(env_cfg.ros2, base_env)
    slam_sub_mgr = OrbSlamSubscriberManager(env_cfg.slam_subscriber, base_env)
    
    rl_env = RslRlWrapper(base_env, device="cuda:0")

    # 3. 載入 Policy
    runner = OnPolicyRunner(rl_env, {"seed": 42, "device": "cuda:0", "policy": {"class_name": "ActorCritic", "actor_hidden_dims": [512, 256, 128], "critic_hidden_dims": [512, 256, 128], "activation": "elu"}, "algorithm": {"class_name": "PPO"}}, log_dir=os.path.dirname(MODEL_PATH), device="cuda:0")
    runner.load(MODEL_PATH)
    policy = runner.get_inference_policy(device="cuda:0")

    # 4. 初始化
    obs, _ = rl_env.reset()
    ros2_mgr.reset(env_ids=range(base_env.num_envs))
    
    # 用來記錄軌跡誤差
    total_steps = int(TEST_DURATION_SEC / base_env.step_dt)
    trajectory_log = []

    print(f"[START] 開始定速測試: V_x={CMD_VEL_X} m/s, 時間={TEST_DURATION_SEC}s")

    for i in range(total_steps):
        # ---------------------------------------------------
        # [關鍵]：手動覆寫指令 (Hardcoded Commands)
        # ---------------------------------------------------
        # 這裡不依賴 Nav2，直接產生一個固定的速度向量
        # 格式: [lin_vel_x, lin_vel_y, ang_vel_z]
        target_vel = torch.tensor([[CMD_VEL_X, 0.0, CMD_YAW]], device=base_env.device)
        
        # 強制注入環境
        if base_env.command_manager:
            base_env.command_manager.set_command("base_velocity", target_vel)

        # ---------------------------------------------------
        # RL 推論與執行
        # ---------------------------------------------------
        with torch.inference_mode():
            actions = policy(obs)
        
        obs, _, dones, extras = rl_env.step(actions)
        
        # 更新 Manager
        dt = base_env.step_dt
        ros2_mgr.update(dt)
        slam_sub_mgr.update(dt)

        # ---------------------------------------------------
        # 收集數據 (計算誤差)
        # ---------------------------------------------------
        if i % 10 == 0: # 每 10 步記錄一次
            # 1. 真值 (Ground Truth)
            gt_pos = base_env.scene["robot"].data.root_pos_w.torch[0, :3].cpu().numpy()
            
            # 2. SLAM 估測值 (從 Manager 讀取)
            # 注意：這裡假設您的 slam_sub_mgr 有儲存最新的 slam_pose
            # 如果沒有，您可能需要去 manager 裡面加一個 self.current_slam_pose
            slam_error = -1.0
            if "observations" in extras and "slam_info" in extras["observations"]:
                 # 這是我們之前在 cfg 裡寫好的 observation term
                 # 如果您想直接拿座標，可能需要修改 OrbSlamSubscriberManager 讓它暴露座標變數
                 pass

            # 這裡示範如何計算「回到原點」的距離 (假設測試是走直線)
            # 或者您可以單純 print 出來讓老師看
            print(f"Step {i} | GT Pos: ({gt_pos[0]:.2f}, {gt_pos[1]:.2f})")
            
            trajectory_log.append(gt_pos)

    print("[DONE] 測試結束")
    ros2_mgr.close()
    slam_sub_mgr.close()
    base_env.close()
    simulation_app.close()

if __name__ == "__main__":
    import argparse
    main()