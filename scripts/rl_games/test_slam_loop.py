# /scripts/test_slam_loop.py
import gymnasium as gym
import torch
import rclpy
import time

# 匯入 Isaac Lab App Launcher
from isaaclab.app import AppLauncher

# 匯入你的環境
# (確保這些 .py 檔案的路徑相對於你執行腳本的位置是正確的)
from spot_vslam.spot_vslam.my_project.envs.rl_env import ManagerBasedRLEnv
from spot_vslam.spot_vslam.my_project.cfg.spot_vslam_test_cfg import SpotSlamTestEnvCfg

# !! (重要) !! 匯入你的機器人設定檔
# 我們需要它來填入 cfg.scene.robot
# 根據你的專案結構，你可能需要從 isaaclab_tasks 匯入
# 例如: from isaaclab_tasks.locomotion.spot.spot_ppo_cfg import SpotFlatCfg
#
# **請根據你的機器人修改這一行**
from isaaclab_tasks.locomotion.spot.spot_ppo_cfg import SpotFlatCfg 


def main():
    """主測試函式"""
    
    # 1. 初始化 ROS 2
    rclpy.init()

    # 2. 啟動 Isaac Sim
    app_launcher = AppLauncher(headless=False) # 設為 True 則為無頭(背景)模式
    simulation_app = app_launcher.app

    # 3. 載入我們的「乾淨」設定
    print("正在載入環境設定...")
    env_cfg = SpotSlamTestEnvCfg()

    # 4. !! (重要) !! 填入 MISSING 的機器人設定
    # 我們在 SpotSlamTestEnvCfg 中將 robot 設為 MISSING
    # 現在我們手動填入它
    env_cfg.scene.robot = SpotFlatCfg() # <-- **如果你的機器人不是 SpotFlatCfg，請在這裡修改**
    
    # 5. 建立 Isaac Lab 環境
    print("正在建立環境 (num_envs=1)...")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print("環境建立完畢。")

    # 獲取動作維度 (lin_x, lin_y, ang_z)
    # 我們在 SlamTestActionsCfg 中定義了 "base_vel"
    num_actions = env.action_space["base_vel"].shape[1] 
    num_envs = env.num_envs
    device = env.device
    
    # 重置環境
    obs_dict, info_dict = env.reset()
    print("環境重置完畢。")
    print(f"ROS Topics 應在 {env_cfg.ros2.topic_prefix}0/... 上發布")
    print("ORB-SLAM3 應訂閱：")
    print(f"  {env_cfg.ros2.topic_prefix}0/camera/image_raw")
    print(f"  {env_cfg.ros2.topic_prefix}0/camera/camera_info")
    print("\n按 Enter 鍵開始執行動作序列...")
    input() # 等待你啟動 ORB-SLAM3

    # 測試動作序列 (持續 1000 步)
    # (速度 x, 速度 y, 角速度 z)
    action_dict = {
        "base_vel": torch.zeros((num_envs, num_actions), device=device)
    }
    
    num_steps = 1000
    print(f"開始模擬 {num_steps} 步 (前進 -> 後退 -> 左轉 -> 右轉)")

    for i in range(num_steps):
        
        # 產生動作
        if 0 < i <= 200:
            # 前進 (lin_x = 0.5)
            action_dict["base_vel"][:, 0] = 1.0 # 乘以 scale (0.5) -> 0.5 m/s
        elif 200 < i <= 400:
            # 後退 (lin_x = -0.5)
            action_dict["base_vel"][:, 0] = -1.0 # 乘以 scale (0.5) -> -0.5 m/s
        elif 400 < i <= 600:
            # 左轉 (ang_z = 0.3)
            action_dict["base_vel"][:, 2] = 1.0 # 乘以 scale (0.3) -> 0.3 rad/s
        elif 600 < i <= 800:
            # 右轉 (ang_z = -0.3)
            action_dict["base_vel"][:, 2] = -1.0 # 乘以 scale (0.3) -> -0.3 rad/s
        else:
            # 停止
            action_dict["base_vel"][:, :] = 0.0
        
        # 執行一步
        obs, rewards, dones, info = env.step(action_dict)
        
        if i % 100 == 0:
            print(f"Step {i}")

    print("模擬結束。")

    # 7. 關閉
    env.close()
    rclpy.shutdown()
    simulation_app.close() # 關閉 Isaac Sim

if __name__ == "__main__":
    main()
