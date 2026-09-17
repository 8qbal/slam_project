# /scripts/rl_games/test_rotate_180.py

# --- ▼▼▼ 在這裡新增 ▼▼▼ ---
# 這是解決 "ModuleNotFoundError: No module named 'isaacsim.core'" 的關鍵
import sys
import os

# 1. 假設你的 IsaacLab 目錄固定在 /home/bernie/Isaac_lab/IsaacLab
#    (如果不是，請修改這個路徑)
ISAACLAB_DIR = "/home/bernie/Isaac_lab/IsaacLab"

# 2. 定義 Isaac Sim 函式庫的關鍵路徑
ISAAC_SIM_BIN_PATH = os.path.join(ISAACLAB_DIR, "isaac_sim", "bin")

# 3. 將 Isaac Sim 函式庫手動添加到 Python 的 sys.path
if ISAAC_SIM_BIN_PATH not in sys.path:
    print(f"[INFO] 手動添加 Isaac Sim 路徑: {ISAAC_SIM_BIN_PATH}")
    sys.path.append(ISAAC_SIM_BIN_PATH)
else:
    print(f"[INFO] Isaac Sim 路徑已在 sys.path 中。")
# --- ▲▲▲ 新增結束 ▲▲▲ ---
# /scripts/rl_games/test_rotate_180_ground_truth.py
# (閉迴路控制測試 - 使用 Isaac Lab 真實姿態)

import gymnasium as gym
import torch
import rclpy
import time
import math

from isaaclab.app import AppLauncher
from spot_vslam.spot_vslam.my_project.envs.rl_env import ManagerBasedRLEnv
from spot_vslam.spot_vslam.my_project.cfg.spot_vslam_test_cfg import SpotSlamTestEnvCfg
from spot_vslam.spot_vslam.my_project.cfg.spot_robot_cfg import SPOT_CFG # 匯入你真實的機器人

def quat_to_yaw(q_xyzw: torch.Tensor) -> float:
    """將 [qx, qy, qz, qw] 四元數轉換為偏航角 (yaw)"""
    # 傳入的 quat 是 (N, 4) tensor，我們只取第一個 env
    q = q_xyzw[0] 
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)).item()

def main():
    """主測試函式 - 180 度旋轉 (使用 Ground Truth)"""
    
    # --- 1. 初始化 ---
    rclpy.init()
    app_launcher = AppLauncher(headless=False)
    simulation_app = app_launcher.app

    # --- 2. 載入「乾淨」的環境設定 ---
    print("正在載入環境設定 (spot_vslam_test_cfg.py)...")
    env_cfg = SpotSlamTestEnvCfg()
    env_cfg.scene.robot = SPOT_CFG.copy() # 注入機器人
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    obs_dict, info_dict = env.reset()
    print("環境重置完畢。")
    print(f"ROS Topics 應在 {env_cfg.ros2.topic_prefix}0/... 上發布")
    print(f"[重要] ORB-SLAM3 必須訂閱: /spot_test0/camera/image_raw")
    print(f"[重要] ORB-SLAM3 必須發布 State 到: /spot_test0/orbslam/tracking_state")
    print(f"[重要] ORB-SLAM3 必須發布 LocalPC 到: /spot_test0/orbslam/local_pc")
    
    # --- 3. 獲取初始姿態 (來自 Isaac Lab Ground Truth) ---
    print("\n正在獲取 Isaac Lab 初始姿態...")
    action_dict = {"base_vel": torch.zeros((1, 3), device=env.device)}
    
    # 執行一步空動作，以確保 scene.robot.data 被填充
    env.step(action_dict) 
    time.sleep(0.1)
    
    # 從 Isaac Lab 場景中讀取「真實」姿態
    # (注意：我們只取 env 0)
    initial_pose_quat = env.scene.robot.data.root_state_quat_xyzw # (N, 4)
    initial_yaw = quat_to_yaw(initial_pose_quat)
    print(f"成功獲取初始姿態！ 初始航向 (Yaw): {math.degrees(initial_yaw):.2f} 度")

    # --- 4. 設定 180 度旋轉目標 ---
    KP_ANGULAR = 1.5
    GOAL_TOLERANCE_RAD = 0.05

    goal_yaw = initial_yaw + math.pi
    while goal_yaw > math.pi: goal_yaw -= 2 * math.pi
    while goal_yaw < -math.pi: goal_yaw += 2 * math.pi
    print(f"設定目標航向為: {math.degrees(goal_yaw):.2f} 度")
    
    # --- 5. 閉迴路控制迴圈 ---
    print("按 Enter 鍵開始執行 180 度旋轉測試...")
    input()
    
    max_steps = 500
    for i in range(max_steps):
        
        # =========================================================
        # === 這是「證明」的核心 ===
        
        # 1. 從 Isaac Lab 獲取「真實當前姿態」
        current_pose_quat = env.scene.robot.data.root_state_quat_xyzw
        current_yaw = quat_to_yaw(current_pose_quat)
        
        # 2. 監控 SLAM 狀態 (從 Subscriber 獲取)
        slam_state_val = env.slam_subscriber_manager.state_buffer[0].item()
        slam_pc_count = env.slam_subscriber_manager.local_pc_count[0].item()
        
        state_map = {0: "INIT/UNKNOWN", 1: "TRACKING", 2: "LOST"}
        slam_state_str = state_map.get(slam_state_val, "ERROR")
        
        # === 控制邏輯 (P-Controller) ===
        
        # 3. 計算「誤差」 (目標姿態 vs 真實當前姿態)
        error_yaw = goal_yaw - current_yaw
        while error_yaw > math.pi: error_yaw -= 2 * math.pi
        while error_yaw < -math.pi: error_yaw += 2 * math.pi
        
        # 4. 檢查是否已到達目標
        if abs(error_yaw) < GOAL_TOLERANCE_RAD:
            print(f"\n[成功] 已到達目標航向！ (誤差: {math.degrees(error_yaw):.2f} 度)")
            break
        
        # 5. 根據「誤差」計算「動作」
        action_z = KP_ANGULAR * error_yaw
        action_z = torch.clamp(torch.tensor(action_z), -1.0, 1.0).item()
        
        action_dict["base_vel"][:, 0] = 0.0 # 不前進
        action_dict["base_vel"][:, 1] = 0.0 # 不側移
        action_dict["base_vel"][:, 2] = action_z # 執行旋轉
        
        # =========================================================

        # 6. 執行這一步 (這會更新 env.scene.robot.data 並觸發 ROS 訊息)
        obs, rewards, dones, info = env.step(action_dict)
        
        if i % 10 == 0:
            print(f"  Step {i} | "
                  f"目標: {math.degrees(goal_yaw):.1f}° | "
                  f"GT姿態: {math.degrees(current_yaw):.1f}° | "
                  f"動作 (Vz): {action_z:.2f} | "
                  f"SLAM狀態: {slam_state_str} | "  # <--- 監控 SLAM 狀態
                  f"SLAM點數: {slam_pc_count}")      # <--- 監控特徵點
            
        time.sleep(env.step_dt) # 模擬即時
    
    if i == max_steps - 1:
        print(f"\n[失敗] 超時 {max_steps} 步，未能到達目標。")

    # --- 6. 關閉 ---
    print("測試迴圈結束。正在關閉...")
    action_dict["base_vel"][:, :] = 0.0
    env.step(action_dict)
    
    env.close()
    rclpy.shutdown()
    simulation_app.close()

if __name__ == "__main__":
    main()