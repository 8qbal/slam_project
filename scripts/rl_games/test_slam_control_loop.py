# /scripts/test_slam_control_loop.py
# (閉迴路控制測試)

import gymnasium as gym
import torch
import rclpy
import time
import math

# 匯入 Isaac Lab App Launcher
from isaaclab.app import AppLauncher

# 匯入你的環境
from spot_vslam.spot_vslam.my_project.envs.rl_env import ManagerBasedRLEnv
from spot_vslam.spot_vslam.my_project.cfg.spot_vslam_test_cfg import SpotSlamTestEnvCfg

# !! (重要) !! 匯入你的機器人設定檔
from spot_vslam.spot_vslam.my_project.spot_new.spot_with_camera import SPOT_CFG # <-- **如果你的機器人不是這個，請修改**

def quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """將 [qx, qy, qz, qw] 四元數轉換為偏航角 (yaw)"""
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

def main():
    """主測試函式 - 閉迴路控制"""
    
    # --- 1. 初始化 ---
    rclpy.init()
    app_launcher = AppLauncher(headless=False)
    simulation_app = app_launcher.app

    # --- 2. 載入「乾淨」的環境設定 ---
    print("正在載入環境設定 (spot_vslam_test_cfg.py)...")
    env_cfg = SpotSlamTestEnvCfg()

    # 填入 MISSING 的機器人設定
    env_cfg.scene.robot = SPOT_CFG.copy() # <-- **在這裡修改你的機器人**
    
    # 建立環境 (num_envs=1)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 重置環境
    obs_dict, info_dict = env.reset()
    print("環境重置完畢。")
    print(f"ROS Topics 應在 {env_cfg.ros2.topic_prefix}0/... 上發布")
    print("ORB-SLAM3 應訂閱：")
    print(f"  {env_cfg.ros2.topic_prefix}0/camera/image_raw")
    print(f"  {env_cfg.ros2.topic_prefix}0/camera/camera_info")
    print(f"\n[重要] ORB-SLAM3 必須發布 Pose 到: {env_cfg.slam_subscriber.pose_topic}")
    print("\n按 Enter 鍵開始「閉迴路」控制測試...")
    input()

    # --- 3. 設定控制器參數和目標 ---
    KP_LINEAR = 1.0  # P-控制器增益 (直線速度)
    KP_ANGULAR = 1.5 # P-控制器增益 (角速度)
    GOAL_TOLERANCE_LINEAR = 0.1 # 10 公分容忍誤差
    GOAL_TOLERANCE_ANGULAR = 0.1 # 0.1 弧度容忍誤差 (~6 度)

    # 定義一系列任務 (目標姿態)
    goals = [
        {"x": 2.0, "y": 0.0, "yaw": 0.0, "desc": "移動到 (2, 0)"},
        {"x": 2.0, "y": 0.0, "yaw": 1.57, "desc": "轉向 90 度"},
        {"x": 2.0, "y": 2.0, "yaw": 1.57, "desc": "移動到 (2, 2)"},
        {"x": 2.0, "y": 2.0, "yaw": 0.0, "desc": "轉向 0 度"},
        {"x": 0.0, "y": 0.0, "yaw": 0.0, "desc": "回到原點 (0, 0)"}
    ]
    current_goal_index = 0

    action_dict = {
        "base_vel": torch.zeros((env.num_envs, env.action_space["base_vel"].shape[1]), device=env.device)
    }
    
    # --- 4. 閉迴路控制迴圈 ---
    num_steps = 3000 # 總共 3000 步
    for i in range(num_steps):
        
        if current_goal_index >= len(goals):
            print("所有目標已完成！")
            action_dict["base_vel"][:, :] = 0.0
            env.step(action_dict)
            break

        goal = goals[current_goal_index]

        # =========================================================
        # === 這是「證明」的核心 ===
        # 1. 從 SLAM Subscriber 獲取「當前姿態」
        # (pose_buffer 是 [x, y, z, qx, qy, qz, qw])
        # 我們只取 env 0 (因為 num_envs=1)
        current_pose = env.slam_subscriber_manager.pose_buffer[0]
        current_x = current_pose[0]
        current_y = current_pose[1]
        current_yaw = quat_to_yaw(current_pose[3:7])
        
        # 2. 計算「誤差」 (目標姿態 vs 當前姿態)
        error_x = goal["x"] - current_x
        error_y = goal["y"] - current_y
        error_yaw = goal["yaw"] - current_yaw
        
        # 將偏航角誤差標準化到 (-pi, pi)
        while error_yaw > math.pi: error_yaw -= 2 * math.pi
        while error_yaw < -math.pi: error_yaw += 2 * math.pi
        
        distance_to_goal = torch.sqrt(error_x**2 + error_y**2)

        # 3. 檢查是否已到達目標
        if distance_to_goal < GOAL_TOLERANCE_LINEAR and abs(error_yaw) < GOAL_TOLERANCE_ANGULAR:
            print(f"已到達目標 {current_goal_index}: {goal['desc']}")
            current_goal_index += 1
            time.sleep(1.0) # 停頓 1 秒
            continue

        # 4. 根據「誤差」計算「動作」(P-Controller)
        # 這是一個簡易的 P 控制器
        
        # 目標航向角 (從機器人當前位置指向目標位置)
        angle_to_goal = torch.atan2(error_y, error_x)
        error_heading = angle_to_goal - current_yaw
        
        # 標準化航向角誤差
        while error_heading > math.pi: error_heading -= 2 * math.pi
        while error_heading < -math.pi: error_heading += 2 * math.pi
        
        lin_vel_x = 0.0
        ang_vel_z = 0.0
        
        # 如果航向誤差太大 (大於 0.1 弧度)，優先轉向
        if abs(error_heading) > GOAL_TOLERANCE_ANGULAR:
            ang_vel_z = KP_ANGULAR * error_heading
        # 如果航向正確，則前進
        elif distance_to_goal > GOAL_TOLERANCE_LINEAR:
            lin_vel_x = KP_LINEAR * distance_to_goal
            # 同時修正航向
            ang_vel_z = KP_ANGULAR * error_heading
        # 如果距離和航向都接近了，就只修正最終的姿態 (yaw)
        else:
            ang_vel_z = KP_ANGULAR * error_yaw

        # 限制動作的輸出
        lin_vel_x = torch.clamp(lin_vel_x, -1.0, 1.0) # 乘以 cfg.actions.base_vel.scale[0]
        ang_vel_z = torch.clamp(ang_vel_z, -1.0, 1.0) # 乘以 cfg.actions.base_vel.scale[2]

        action_dict["base_vel"][:, 0] = lin_vel_x
        action_dict["base_vel"][:, 2] = ang_vel_z
        
        # === 證明結束 ===
        # =========================================================

        # 5. 執行這一步
        obs, rewards, dones, info = env.step(action_dict)
        
        # 打印狀態
        if i % 30 == 0: # 每 30 步 (約 0.6 秒) 打印一次
            print(f"Step {i} | 目標 {current_goal_index}: {goal['desc']}")
            print(f"  SLAM 姿態: x={current_x:.2f}, y={current_y:.2f}, yaw={current_yaw:.2f}")
            print(f"  目標姿態: x={goal['x']:.2f}, y={goal['y']:.2f}, yaw={goal['yaw']:.2f}")
            print(f"  誤差 (Error): dist={distance_to_goal:.2f}, yaw_err={error_yaw:.2f}")
            print(f"  動作 (Action): V_x={lin_vel_x:.2f}, W_z={ang_vel_z:.2f}")
            
        time.sleep(env.step_dt) # 模擬即時

    # --- 5. 關閉 ---
    print("測試迴圈結束。")
    env.close()
    rclpy.shutdown()
    simulation_app.close()

if __name__ == "__main__":
    main()