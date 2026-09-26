# source/spot_vslam/spot_vslam/my_project/envs/test_rotate_180.py
# 狀態：[已修復] 移除重複的 ros2_mgr.update，解決 Double Publishing 問題

import argparse
from isaaclab.app import AppLauncher

# --- 1. 設定參數 ---
parser = argparse.ArgumentParser(description="VSLAM 壓力測試")
parser.add_argument("--task", type=str, default="Spot-test180-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --- 2. 啟動模擬器 ---
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 3. 匯入庫 ---
import gymnasium as gym
import torch
import rclpy
import time
import math
import matplotlib.pyplot as plt
from spot_vslam.envs.rl_env import ManagerBasedRLEnv
from spot_vslam.tasks.spot_vslam.spot_vslam_test_cfg import SpotSlamTestEnvCfg
from spot_vslam.assets.spot_with_camera import SPOT_CFG

from spot_vslam.managers.ros2_manager import Ros2Manager
from spot_vslam.managers.orb_slam_subscriber_manager import OrbSlamSubscriberManager

def smart_step(env, action):
    ret = env.step(action)
    if isinstance(ret, tuple):
        if len(ret) == 4: return ret[0], ret[3]
        elif len(ret) == 5: return ret[0], ret[4]
        else: return ret[0], {}
    else: return ret, {}

def main():
    rclpy.init()

    SPIN_LOOPS = 2       
    SPIN_SPEED = 0.4     
    MOVE_DISTANCE = 2.0
    MOVE_SPEED = 0.005
    START_HEIGHT = 1.0  
    CAMERA_OFFSET_X = 0.4 
    
    print("正在載入環境與 Manager...")
    env_cfg = SpotSlamTestEnvCfg()
    env_cfg.scene.robot = SPOT_CFG.copy()
    env_cfg.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
    env_cfg.sim.gravity = (0.0, 0.0, 0.0) 
    if env_cfg.observations.policy is not None:
        env_cfg.observations.policy.concatenate_terms = False

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # [注意] 這裡不需要再手動建立 Ros2Manager 了，因為 env 內部已經有了
    # 但為了保險起見 (如果是外部需要 reset)，我們先留著變數，但不在迴圈呼叫 update
    # 如果 base_env.ros2_manager 存在，我們就用它
    if hasattr(base_env, "ros2_manager"):
        print("[INFO] 檢測到環境內部已有 Ros2Manager，將使用內部實例。")
        ros2_mgr = base_env.ros2_manager
    else:
        print("[INFO] 環境內部無 Ros2Manager，手動建立。")
        ros2_mgr = Ros2Manager(env_cfg.ros2, base_env)

    # SLAM Subscriber 還是需要手動建立並掛載
    slam_sub_mgr = OrbSlamSubscriberManager(env_cfg.slam_subscriber, base_env)
    base_env.slam_subscriber_manager = slam_sub_mgr

    robot = base_env.scene["robot"]
    stand_joint_pos = robot.data.default_joint_pos.torch.clone()

    base_env.reset()
    # 確保 ros2_mgr 重置
    if hasattr(ros2_mgr, "reset"): ros2_mgr.reset(env_ids=range(base_env.num_envs))
    slam_sub_mgr.reset(env_ids=range(base_env.num_envs))
    
    dt = base_env.step_dt
    sim_pos = torch.tensor([[0.0, 0.0, START_HEIGHT]], device=base_env.device)
    sim_yaw_rad = 0.0

    print("環境初始化完成。等待 ROS 穩定...")
    for _ in range(30):
        simulation_app.update()

    # [階段 0] 熱身
    print(f"\n[階段 0] 熱身直到 SLAM 初始化...")
    warmup_steps = 0
    slam_ready = False
    
    while not slam_ready:
        if not simulation_app.is_running(): return
        
        sway = 0.2 * math.sin(warmup_steps * 0.1)
        sim_pos[0, 1] = sway 
        
        cy = math.cos(sim_yaw_rad * 0.5)
        sy = math.sin(sim_yaw_rad * 0.5)
        sim_quat = torch.tensor([[0.0, 0.0, sy, cy]], device=base_env.device)
        
        robot.write_root_pose_to_sim_index(root_pose=torch.cat([sim_pos, sim_quat], dim=-1))
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base_env.device))
        robot.write_joint_position_to_sim_index(position=stand_joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(stand_joint_pos))
        
        # [關鍵修正] 只呼叫 smart_step，不要再呼叫 ros2_mgr.update(dt)！
        # smart_step -> env.step -> env.ros2_manager.update (內部已執行)
        smart_step(base_env, stand_joint_pos)
        
        # Subscriber 是純接收，可以手動呼叫 (或者 env.step 內部也有呼叫，重複呼叫沒副作用)
        slam_sub_mgr.update(dt)
        
        if slam_sub_mgr.state_buffer[0].item() == 1: 
            print(f"\n✅ SLAM TRACKING!")
            slam_ready = True
            for _ in range(20): 
                smart_step(base_env, stand_joint_pos)
                slam_sub_mgr.update(dt)
            break
        warmup_steps += 1
        if warmup_steps > 3000: break

    # 歸零準備正式測試
    sim_pos[0, 0] = 0.0
    sim_pos[0, 1] = 0.0
    robot.write_root_pose_to_sim_index(root_pose=torch.cat([sim_pos, sim_quat], dim=-1))
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base_env.device))
    for _ in range(10): smart_step(base_env, stand_joint_pos)

    print(f"\n=== 正式測試開始 (紀錄數據) ===")
    
    start_base_gt = robot.data.root_pos_w.torch[0, :3].clone().cpu().numpy()
    start_yaw_gt = 0.0 
    
    start_cam_gt_x = start_base_gt[0] + CAMERA_OFFSET_X * math.cos(start_yaw_gt)
    start_cam_gt_y = start_base_gt[1] + CAMERA_OFFSET_X * math.sin(start_yaw_gt)
    
    start_slam = slam_sub_mgr.pose_buffer[0].clone().cpu().numpy()
    
    log_gt_x = []
    log_gt_y = []
    log_slam_x = []
    log_slam_y = []
    log_error = []
    log_steps = []

    total_steps = 0

    # [階段 1] 轉圈
    print(f"\n[階段 1] 自轉 {SPIN_LOOPS} 圈...")
    target_yaw = SPIN_LOOPS * 2 * math.pi
    
    while sim_yaw_rad < target_yaw:
        if not simulation_app.is_running(): return

        sim_yaw_rad += SPIN_SPEED * dt
        
        cy = math.cos(sim_yaw_rad * 0.5)
        sy = math.sin(sim_yaw_rad * 0.5)
        sim_quat = torch.tensor([[0.0, 0.0, sy, cy]], device=base_env.device)
        robot.write_root_pose_to_sim_index(root_pose=torch.cat([sim_pos, sim_quat], dim=-1))
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base_env.device))
        robot.write_joint_position_to_sim_index(position=stand_joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(stand_joint_pos))
        
        # [關鍵修正] 移除重複的 ros2_mgr.update
        smart_step(base_env, stand_joint_pos)
        slam_sub_mgr.update(dt)
        
        if total_steps % 5 == 0:
            curr_base_gt = robot.data.root_pos_w.torch[0, :3].cpu().numpy()
            curr_cam_gt_x = curr_base_gt[0] + CAMERA_OFFSET_X * math.cos(sim_yaw_rad)
            curr_cam_gt_y = curr_base_gt[1] + CAMERA_OFFSET_X * math.sin(sim_yaw_rad)
            curr_slam = slam_sub_mgr.pose_buffer[0].cpu().numpy()
            
            vec_gt_x = curr_cam_gt_x - start_cam_gt_x
            vec_gt_y = curr_cam_gt_y - start_cam_gt_y
            vec_slam_x = curr_slam[0] - start_slam[0]
            vec_slam_y = curr_slam[1] - start_slam[1]
            
            err_x = vec_gt_x - vec_slam_x
            err_y = vec_gt_y - vec_slam_y
            error = math.sqrt(err_x**2 + err_y**2)
            
            log_gt_x.append(vec_gt_x)
            log_gt_y.append(vec_gt_y)
            log_slam_x.append(vec_slam_x)
            log_slam_y.append(vec_slam_y)
            log_error.append(error)
            log_steps.append(total_steps)
            
            if total_steps % 20 == 0:
                print(f"Spinning... Ang:{math.degrees(sim_yaw_rad):.1f} | Err:{error:.4f}m")

        total_steps += 1

    # [階段 2] 直線移動
    print(f"\n[階段 2] 直線移動 {MOVE_DISTANCE}m...")
    start_pos_x = sim_pos[0, 0].item()
    start_pos_y = sim_pos[0, 1].item()
    dist_traveled = 0.0
    dir_x = math.cos(sim_yaw_rad)
    dir_y = math.sin(sim_yaw_rad)

    while dist_traveled < MOVE_DISTANCE:
        if not simulation_app.is_running(): return
        
        sim_pos[0, 0] += dir_x * MOVE_SPEED
        sim_pos[0, 1] += dir_y * MOVE_SPEED
        dist_traveled = math.sqrt((sim_pos[0,0]-start_pos_x)**2 + (sim_pos[0,1]-start_pos_y)**2)
        
        cy = math.cos(sim_yaw_rad * 0.5)
        sy = math.sin(sim_yaw_rad * 0.5)
        sim_quat = torch.tensor([[0.0, 0.0, sy, cy]], device=base_env.device)
        robot.write_root_pose_to_sim_index(root_pose=torch.cat([sim_pos, sim_quat], dim=-1))
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base_env.device))
        robot.write_joint_position_to_sim_index(position=stand_joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(stand_joint_pos))
        
        # [關鍵修正] 移除重複的 ros2_mgr.update
        smart_step(base_env, stand_joint_pos)
        slam_sub_mgr.update(dt)
        
        if total_steps % 5 == 0:
            curr_base_gt = robot.data.root_pos_w.torch[0, :3].cpu().numpy()
            curr_cam_gt_x = curr_base_gt[0] + CAMERA_OFFSET_X * math.cos(sim_yaw_rad)
            curr_cam_gt_y = curr_base_gt[1] + CAMERA_OFFSET_X * math.sin(sim_yaw_rad)
            curr_slam = slam_sub_mgr.pose_buffer[0].cpu().numpy()
            
            vec_gt_x = curr_cam_gt_x - start_cam_gt_x
            vec_gt_y = curr_cam_gt_y - start_cam_gt_y
            vec_slam_x = curr_slam[0] - start_slam[0]
            vec_slam_y = curr_slam[1] - start_slam[1]
            
            err_x = vec_gt_x - vec_slam_x
            err_y = vec_gt_y - vec_slam_y
            error = math.sqrt(err_x**2 + err_y**2)
            
            log_gt_x.append(vec_gt_x)
            log_gt_y.append(vec_gt_y)
            log_slam_x.append(vec_slam_x)
            log_slam_y.append(vec_slam_y)
            log_error.append(error)
            log_steps.append(total_steps)
            
            if total_steps % 20 == 0:
                print(f"Moving... Dist:{dist_traveled:.2f} | Err:{error:.4f}m")

        total_steps += 1

    print("\n[測試結束] 正在繪製誤差圖...")
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(log_gt_x, log_gt_y, 'b-', label='Ground Truth (Camera)')
    plt.plot(log_slam_x, log_slam_y, 'r--', label='SLAM Estimate')
    plt.title(f'Trajectory Comparison (Offset: {CAMERA_OFFSET_X}m)')
    plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.legend(); plt.grid(True); plt.axis('equal') 
    
    plt.subplot(1, 2, 2)
    plt.plot(log_steps, log_error, 'g-')
    plt.title('Absolute Position Error (ATE)')
    plt.xlabel('Steps'); plt.ylabel('Error (m)'); plt.grid(True)
    
    filename = 'vslam_result_compensated.png'
    plt.savefig(filename)
    print(f"圖表已儲存至: {filename}")

    # ros2_mgr 已經由 base_env 管理，這裡不需手動 close，或讓 env.close() 去處理
    slam_sub_mgr.close()
    base_env.close()
    rclpy.shutdown()
    simulation_app.close()

if __name__ == "__main__":
    main()