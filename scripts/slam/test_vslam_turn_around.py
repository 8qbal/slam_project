# source/spot_vslam/spot_vslam/my_project/envs/test_square_spiral_v2.py
# 狀態：[完美版] 3圈建圖 -> 紅線分隔 -> 方形螺旋 (直走->左轉)

import argparse
from isaaclab.app import AppLauncher

# --- 1. 設定參數 ---
parser = argparse.ArgumentParser(description="VSLAM 測試：3圈建圖 + 方形螺旋")
parser.add_argument("--task", type=str, default="Spot-test-spiral-v0")
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

    # [保護機制] 30 FPS 限速，確保 SLAM Viewer 不崩潰
    print("[INFO] 啟用 30 FPS 限速...")
    simulation_app.set_setting("/app/runLoops/main/rateLimitEnabled", True)
    simulation_app.set_setting("/app/runLoops/main/rateLimit", 30)

    # ==========================
    # [參數設定區]
    # ==========================
    # 1. 轉圈設定
    SPIN_LOOPS = 3       # 轉 3 圈 (建圖)
    SPIN_SPEED = 0.4     # 轉速 (rad/step)
    
    # 2. 方形螺旋設定
    # 依序執行：直走 -> 左轉
    # 1.0m -> 左轉 -> 1.0m -> 左轉 -> 2.0m -> 左轉 -> 2.0m -> 左轉
    SPIRAL_LENGTHS = [1.0, 1.0, 2.0, 2.0] 
    
    MOVE_SPEED = 0.005   # 直線速度 (m/step)
    TURN_SPEED = 0.4     # 轉彎速度 (rad/step)
    
    START_HEIGHT = 1.0  
    CAMERA_OFFSET_X = 0.4 # 相機在前方 0.4m
    
    print("正在載入環境與 Manager...")
    env_cfg = SpotSlamTestEnvCfg()
    env_cfg.scene.robot = SPOT_CFG.copy()
    env_cfg.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
    env_cfg.sim.gravity = (0.0, 0.0, 0.0) 
    if env_cfg.observations.policy is not None:
        env_cfg.observations.policy.concatenate_terms = False

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    
    if hasattr(base_env, "ros2_manager"):
        ros2_mgr = base_env.ros2_manager
    else:
        ros2_mgr = Ros2Manager(env_cfg.ros2, base_env)

    slam_sub_mgr = OrbSlamSubscriberManager(env_cfg.slam_subscriber, base_env)
    base_env.slam_subscriber_manager = slam_sub_mgr

    robot = base_env.scene["robot"]
    stand_joint_pos = robot.data.default_joint_pos.clone()

    base_env.reset()
    if hasattr(ros2_mgr, "reset"): ros2_mgr.reset(env_ids=range(base_env.num_envs))
    slam_sub_mgr.reset(env_ids=range(base_env.num_envs))
    
    dt = base_env.step_dt
    sim_pos = torch.tensor([[0.0, 0.0, START_HEIGHT]], device=base_env.device)
    sim_yaw_rad = 0.0 # 這是機器人的累積角度

    print("環境初始化完成。等待 ROS 穩定...")
    for _ in range(30):
        simulation_app.update()

    # ==================================================================================
    # [階段 0] 熱身 (原地晃動直到 SLAM Tracking)
    # ==================================================================================
    print(f"\n[階段 0] 熱身直到 SLAM 初始化...")
    warmup_steps = 0
    slam_ready = False
    
    while not slam_ready:
        if not simulation_app.is_running(): return
        
        sway = 0.2 * math.sin(warmup_steps * 0.1)
        sim_pos[0, 1] = sway 
        
        cy = math.cos(sim_yaw_rad * 0.5)
        sy = math.sin(sim_yaw_rad * 0.5)
        sim_quat = torch.tensor([[cy, 0.0, 0.0, sy]], device=base_env.device)
        
        robot.write_root_pose_to_sim(torch.cat([sim_pos, sim_quat], dim=-1))
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=base_env.device))
        robot.write_joint_state_to_sim(position=stand_joint_pos, velocity=torch.zeros_like(stand_joint_pos))
        
        smart_step(base_env, stand_joint_pos)
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
    robot.write_root_pose_to_sim(torch.cat([sim_pos, sim_quat], dim=-1))
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=base_env.device))
    for _ in range(10): smart_step(base_env, stand_joint_pos)

    print(f"\n=== 正式測試開始 ===")
    
    # 紀錄基準點
    start_base_gt = robot.data.root_pos_w[0, :3].clone().cpu().numpy()
    start_yaw_gt = 0.0 
    
    start_cam_gt_x = start_base_gt[0] + CAMERA_OFFSET_X * math.cos(start_yaw_gt)
    start_cam_gt_y = start_base_gt[1] + CAMERA_OFFSET_X * math.sin(start_yaw_gt)
    
    start_slam = slam_sub_mgr.pose_buffer[0].clone().cpu().numpy()
    
    log_gt_x, log_gt_y = [], []
    log_slam_x, log_slam_y = [], []
    log_error, log_steps = [], []

    total_steps = 0
    spin_end_step = 0 # 用來記錄建圖完成的時間點

    # ==================================================================================
    # [階段 1] 原地轉 3 圈 (Mapping Phase)
    # ==================================================================================
    print(f"\n[階段 1] 原地轉 {SPIN_LOOPS} 圈 (建立地圖)...")
    target_yaw = SPIN_LOOPS * 2 * math.pi
    
    while sim_yaw_rad < target_yaw:
        if not simulation_app.is_running(): return

        sim_yaw_rad += SPIN_SPEED * dt
        
        cy = math.cos(sim_yaw_rad * 0.5)
        sy = math.sin(sim_yaw_rad * 0.5)
        sim_quat = torch.tensor([[cy, 0.0, 0.0, sy]], device=base_env.device)
        robot.write_root_pose_to_sim(torch.cat([sim_pos, sim_quat], dim=-1))
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=base_env.device))
        robot.write_joint_state_to_sim(position=stand_joint_pos, velocity=torch.zeros_like(stand_joint_pos))
        
        smart_step(base_env, stand_joint_pos)
        slam_sub_mgr.update(dt)
        
        # 紀錄數據
        if total_steps % 5 == 0:
            curr_base_gt = robot.data.root_pos_w[0, :3].cpu().numpy()
            curr_cam_gt_x = curr_base_gt[0] + CAMERA_OFFSET_X * math.cos(sim_yaw_rad)
            curr_cam_gt_y = curr_base_gt[1] + CAMERA_OFFSET_X * math.sin(sim_yaw_rad)
            curr_slam = slam_sub_mgr.pose_buffer[0].cpu().numpy()
            
            vec_gt_x = curr_cam_gt_x - start_cam_gt_x
            vec_gt_y = curr_cam_gt_y - start_cam_gt_y
            vec_slam_x = curr_slam[0] - start_slam[0]
            vec_slam_y = curr_slam[1] - start_slam[1]
            
            error = math.sqrt((vec_gt_x - vec_slam_x)**2 + (vec_gt_y - vec_slam_y)**2)
            
            log_gt_x.append(vec_gt_x)
            log_gt_y.append(vec_gt_y)
            log_slam_x.append(vec_slam_x)
            log_slam_y.append(vec_slam_y)
            log_error.append(error)
            log_steps.append(total_steps)
            
            if total_steps % 50 == 0:
                print(f"Spinning... Ang:{math.degrees(sim_yaw_rad):.1f}° | Err:{error:.4f}m")

        total_steps += 1

    # 紀錄建圖結束的時間點 (為了畫紅線)
    spin_end_step = total_steps
    print(f"✅ 建圖完成 (Step: {spin_end_step})，準備開始方形移動...")
    
    # 暫停一下讓 SLAM 存好關鍵幀
    for _ in range(50):
        smart_step(base_env, stand_joint_pos)
        slam_sub_mgr.update(dt)

    # ==================================================================================
    # [階段 2] 方形螺旋 (Straight -> Turn Left -> Straight -> Turn Left ...)
    # ==================================================================================
    print(f"\n[階段 2] 方形螺旋測試 {SPIRAL_LENGTHS}...")

    for seg_idx, target_dist in enumerate(SPIRAL_LENGTHS):
        print(f"\n>>> [Segment {seg_idx+1}] 直走 {target_dist}m...")
        
        # --- A. 直走 ---
        seg_traveled = 0.0
        start_seg_x = sim_pos[0, 0].item()
        start_seg_y = sim_pos[0, 1].item()
        
        dir_x = math.cos(sim_yaw_rad)
        dir_y = math.sin(sim_yaw_rad)
        
        while seg_traveled < target_dist:
            if not simulation_app.is_running(): return
            
            sim_pos[0, 0] += dir_x * MOVE_SPEED
            sim_pos[0, 1] += dir_y * MOVE_SPEED
            
            seg_traveled = math.sqrt((sim_pos[0,0]-start_seg_x)**2 + (sim_pos[0,1]-start_seg_y)**2)
            
            cy = math.cos(sim_yaw_rad * 0.5)
            sy = math.sin(sim_yaw_rad * 0.5)
            sim_quat = torch.tensor([[cy, 0.0, 0.0, sy]], device=base_env.device)
            robot.write_root_pose_to_sim(torch.cat([sim_pos, sim_quat], dim=-1))
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=base_env.device))
            robot.write_joint_state_to_sim(position=stand_joint_pos, velocity=torch.zeros_like(stand_joint_pos))
            
            smart_step(base_env, stand_joint_pos)
            slam_sub_mgr.update(dt)
            
            if total_steps % 5 == 0:
                curr_base_gt = robot.data.root_pos_w[0, :3].cpu().numpy()
                curr_cam_gt_x = curr_base_gt[0] + CAMERA_OFFSET_X * math.cos(sim_yaw_rad)
                curr_cam_gt_y = curr_base_gt[1] + CAMERA_OFFSET_X * math.sin(sim_yaw_rad)
                curr_slam = slam_sub_mgr.pose_buffer[0].cpu().numpy()
                
                vec_gt_x = curr_cam_gt_x - start_cam_gt_x
                vec_gt_y = curr_cam_gt_y - start_cam_gt_y
                vec_slam_x = curr_slam[0] - start_slam[0]
                vec_slam_y = curr_slam[1] - start_slam[1]
                
                error = math.sqrt((vec_gt_x - vec_slam_x)**2 + (vec_gt_y - vec_slam_y)**2)
                
                log_gt_x.append(vec_gt_x)
                log_gt_y.append(vec_gt_y)
                log_slam_x.append(vec_slam_x)
                log_slam_y.append(vec_slam_y)
                log_error.append(error)
                log_steps.append(total_steps)
            
            total_steps += 1

        # --- B. 左轉 90 度 (每走完一段就要轉) ---
        print(f"    [Turn] 左轉 90 度...")
        target_yaw_turn = sim_yaw_rad + (math.pi / 2)
        
        while sim_yaw_rad < target_yaw_turn:
            if not simulation_app.is_running(): return
            
            sim_yaw_rad += TURN_SPEED * dt
            
            cy = math.cos(sim_yaw_rad * 0.5)
            sy = math.sin(sim_yaw_rad * 0.5)
            sim_quat = torch.tensor([[cy, 0.0, 0.0, sy]], device=base_env.device)
            robot.write_root_pose_to_sim(torch.cat([sim_pos, sim_quat], dim=-1))
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=base_env.device))
            robot.write_joint_state_to_sim(position=stand_joint_pos, velocity=torch.zeros_like(stand_joint_pos))
            
            smart_step(base_env, stand_joint_pos)
            slam_sub_mgr.update(dt)
            
            if total_steps % 5 == 0:
                curr_base_gt = robot.data.root_pos_w[0, :3].cpu().numpy()
                curr_cam_gt_x = curr_base_gt[0] + CAMERA_OFFSET_X * math.cos(sim_yaw_rad)
                curr_cam_gt_y = curr_base_gt[1] + CAMERA_OFFSET_X * math.sin(sim_yaw_rad)
                curr_slam = slam_sub_mgr.pose_buffer[0].cpu().numpy()
                
                vec_gt_x = curr_cam_gt_x - start_cam_gt_x
                vec_gt_y = curr_cam_gt_y - start_cam_gt_y
                vec_slam_x = curr_slam[0] - start_slam[0]
                vec_slam_y = curr_slam[1] - start_slam[1]
                
                error = math.sqrt((vec_gt_x - vec_slam_x)**2 + (vec_gt_y - vec_slam_y)**2)
                
                log_gt_x.append(vec_gt_x)
                log_gt_y.append(vec_gt_y)
                log_slam_x.append(vec_slam_x)
                log_slam_y.append(vec_slam_y)
                log_error.append(error)
                log_steps.append(total_steps)
            
            total_steps += 1

    print("\n[測試結束] 正在繪製誤差圖...")
    
    # --- 繪製 Matplotlib 圖表 ---
    plt.figure(figsize=(12, 6))
    
    # 子圖 1: 軌跡比較
    plt.subplot(1, 2, 1)
    plt.plot(log_gt_x, log_gt_y, 'b-', label='Ground Truth (Camera)')
    plt.plot(log_slam_x, log_slam_y, 'r--', label='SLAM Estimate')
    plt.title(f'Square Spiral Trajectory (Offset: {CAMERA_OFFSET_X}m)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') 
    
    # 子圖 2: 誤差隨時間變化 (包含紅色虛線)
    plt.subplot(1, 2, 2)
    plt.plot(log_steps, log_error, 'g-', label='ATE')
    
    # [關鍵新增] 畫紅色垂直虛線代表建圖完成
    plt.axvline(x=spin_end_step, color='r', linestyle='--', label='Mapping Done (Spin End)')
    
    plt.title('Absolute Position Error (ATE)')
    plt.xlabel('Steps')
    plt.ylabel('Error (m)')
    plt.legend()
    plt.grid(True)
    
    filename = 'vslam_result_spiral.png'
    plt.savefig(filename)
    print(f"圖表已儲存至: {filename}")

    slam_sub_mgr.close()
    try:
        base_env.close()
    except AttributeError:
        pass
    rclpy.shutdown()
    simulation_app.close()

if __name__ == "__main__":
    main()