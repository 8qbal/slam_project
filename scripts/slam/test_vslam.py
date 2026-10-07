# source/spot_vslam/spot_vslam/my_project/envs/test_square_spiral.py
# Status: square spiral path test

import argparse
from isaaclab.app import AppLauncher

# --- 1. Parameters ---
parser = argparse.ArgumentParser(description="VSLAM test: square spiral path")
parser.add_argument("--task", type=str, default="Spot-test-spiral-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --- 2. Launch the simulator ---
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 3. Imports ---
import gymnasium as gym
import torch
import rclpy
import time
import math
import matplotlib.pyplot as plt
from spot_vslam.envs.rl_env import ManagerBasedRLEnv
from spot_vslam.tasks.slam_test.slam_test_env_cfg import SlamTestEnvCfg as SpotSlamTestEnvCfg
from spot_vslam.assets.spot import SPOT_CFG

from spot_vslam.slam.ros2_manager import Ros2Manager
from spot_vslam.slam.orb_slam_subscriber import OrbSlamSubscriberManager

def smart_step(env, action):
    ret = env.step(action)
    if isinstance(ret, tuple):
        if len(ret) == 4: return ret[0], ret[3]
        elif len(ret) == 5: return ret[0], ret[4]
        else: return ret[0], {}
    else: return ret, {}

def main():
    rclpy.init()

    # ==========================
    # [Parameters]
    # ==========================
    # Spiral path: length of each straight segment (m)
    # Logic: move 1 m -> turn 90 -> move 2 m -> turn 90 -> move 3 m ...
    SPIRAL_LENGTHS = [1.0, 2.0, 3.0, 4.0] 
    
    MOVE_SPEED = 0.005   # straight-line speed (m/step)
    TURN_SPEED = 0.4     # turn rate (rad/step)
    
    START_HEIGHT = 1.0  
    CAMERA_OFFSET_X = 0.4 # camera 0.4 m ahead
    
    print("Loading the environment and managers...")
    env_cfg = SpotSlamTestEnvCfg()
    env_cfg.scene.robot = SPOT_CFG.copy()
    env_cfg.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
    env_cfg.sim.gravity = (0.0, 0.0, 0.0) 
    if env_cfg.observations.policy is not None:
        env_cfg.observations.policy.concatenate_terms = False

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # don't create the managers twice
    if hasattr(base_env, "ros2_manager"):
        ros2_mgr = base_env.ros2_manager
    else:
        ros2_mgr = Ros2Manager(env_cfg.ros2, base_env)

    slam_sub_mgr = OrbSlamSubscriberManager(env_cfg.slam_subscriber, base_env)
    base_env.slam_subscriber_manager = slam_sub_mgr

    robot = base_env.scene["robot"]
    stand_joint_pos = robot.data.default_joint_pos.torch.clone()

    base_env.reset()
    if hasattr(ros2_mgr, "reset"): ros2_mgr.reset(env_ids=range(base_env.num_envs))
    slam_sub_mgr.reset(env_ids=range(base_env.num_envs))
    
    dt = base_env.step_dt
    sim_pos = torch.tensor([[0.0, 0.0, START_HEIGHT]], device=base_env.device)
    sim_yaw_rad = 0.0

    print("Environment ready. Waiting for ROS to settle...")
    for _ in range(30):
        simulation_app.update()

    # ==================================================================================
    # [Phase 0] Warm-up (wiggle in place until SLAM tracks)
    # ==================================================================================
    print(f"\n[Phase 0] Warming up until SLAM initializes...")
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

    # Zero everything for the test
    sim_pos[0, 0] = 0.0
    sim_pos[0, 1] = 0.0
    robot.write_root_pose_to_sim_index(root_pose=torch.cat([sim_pos, sim_quat], dim=-1))
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base_env.device))
    for _ in range(10): smart_step(base_env, stand_joint_pos)

    # ==================================================================================
    # Test: square spiral
    # ==================================================================================
    print(f"\n=== Test start (square spiral) ===")
    print(f"Planned path: {SPIRAL_LENGTHS} (m)")
    
    # Reference start point
    start_base_gt = robot.data.root_pos_w.torch[0, :3].clone().cpu().numpy()
    start_yaw_gt = 0.0 
    
    start_cam_gt_x = start_base_gt[0] + CAMERA_OFFSET_X * math.cos(start_yaw_gt)
    start_cam_gt_y = start_base_gt[1] + CAMERA_OFFSET_X * math.sin(start_yaw_gt)
    
    start_slam = slam_sub_mgr.pose_buffer[0].clone().cpu().numpy()
    
    log_gt_x, log_gt_y = [], []
    log_slam_x, log_slam_y = [], []
    log_error, log_steps = [], []

    total_steps = 0

    # --- Main loop: drive each segment, then turn left ---
    for seg_idx, target_dist in enumerate(SPIRAL_LENGTHS):
        print(f"\n>>> [Segment {seg_idx+1}] Forward {target_dist} m...")
        
        # 1. Straight segment
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
            
            # Write to the simulator
            cy = math.cos(sim_yaw_rad * 0.5)
            sy = math.sin(sim_yaw_rad * 0.5)
            sim_quat = torch.tensor([[0.0, 0.0, sy, cy]], device=base_env.device)
            robot.write_root_pose_to_sim_index(root_pose=torch.cat([sim_pos, sim_quat], dim=-1))
            robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base_env.device))
            robot.write_joint_position_to_sim_index(position=stand_joint_pos)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(stand_joint_pos))
            
            # Update ROS/SLAM
            smart_step(base_env, stand_joint_pos)
            slam_sub_mgr.update(dt)
            
            # Log data (every 5 steps)
            if total_steps % 5 == 0:
                curr_base_gt = robot.data.root_pos_w.torch[0, :3].cpu().numpy()
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

        # 2. Turn left 90 degrees
        if seg_idx < len(SPIRAL_LENGTHS): # turn unless this is the last segment
            print(f"    [Turn] Turning left 90 degrees...")
            target_yaw = sim_yaw_rad + (math.pi / 2)
            
            while sim_yaw_rad < target_yaw:
                if not simulation_app.is_running(): return
                
                sim_yaw_rad += TURN_SPEED * dt # turn
                
                # Write to the simulator
                cy = math.cos(sim_yaw_rad * 0.5)
                sy = math.sin(sim_yaw_rad * 0.5)
                sim_quat = torch.tensor([[0.0, 0.0, sy, cy]], device=base_env.device)
                robot.write_root_pose_to_sim_index(root_pose=torch.cat([sim_pos, sim_quat], dim=-1))
                robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base_env.device))
                robot.write_joint_position_to_sim_index(position=stand_joint_pos)
                robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(stand_joint_pos))
                
                smart_step(base_env, stand_joint_pos)
                slam_sub_mgr.update(dt)
                
                # Log while turning too (to see drift)
                if total_steps % 5 == 0:
                    curr_base_gt = robot.data.root_pos_w.torch[0, :3].cpu().numpy()
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
            
            # Make sure the turn is exact (optional: snap to the exact angle)
            # sim_yaw_rad = target_yaw 

    print("\n[Test done] Plotting the error...")
    
    # --- Plot ---
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: trajectory
    plt.subplot(1, 2, 1)
    plt.plot(log_gt_x, log_gt_y, 'b-', label='Ground Truth (Camera)')
    plt.plot(log_slam_x, log_slam_y, 'r--', label='SLAM Estimate')
    plt.title(f'Square Spiral Trajectory (Offset: {CAMERA_OFFSET_X}m)')
    plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.legend(); plt.grid(True); plt.axis('equal') 
    
    # Subplot 2: error
    plt.subplot(1, 2, 2)
    plt.plot(log_steps, log_error, 'g-')
    plt.title('Absolute Position Error (ATE)')
    plt.xlabel('Steps'); plt.ylabel('Error (m)'); plt.grid(True)
    
    filename = 'vslam_walk_around.png'
    plt.savefig(filename)
    print(f"Plot saved to: {filename}")

    slam_sub_mgr.close()
    try:
        base_env.close()
    except AttributeError:
        pass
    rclpy.shutdown()
    simulation_app.close()

if __name__ == "__main__":
    main()