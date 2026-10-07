# /scripts/test_slam_control_loop.py
# (closed-loop control test)

import gymnasium as gym
import torch
import rclpy
import time
import math

# Isaac Lab app launcher
from isaaclab.app import AppLauncher

# Our environment
from spot_vslam.envs.rl_env import ManagerBasedRLEnv
from spot_vslam.tasks.spot_vslam.spot_vslam_test_cfg import SpotSlamTestEnvCfg

# !! (Important) !! the robot config
from spot_vslam.assets.spot_with_camera import SPOT_CFG # <-- **change this if your robot is different**

def quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """Convert a [qx, qy, qz, qw] quaternion to yaw."""
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

def main():
    """Main test: closed-loop control."""
    
    # --- 1. Init ---
    rclpy.init()
    app_launcher = AppLauncher(headless=False)
    simulation_app = app_launcher.app

    # --- 2. Load a "clean" env config ---
    print("Loading the env config (spot_vslam_test_cfg.py)...")
    env_cfg = SpotSlamTestEnvCfg()

    # Fill in the MISSING robot config
    env_cfg.scene.robot = SPOT_CFG.copy() # <-- **change your robot here**
    
    # Create the env (num_envs=1)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Reset the env
    obs_dict, info_dict = env.reset()
    print("Env reset.")
    print(f"ROS topics should be published on {env_cfg.ros2.topic_prefix}0/...")
    print("ORB-SLAM3 should subscribe to:")
    print(f"  {env_cfg.ros2.topic_prefix}0/camera/image_raw")
    print(f"  {env_cfg.ros2.topic_prefix}0/camera/camera_info")
    print(f"\n[Important] ORB-SLAM3 must publish the pose to: {env_cfg.slam_subscriber.pose_topic}")
    print("\nPress Enter to start the closed-loop control test...")
    input()

    # --- 3. Controller gains and goals ---
    KP_LINEAR = 1.0  # P gain (linear velocity)
    KP_ANGULAR = 1.5 # P gain (angular velocity)
    GOAL_TOLERANCE_LINEAR = 0.1 # 10 cm tolerance
    GOAL_TOLERANCE_ANGULAR = 0.1 # 0.1 rad tolerance (~6 deg)

    # Sequence of goals (target poses)
    goals = [
        {"x": 2.0, "y": 0.0, "yaw": 0.0, "desc": "move to (2, 0)"},
        {"x": 2.0, "y": 0.0, "yaw": 1.57, "desc": "turn to 90 deg"},
        {"x": 2.0, "y": 2.0, "yaw": 1.57, "desc": "move to (2, 2)"},
        {"x": 2.0, "y": 2.0, "yaw": 0.0, "desc": "turn to 0 deg"},
        {"x": 0.0, "y": 0.0, "yaw": 0.0, "desc": "back to the origin (0, 0)"}
    ]
    current_goal_index = 0

    action_dict = {
        "base_vel": torch.zeros((env.num_envs, env.action_space["base_vel"].shape[1]), device=env.device)
    }
    
    # --- 4. Closed-loop control ---
    num_steps = 3000 # 3000 steps in total
    for i in range(num_steps):
        
        if current_goal_index >= len(goals):
            print("All goals done!")
            action_dict["base_vel"][:, :] = 0.0
            env.step(action_dict)
            break

        goal = goals[current_goal_index]

        # =========================================================
        # === The core of the "proof" ===
        # 1. Current pose from the SLAM subscriber
        # (pose_buffer is [x, y, z, qx, qy, qz, qw])
        # env 0 only (num_envs = 1)
        current_pose = env.slam_subscriber_manager.pose_buffer[0]
        current_x = current_pose[0]
        current_y = current_pose[1]
        current_yaw = quat_to_yaw(current_pose[3:7])
        
        # 2. Error (goal pose vs current pose)
        error_x = goal["x"] - current_x
        error_y = goal["y"] - current_y
        error_yaw = goal["yaw"] - current_yaw
        
        # Wrap the yaw error to (-pi, pi)
        while error_yaw > math.pi: error_yaw -= 2 * math.pi
        while error_yaw < -math.pi: error_yaw += 2 * math.pi
        
        distance_to_goal = torch.sqrt(error_x**2 + error_y**2)

        # 3. Goal reached?
        if distance_to_goal < GOAL_TOLERANCE_LINEAR and abs(error_yaw) < GOAL_TOLERANCE_ANGULAR:
            print(f"Reached goal {current_goal_index}: {goal['desc']}")
            current_goal_index += 1
            time.sleep(1.0) # pause 1 s
            continue

        # 4. Action from the error (P controller)
        # A simple P controller
        
        # Desired heading (from the robot towards the goal)
        angle_to_goal = torch.atan2(error_y, error_x)
        error_heading = angle_to_goal - current_yaw
        
        # Wrap the heading error
        while error_heading > math.pi: error_heading -= 2 * math.pi
        while error_heading < -math.pi: error_heading += 2 * math.pi
        
        lin_vel_x = 0.0
        ang_vel_z = 0.0
        
        # If the heading error is large (> 0.1 rad), turn first
        if abs(error_heading) > GOAL_TOLERANCE_ANGULAR:
            ang_vel_z = KP_ANGULAR * error_heading
        # If the heading is right, drive forward
        elif distance_to_goal > GOAL_TOLERANCE_LINEAR:
            lin_vel_x = KP_LINEAR * distance_to_goal
            # while correcting the heading
            ang_vel_z = KP_ANGULAR * error_heading
        # When position and heading are close, only correct the final yaw
        else:
            ang_vel_z = KP_ANGULAR * error_yaw

        # Limit the action
        lin_vel_x = torch.clamp(lin_vel_x, -1.0, 1.0) # times cfg.actions.base_vel.scale[0]
        ang_vel_z = torch.clamp(ang_vel_z, -1.0, 1.0) # times cfg.actions.base_vel.scale[2]

        action_dict["base_vel"][:, 0] = lin_vel_x
        action_dict["base_vel"][:, 2] = ang_vel_z
        
        # === End of the proof ===
        # =========================================================

        # 5. Step
        obs, rewards, dones, info = env.step(action_dict)
        
        # Print the state
        if i % 30 == 0: # every 30 steps (~0.6 s)
            print(f"Step {i} | goal {current_goal_index}: {goal['desc']}")
            print(f"  SLAM pose: x={current_x:.2f}, y={current_y:.2f}, yaw={current_yaw:.2f}")
            print(f"  Goal pose: x={goal['x']:.2f}, y={goal['y']:.2f}, yaw={goal['yaw']:.2f}")
            print(f"  Error: dist={distance_to_goal:.2f}, yaw_err={error_yaw:.2f}")
            print(f"  Action: V_x={lin_vel_x:.2f}, W_z={ang_vel_z:.2f}")
            
        time.sleep(env.step_dt) # real time

    # --- 5. Close ---
    print("Test loop done.")
    env.close()
    rclpy.shutdown()
    simulation_app.close()

if __name__ == "__main__":
    main()