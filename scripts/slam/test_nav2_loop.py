# source/spot_vslam/spot_vslam/my_project/envs/test_hybrid_spiral.py
# 狀態：[座標修正版] 圖表起點改為 Camera Offset (0.4, 0.0)

import argparse
import os
import sys
from isaaclab.app import AppLauncher

# 1. 設定參數
parser = argparse.ArgumentParser(description="Hybrid VSLAM Test")
parser.add_argument("--task", type=str, default="Spot-test-hybrid")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. 啟動模擬器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3. 匯入庫
import torch
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
import math
import matplotlib.pyplot as plt

from rsl_rl.runners import OnPolicyRunner
from spot_vslam.envs.rl_env import ManagerBasedRLEnv
from spot_vslam.tasks.spot_vslam.spot_vslam_test_cfg import SpotSlamTestEnvCfg
from spot_vslam.assets.spot_with_camera import SPOT_CFG
from spot_vslam.managers.ros2_manager import Ros2Manager
from spot_vslam.managers.orb_slam_subscriber_manager import OrbSlamSubscriberManager

# 匯入 Config
try:
    from spot_vslam.envs.spiral_config import SpiralTestConfig as Cfg
except ImportError:
    print("[ERROR] 找不到 spiral_config.py")
    sys.exit(1)

# --- 標準 Wrapper ---
class RslRlWrapper:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        obs_dict, _ = self.env.reset()
        self.num_obs = obs_dict["policy"].shape[1]
        self.num_envs = self.env.num_envs
        self.num_actions = self.env.action_manager.action_term_dim[0]
        if hasattr(self.env, "max_episode_length_s"):
            self.max_episode_length = self.env.max_episode_length_s / self.env.step_dt
        else:
            self.max_episode_length = 1000

    def get_observations(self):
        obs_dict = self.env.observation_manager.compute()
        return obs_dict["policy"], {"observations": obs_dict}

    def reset(self):
        obs_dict, extras = self.env.reset()
        extras["observations"] = obs_dict
        return obs_dict["policy"], extras

    def step(self, actions):
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = terminated | truncated
        extras["observations"] = obs_dict
        return obs_dict["policy"], rew, dones, extras

# --- Nav2 控制節點 ---
class Nav2Controller(Node):
    def __init__(self):
        super().__init__('isaac_nav2_controller')
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.current_vel = torch.zeros(3) 

    def cmd_callback(self, msg):
        self.current_vel[0] = msg.linear.x
        self.current_vel[1] = msg.linear.y
        self.current_vel[2] = msg.angular.z

    def send_goal(self, x, y, yaw):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'odom'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.nav_client.wait_for_server()
        self.nav_client.send_goal_async(goal)

def generate_spiral_waypoints(lengths):
    waypoints = []
    curr_x, curr_y, curr_yaw = 0.0, 0.0, 0.0
    for dist in lengths:
        curr_x += dist * math.cos(curr_yaw)
        curr_y += dist * math.sin(curr_yaw)
        next_yaw = curr_yaw + (math.pi / 2)
        waypoints.append((curr_x, curr_y, next_yaw))
        curr_yaw = next_yaw
    return waypoints

# [新增] 輔助函式：從四元數計算 Yaw
def get_yaw_from_quat(quat):
    # Isaac Lab quat order is [x, y, z, w]
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return yaw

def main():
    rclpy.init()

    # [保護] 限速
    if Cfg.ENABLE_RATE_LIMIT:
        print(f"[INFO] 啟用 {Cfg.RATE_LIMIT_FPS} FPS 限速...")
        simulation_app.set_setting("/app/runLoops/main/rateLimitEnabled", True)
        simulation_app.set_setting("/app/runLoops/main/rateLimit", Cfg.RATE_LIMIT_FPS)

    # 初始化環境
    print("正在載入環境...")
    env_cfg = SpotSlamTestEnvCfg()
    env_cfg.scene.robot = SPOT_CFG.copy()
    env_cfg.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
    env_cfg.sim.gravity = (0.0, 0.0, -9.81) 
    if env_cfg.observations.policy is not None:
        env_cfg.observations.policy.concatenate_terms = True

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    
    if hasattr(base_env, "ros2_manager"):
        ros2_mgr = base_env.ros2_manager
    else:
        ros2_mgr = Ros2Manager(env_cfg.ros2, base_env)

    slam_sub_mgr = OrbSlamSubscriberManager(env_cfg.slam_subscriber, base_env)
    base_env.slam_subscriber_manager = slam_sub_mgr

    nav_node = Nav2Controller()
    robot = base_env.scene["robot"]
    
    # --- 載入 RL Policy ---
    print("[INFO] 載入 RL Policy...")
    rl_env = RslRlWrapper(base_env, device="cuda:0")
    log_root = os.path.dirname(Cfg.MODEL_PATH)
    try:
        runner = OnPolicyRunner(rl_env, Cfg.AGENT_CFG, log_dir=log_root, device="cuda:0")
        runner.load(Cfg.MODEL_PATH)
        policy = runner.get_inference_policy(device="cuda:0")
        print("[SUCCESS] 模型載入成功！")
    except Exception as e:
        print(f"[ERROR] 模型載入失敗: {e}")
        return

    # 重置環境
    obs, _ = rl_env.reset()
    if hasattr(ros2_mgr, "reset"): ros2_mgr.reset(env_ids=range(base_env.num_envs))
    slam_sub_mgr.reset(env_ids=range(base_env.num_envs))
    
    dt = base_env.step_dt
    
    # 物理狀態 (用於 Phase 1 漂浮計算)
    FLOATING_HEIGHT = 0.55 
    sim_pos = torch.tensor([[0.0, 0.0, FLOATING_HEIGHT]], device=base_env.device)
    sim_yaw_rad = 0.0

    print("等待 ROS 穩定...")
    for _ in range(50): simulation_app.update()

    # 數據紀錄
    log_gt_x, log_gt_y = [], []
    log_slam_x, log_slam_y = [], []
    log_error, log_steps = [], []
    total_steps = 0
    spin_end_step = 0

    # [關鍵修正] 計算相機的起始位置 (Offset 0.4)
    start_base_gt = robot.data.root_pos_w.torch[0, :3].clone().cpu().numpy()
    
    # 假設起始 Yaw 為 0，相機在 Robot 前方 0.4m
    start_cam_gt_x = start_base_gt[0] + Cfg.CAMERA_OFFSET_X 
    start_cam_gt_y = start_base_gt[1]
    
    start_slam = None 

    # ==================================================================================
    # [階段 1] 漂浮建圖 (God Mode)
    # ==================================================================================
    print(f"\n[階段 1] 漂浮建圖 (原地轉 {Cfg.SPIN_LOOPS} 圈)...")
    target_yaw = Cfg.SPIN_LOOPS * 2 * math.pi
    
    zero_actions = torch.zeros(base_env.num_envs, rl_env.num_actions, device=base_env.device)

    while sim_yaw_rad < target_yaw:
        if not simulation_app.is_running(): return

        # 1. 計算下一個漂浮位置
        sim_yaw_rad += Cfg.SPIN_SPEED * dt
        
        # 2. 強制寫入模擬器
        cy = math.cos(sim_yaw_rad * 0.5)
        sy = math.sin(sim_yaw_rad * 0.5)
        sim_quat = torch.tensor([[0.0, 0.0, sy, cy]], device=base_env.device)
        
        robot.write_root_pose_to_sim_index(root_pose=torch.cat([sim_pos, sim_quat], dim=-1))
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base_env.device))
        
        # 3. 推進模擬器
        rl_env.step(zero_actions) 
        
        # 4. 更新 ROS
        slam_sub_mgr.update(dt)
        ros2_mgr.update(dt)

        # 5. 檢查 Tracking
        if start_slam is None:
            if slam_sub_mgr.state_buffer[0].item() == 1:
                print("✅ SLAM Tracking 成功！")
                start_slam = slam_sub_mgr.pose_buffer[0].clone().cpu().numpy()
        
        # 6. 記錄 (修正座標系)
        if start_slam is not None and total_steps % Cfg.LOG_INTERVAL == 0:
            # A. 取得機器人當前狀態
            curr_base = robot.data.root_pos_w.torch[0, :3].cpu().numpy()
            curr_quat = robot.data.root_quat_w.torch[0].cpu().numpy()
            curr_yaw = get_yaw_from_quat(curr_quat) # 使用真實角度

            # B. 計算 GT 相機絕對位置 (Base + Rotation * Offset)
            curr_cam_x = curr_base[0] + Cfg.CAMERA_OFFSET_X * math.cos(curr_yaw)
            curr_cam_y = curr_base[1] + Cfg.CAMERA_OFFSET_X * math.sin(curr_yaw)
            
            # C. 計算 GT 向量 (相對於世界原點)
            vec_gt_x = curr_cam_x - start_base_gt[0]
            vec_gt_y = curr_cam_y - start_base_gt[1]

            # D. 計算 SLAM 向量 (並平移到相機起點)
            # SLAM 原始數據是從 (0,0) 開始的，我們要把它移到 (0.4, 0)
            curr_slam = slam_sub_mgr.pose_buffer[0].cpu().numpy()
            
            vec_slam_x = (curr_slam[0] - start_slam[0]) + start_cam_gt_x
            vec_slam_y = (curr_slam[1] - start_slam[1]) + start_cam_gt_y
            
            # E. 誤差計算
            err = math.sqrt((vec_gt_x - vec_slam_x)**2 + (vec_gt_y - vec_slam_y)**2)
            
            log_gt_x.append(vec_gt_x); log_gt_y.append(vec_gt_y)
            log_slam_x.append(vec_slam_x); log_slam_y.append(vec_slam_y)
            log_error.append(err); log_steps.append(total_steps)
            
        total_steps += 1

    spin_end_step = total_steps
    print(f"✅ 建圖階段結束 (Step: {spin_end_step})。")
    print(">>> 切換至 RL 走路模式...")

    # ==================================================================================
    # [階段 2] Nav2 閉環控制 (RL Walking)
    # ==================================================================================
    waypoints = generate_spiral_waypoints(Cfg.SPIRAL_LENGTHS)
    
    obs, _ = rl_env.reset() 

    for wp_idx, (goal_x, goal_y, goal_yaw) in enumerate(waypoints):
        print(f"\n>>> 前往目標 {wp_idx+1}: ({goal_x:.2f}, {goal_y:.2f})")
        
        nav_node.send_goal(goal_x, goal_y, goal_yaw)
        
        arrived = False
        timeout_steps = 3000
        step_count = 0
        
        while not arrived and step_count < timeout_steps:
            if not simulation_app.is_running(): return
            
            # 1. Nav2 -> RL
            rclpy.spin_once(nav_node, timeout_sec=0)
            cmd_tensor = torch.zeros(base_env.num_envs, 3, device=base_env.device)
            
            vx = nav_node.current_vel[0].item()
            vy = nav_node.current_vel[1].item()
            wz = nav_node.current_vel[2].item()

            cmd_tensor[:, 0] = max(-0.5, min(0.5, vx))
            cmd_tensor[:, 1] = max(-0.3, min(0.3, vy))
            cmd_tensor[:, 2] = max(-0.5, min(0.5, wz))
            
            if base_env.command_manager:
                base_env.command_manager.get_term("base_velocity").vel_command_b[:] = cmd_tensor
            
            # 2. RL Action
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, dones, _ = rl_env.step(actions)
            
            # 3. ROS Update
            slam_sub_mgr.update(dt)
            ros2_mgr.update(dt)
            
            # 4. 記錄 (同樣使用修正後的座標系)
            if start_slam is not None and total_steps % Cfg.LOG_INTERVAL == 0:
                # GT Cam
                curr_base = robot.data.root_pos_w.torch[0, :3].cpu().numpy()
                curr_quat = robot.data.root_quat_w.torch[0].cpu().numpy()
                curr_yaw = get_yaw_from_quat(curr_quat)

                curr_cam_x = curr_base[0] + Cfg.CAMERA_OFFSET_X * math.cos(curr_yaw)
                curr_cam_y = curr_base[1] + Cfg.CAMERA_OFFSET_X * math.sin(curr_yaw)
                
                vec_gt_x = curr_cam_x - start_base_gt[0]
                vec_gt_y = curr_cam_y - start_base_gt[1]
                
                # SLAM (Shifted)
                curr_slam = slam_sub_mgr.pose_buffer[0].cpu().numpy()
                
                # 計算與目標的距離 (用 SLAM 座標比較)
                # 注意：Nav2 目標是在 map/odom frame，我們這邊簡單用 shift 後的 SLAM 座標來比
                slam_shifted_x = (curr_slam[0] - start_slam[0]) + start_cam_gt_x
                slam_shifted_y = (curr_slam[1] - start_slam[1]) + start_cam_gt_y

                dist = math.sqrt((slam_shifted_x - goal_x)**2 + (slam_shifted_y - goal_y)**2)
                if dist < Cfg.GOAL_TOLERANCE:
                    print(f"   [Arrived] 到達目標！誤差: {dist:.3f}m")
                    arrived = True
                
                vec_slam_x = slam_shifted_x
                vec_slam_y = slam_shifted_y
                
                err = math.sqrt((vec_gt_x - vec_slam_x)**2 + (vec_gt_y - vec_slam_y)**2)
                
                log_gt_x.append(vec_gt_x); log_gt_y.append(vec_gt_y)
                log_slam_x.append(vec_slam_x); log_slam_y.append(vec_slam_y)
                log_error.append(err); log_steps.append(total_steps)

            total_steps += 1
            step_count += 1
            
            if dones.any(): 
                print("[WARN] 機器人跌倒，重置環境...")
                obs, _ = rl_env.reset()

    print("\n[測試結束] 繪圖中...")
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(log_gt_x, log_gt_y, 'b-', label='Ground Truth (Cam Offset Corrected)')
    plt.plot(log_slam_x, log_slam_y, 'r--', label='SLAM Estimate (Shifted)')
    wx = [p[0] for p in waypoints]
    wy = [p[1] for p in waypoints]
    plt.scatter(wx, wy, c='g', marker='x', s=100, label='Nav2 Goals')
    plt.title('Hybrid Test: Floating Map -> RL Walk')
    plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.legend(); plt.grid(True); plt.axis('equal') 
    
    plt.subplot(1, 2, 2)
    plt.plot(log_steps, log_error, 'g-', label='ATE')
    plt.axvline(x=spin_end_step, color='r', linestyle='--', label='Switch to Walking')
    plt.title('Absolute Position Error')
    plt.legend(); plt.grid(True)
    
    filename = 'vslam_hybrid_result.png'
    plt.savefig(filename)
    print(f"圖表已儲存至: {filename}")

    slam_sub_mgr.close()
    nav_node.destroy_node()
    try: base_env.close()
    except: pass
    rclpy.shutdown()
    simulation_app.close()

if __name__ == "__main__":
    main()