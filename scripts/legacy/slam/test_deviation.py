import argparse
import torch
import os
import sys
import math
import numpy as np
from isaaclab.app import AppLauncher

# --- 1. Launcher ---
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

# --- 2. Imports ---
import rclpy
from rclpy.node import Node
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner

# Our modules
from spot_vslam.tasks.spot_vslam.spot_vslam_test_cfg import SpotSlamTestEnvCfg
from spot_vslam.assets.spot_with_camera import SPOT_CFG
from spot_vslam.managers.ros2_manager import Ros2Manager
from spot_vslam.managers.orb_slam_subscriber_manager import OrbSlamSubscriberManager

# ==========================================
# Settings
# ==========================================
MODEL_PATH = args_cli.checkpoint
if not MODEL_PATH or not os.path.isfile(MODEL_PATH):
    print(f"[ERROR] Low-level checkpoint not found: {MODEL_PATH!r}. Pass --checkpoint or set SPOT_VSLAM_LOW_LEVEL_CKPT.")
    sys.exit(1)
TEST_DURATION_SEC = 20.0  # test duration (s)
CMD_VEL_X = 0.4           # forward speed (m/s)
CMD_YAW   = 0.0           # yaw rate (rad/s)

# ==========================================
# Simple wrapper (same as before)
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
    
    # 1. Create the env
    env_cfg = SpotSlamTestEnvCfg()
    env_cfg.scene.robot = SPOT_CFG.copy()
    env_cfg.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
    
    # Make sure the floor has friction, or the robot slides
    # env_cfg.scene.terrain.terrain_type = "plane" 
    
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 2. Start the managers
    ros2_mgr = Ros2Manager(env_cfg.ros2, base_env)
    slam_sub_mgr = OrbSlamSubscriberManager(env_cfg.slam_subscriber, base_env)
    
    rl_env = RslRlWrapper(base_env, device="cuda:0")

    # 3. Load the policy
    runner = OnPolicyRunner(rl_env, {"seed": 42, "device": "cuda:0", "policy": {"class_name": "ActorCritic", "actor_hidden_dims": [512, 256, 128], "critic_hidden_dims": [512, 256, 128], "activation": "elu"}, "algorithm": {"class_name": "PPO"}}, log_dir=os.path.dirname(MODEL_PATH), device="cuda:0")
    runner.load(MODEL_PATH)
    policy = runner.get_inference_policy(device="cuda:0")

    # 4. Init
    obs, _ = rl_env.reset()
    ros2_mgr.reset(env_ids=range(base_env.num_envs))
    
    # Trajectory error log
    total_steps = int(TEST_DURATION_SEC / base_env.step_dt)
    trajectory_log = []

    print(f"[START] Constant-speed test: V_x={CMD_VEL_X} m/s, time={TEST_DURATION_SEC}s")

    for i in range(total_steps):
        # ---------------------------------------------------
        # Override the command by hand (hard-coded)
        # ---------------------------------------------------
        # No Nav2 here: a fixed velocity vector
        # Format: [lin_vel_x, lin_vel_y, ang_vel_z]
        target_vel = torch.tensor([[CMD_VEL_X, 0.0, CMD_YAW]], device=base_env.device)
        
        # Inject it into the env
        if base_env.command_manager:
            base_env.command_manager.set_command("base_velocity", target_vel)

        # ---------------------------------------------------
        # RL inference and stepping
        # ---------------------------------------------------
        with torch.inference_mode():
            actions = policy(obs)
        
        obs, _, dones, extras = rl_env.step(actions)
        
        # Update the managers
        dt = base_env.step_dt
        ros2_mgr.update(dt)
        slam_sub_mgr.update(dt)

        # ---------------------------------------------------
        # Collect data (error)
        # ---------------------------------------------------
        if i % 10 == 0: # log every 10 steps
            # 1. Ground truth
            gt_pos = base_env.scene["robot"].data.root_pos_w.torch[0, :3].cpu().numpy()
            
            # 2. SLAM estimate (from the manager)
            # Note: assumes slam_sub_mgr stores the latest slam_pose;
            # if not, add something like self.current_slam_pose to the manager
            slam_error = -1.0
            if "observations" in extras and "slam_info" in extras["observations"]:
                 # the observation term written earlier in the cfg
                 # for raw coordinates, OrbSlamSubscriberManager would need to expose them
                 pass

            # Example: distance back to the origin (assuming a straight-line test)
            # or simply print it
            print(f"Step {i} | GT Pos: ({gt_pos[0]:.2f}, {gt_pos[1]:.2f})")
            
            trajectory_log.append(gt_pos)

    print("[DONE] Test done")
    ros2_mgr.close()
    slam_sub_mgr.close()
    base_env.close()
    simulation_app.close()

if __name__ == "__main__":
    import argparse
    main()