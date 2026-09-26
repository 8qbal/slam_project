import argparse
from spot_vslam.assets import SPOT_VSLAM_USD_DIR
import os
import time
import psutil
import pandas as pd
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Load and play high-level PPO on top of frozen low-level policy.")
parser.add_argument("--low_level_task", type=str, required=True, help="Low-level Isaac task")
parser.add_argument("--low_level_checkpoint", type=str, required=True, help="Path to low-level rl-games .pth checkpoint")
parser.add_argument("--high_level_checkpoint", type=str, required=True, help="Path to high-level SB3 .zip checkpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Use 1 for visualization")
parser.add_argument("--steps", type=int, default=1000, help="Number of play steps")
parser.add_argument("--deterministic", action="store_true", help="Use deterministic high-level policy")
parser.add_argument("--enable_dense_mapping", action="store_true", help="Enable Open3D TSDF dense mapping")
parser.add_argument("--dense_export_every", type=int, default=1000, help="Export mesh/pcd every N frames")
parser.add_argument("--dense_export_dir", type=str, default="./dense_map_output", help="Dense map export directory")
parser.add_argument("--dense_live_vis", action="store_true", help="Live visualize TSDF map")
parser.add_argument("--dense_vis_mesh", action="store_true", help="Visualize mesh instead of point cloud")

# ==========================================
# Experiment 2: 動態場景與目標點設定參數
# ==========================================
parser.add_argument("--scene_usd", type=str, default=None, help="Path to USD environment (e.g. Simple Room)")
parser.add_argument("--target_x", type=float, default=-4.0, help="Target X coordinate for success evaluation")
parser.add_argument("--target_y", type=float, default=15.5, help="Target Y coordinate for success evaluation")
parser.add_argument("--success_radius", type=float, default=1.0, help="Radius (meters) to consider target reached")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import isaaclab_tasks  # noqa: F401
import spot_vslam.tasks  # noqa: F401
from sklearn.linear_model import LinearRegression

try:
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    ISAAC_ASSETS_PATH = ISAAC_NUCLEUS_DIR
except ImportError:
    ISAAC_ASSETS_PATH = (
        "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5"
    )

print(f"[INFO] Using Isaac Assets Path: {ISAAC_ASSETS_PATH}")

simple_room_url = (
    f"{ISAAC_ASSETS_PATH}/Environments/Simple_Warehouse/warehouse.usd"
)

corrider_url = f"{SPOT_VSLAM_USD_DIR}/corrider_map.usd"

warehouse_url = (
    f"{ISAAC_ASSETS_PATH}/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
)

from spot_vslam.managers.ros2_manager import Ros2Manager
from spot_vslam.high_level.high_level_vec_env import (
    HighLevelEnvCfg,
    HighLevelIsaacVecEnv,
)

from spot_vslam.managers.dense_map_manager import (
    DenseMapManager,
    DenseMapManagerCfg,
)

import numpy

class OrbGtComparator:
    """Compare ORB-SLAM pose and Isaac Sim GT pose in a shared relative frame."""
    def __init__(self):
        self.initialized = False
        self.orb_origin_xyyaw = None
        self.gt_origin_xyyaw = None
        self.orb_buffer = []
        self.gt_buffer = []
        self.calibrated = False
        self.a = None
        self.b = None
        self.tx = None
        self.ty = None

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def quat_xyzw_to_yaw(quat_xyzw: np.ndarray) -> float:
        x, y, z, w = quat_xyzw
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def get_gt_xyyaw(self, env_unwrapped) -> np.ndarray:
        # 取得機器人本體位置並計算 Yaw 角
        pos = env_unwrapped.scene["robot"].data.root_pos_w.torch[0].detach().cpu().numpy()
        quat = env_unwrapped.scene["robot"].data.root_quat_w.torch[0].detach().cpu().numpy()
        yaw = self.quat_xyzw_to_yaw(quat)
        
        # 座標轉換：將 GT 從「機器人中心」推到「相機中心」(前方 0.4m)
        cam_x = pos[0] + 0.4 * np.cos(yaw)
        cam_y = pos[1] + 0.4 * np.sin(yaw)
        
        return np.array([cam_x, cam_y, yaw], dtype=np.float64)

    def try_initialize(self, env_unwrapped) -> bool:
        if self.initialized:
            return True
        if not hasattr(env_unwrapped, "orb_slam_res"):
            return False

        res = env_unwrapped.orb_slam_res
        pose_xyyaw = res.get("pose_xyyaw", None)
        status = res.get("status", None)

        if pose_xyyaw is None or status is None:
            return False

        orb_status = float(status[0].detach().cpu().numpy()[0])
        if orb_status < 0.5:
            return False

        orb_xyyaw = pose_xyyaw[0].detach().cpu().numpy().astype(np.float64)
        gt_xyyaw = self.get_gt_xyyaw(env_unwrapped)

        self.orb_origin_xyyaw = orb_xyyaw.copy()
        self.gt_origin_xyyaw = gt_xyyaw.copy()
        self.initialized = True

        print("[ALIGN] ORB tracking acquired. Set this moment as shared origin.")
        print("[ALIGN] ORB origin xyyaw:", self.orb_origin_xyyaw)
        print("[ALIGN] GT  origin xyyaw:", self.gt_origin_xyyaw)
        return True
    
    def estimate_se2(self, orb_list, gt_list):
        """
        Estimate:
            GT = R * ORB + t
        using least squares
        """

        orb = np.array(orb_list)
        gt = np.array(gt_list)

        A = []
        B = []

        for o, g in zip(orb, gt):
            x, y = o[0], o[1]

            A.append([x, -y, 1, 0])
            A.append([y,  x, 0, 1])

            B.append(g[0])
            B.append(g[1])

        A = np.array(A)
        B = np.array(B)

        sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

        a, b, tx, ty = sol
        return a, b, tx, ty

    def compute_relative_error(self, env_unwrapped):

        if not self.initialized:
            ok = self.try_initialize(env_unwrapped)
            if not ok:
                return None

        res = env_unwrapped.orb_slam_res
        pose_xyyaw = res.get("pose_xyyaw", None)
        status = res.get("status", None)

        if pose_xyyaw is None or status is None:
            return None

        orb_status = float(status[0].detach().cpu().numpy()[0])
        if orb_status < 0.5:
            return None

        orb_now = pose_xyyaw[0].detach().cpu().numpy().astype(np.float64)
        gt_now = self.get_gt_xyyaw(env_unwrapped)

        print("Orb:", orb_now)
        print("Gt:", gt_now)

        # buffer (保留但不做 SE2)
        self.orb_buffer.append(orb_now[:2].copy())
        self.gt_buffer.append(gt_now[:2].copy())

        # =========================
        # 原始 yaw / xy（不做 SE2）
        # =========================
        orb_xy = np.array([orb_now[0], orb_now[1]])
        yaw = orb_now[2]

        orb_aligned = np.array([orb_xy[0], orb_xy[1], yaw])

        # ground truth relative
        gt_rel = gt_now - self.gt_origin_xyyaw
        gt_rel[2] = self.wrap_to_pi(gt_rel[2])

        err = orb_aligned - gt_rel
        err[2] = self.wrap_to_pi(err[2])

        return {
            "orb_rel": orb_aligned,
            "gt_rel": gt_rel,
            "err": err,
            "status": orb_status
        }


def save_plots(out_dir, t, pos_e, yaw_e, ox, oy, gx, gy):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(t, pos_e)
    plt.title("Position Error")
    plt.savefig(f"{out_dir}/pos_error.png")
    plt.close()

    plt.figure()
    plt.plot(t, yaw_e)
    plt.title("Yaw Error")
    plt.savefig(f"{out_dir}/yaw_error.png")
    plt.close()

    plt.figure()
    plt.plot(gx, gy, label="GT")
    plt.plot(ox, oy, label="ORB")
    plt.legend()
    plt.title("Trajectory")
    plt.axis("equal")
    plt.savefig(f"{out_dir}/trajectory.png")
    plt.close()


def build_low_level():
    env_cfg = parse_env_cfg(
        args_cli.low_level_task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
    )

    if args_cli.scene_usd is not None:
        print("[INFO] Custom indoor USD scene enabled")

        # =====================================================
        # 1. 關閉地形難度遞增 (Curriculum)
        # =====================================================
        if hasattr(env_cfg, "curriculum"):
            env_cfg.curriculum = None

        # =====================================================
        # 2. 關閉與地形相關的終止條件 (避免不小心觸發 Done)
        # =====================================================
        if hasattr(env_cfg, "terminations"):
            for term_name in list(vars(env_cfg.terminations).keys()):
                term_cfg = getattr(env_cfg.terminations, term_name)
                if term_cfg is None:
                    continue
                if "terrain" in term_name.lower():
                    print(f"[INFO] Disable termination: {term_name}")
                    setattr(env_cfg.terminations, term_name, None)

        # =====================================================
        # 3. 解析 USD 路徑
        # =====================================================
        scene_map = {
            "simple_room": simple_room_url,
            "corrider": corrider_url,
            "warehouse": warehouse_url,
        }
        actual_usd_path = scene_map.get(
            args_cli.scene_usd.lower(),
            args_cli.scene_usd
        )
        print(f"[INFO] Using custom USD scene: {actual_usd_path}")

        # =====================================================
        # 4. 【關鍵修改】直接把 USD 指定為地形！
        # =====================================================
        if hasattr(env_cfg.scene, "terrain"):
            # 告訴 Isaac Lab：不要生成 plane，直接使用 usd 類型
            env_cfg.scene.terrain.terrain_type = "usd"
            # 指定 USD 檔案的路徑
            env_cfg.scene.terrain.usd_path = actual_usd_path
            
            # 注意：我們不再需要 indoor_scene 了，因為場景本身就是 Terrain！

    agent_cfg = load_cfg_from_registry(args_cli.low_level_task, "rl_games_cfg_entry_point")

    resume_path = os.path.expanduser(args_cli.low_level_checkpoint)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Low-level checkpoint not found: {resume_path}")

    print("[INFO] Loading low-level checkpoint:", resume_path)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))

    # 直接建立環境，不需要再去刪除 Prim 了！
    env = gym.make(args_cli.low_level_task, cfg=env_cfg)
    
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)

    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    if getattr(env.unwrapped.cfg, "ros2", None) is not None:
        env.unwrapped.ros2_manager = Ros2Manager(env.unwrapped.cfg.ros2, env.unwrapped)

    print("[INFO] low-level num_envs =", env.unwrapped.num_envs)
    return env, agent


def save_error_plots(out_dir, timesteps, pos_errors, yaw_errors, orb_traj_x, orb_traj_y, gt_traj_x, gt_traj_y, suffix=""):
    os.makedirs(out_dir, exist_ok=True)
    if len(pos_errors) == 0:
        print("[WARN] No valid ORB tracking data. Plots were not generated.")
        return

    suffix_str = f"_{suffix}" if suffix else ""

    # Position error
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, pos_errors)
    plt.xlabel("Step")
    plt.ylabel("Position Error (m)")
    plt.title("ORB vs GT Position Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"position_error{suffix_str}.png"), dpi=200)
    plt.close()

    # Yaw error
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, yaw_errors)
    plt.xlabel("Step")
    plt.ylabel("Yaw Error (deg)")
    plt.title("ORB vs GT Yaw Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"yaw_error{suffix_str}.png"), dpi=200)
    plt.close()

    # Trajectory overlay
    plt.figure(figsize=(7, 7))
    plt.plot(gt_traj_x, gt_traj_y, label="GT relative trajectory")
    plt.plot(orb_traj_x, orb_traj_y, label="ORB relative trajectory")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("ORB vs GT Relative Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"trajectory_overlay{suffix_str}.png"), dpi=200)
    plt.close()

    print(f"[INFO] Saved evaluation plots to: {out_dir} (suffix='{suffix}')")


def save_performance_plots(csv_path, out_dir, suffix="", show_plot=False):
    """讀取 CSV 數據並自動生成 CPU, RAM, FPS, Disk I/O 的趨勢圖"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    if not os.path.exists(csv_path):
        print(f"[WARN] CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    mode_name = os.path.basename(csv_path).replace('.csv', '')
    mode_name_clean = mode_name.split("_step_")[0].split("_final")[0]
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. CPU 使用率 & Disk I/O (雙 Y 軸)
    line1 = axs[0].plot(df['Step'], df['CPU_Usage_Percent'], color='tab:red', linewidth=2, label='CPU (%)')
    axs[0].set_title(f'System Performance Over Time ({mode_name_clean})')
    axs[0].set_ylabel('CPU Usage (%)')
    axs[0].grid(True, linestyle='--', alpha=0.7)

    if 'Disk_Read_MBs' in df.columns and 'Disk_Write_MBs' in df.columns:
        ax0_twin = axs[0].twinx()
        line2 = ax0_twin.plot(df['Step'], df['Disk_Read_MBs'], color='tab:orange', linewidth=1.5, linestyle=':', label='Read (MB/s)')
        line3 = ax0_twin.plot(df['Step'], df['Disk_Write_MBs'], color='tab:purple', linewidth=1.5, linestyle='--', label='Write (MB/s)')
        ax0_twin.set_ylabel('Disk I/O (MB/s)')
        
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        axs[0].legend(lines, labels, loc='upper left')
    else:
        axs[0].legend(loc='upper left')

    # 2. RAM 使用量
    axs[1].plot(df['Step'], df['RAM_Usage_GB'], color='tab:blue', linewidth=2)
    axs[1].set_title('System RAM Usage Over Time')
    axs[1].set_ylabel('RAM Usage (GB)')
    axs[1].grid(True, linestyle='--', alpha=0.7)

    # 3. 系統 FPS
    axs[2].plot(df['Step'], df['System_FPS'], color='tab:green', linewidth=2)
    axs[2].set_title('System Processing Speed (FPS)')
    axs[2].set_xlabel('Simulation Step')
    axs[2].set_ylabel('FPS (Hz)')
    axs[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    suffix_str = f"_{suffix}" if suffix else ""
    plot_path = os.path.join(out_dir, f"{mode_name_clean}_charts{suffix_str}.png")
    
    plt.savefig(plot_path, dpi=200)
    print(f"[INFO] Performance charts saved to {plot_path}")
    
    if show_plot:
        plt.show() 
    
    plt.close(fig)


def main():
    low_env, low_agent = build_low_level()

    hl_cfg = HighLevelEnvCfg(
        hl_decimation=2,
        cmd_smoothing_alpha=0.25,
        vx_min=0.0,
        vx_max=0.5,
        wz_min=-0.25,
        wz_max=0.25,
        max_episode_hl_steps=400,
    )

    vec_env = HighLevelIsaacVecEnv(low_env, low_agent, hl_cfg)
    orb_gt_comparator = OrbGtComparator()

    # =========================
    # ORB-GT logs
    # =========================
    timesteps, pos_errors, yaw_errors = [], [], []
    orb_traj_x, orb_traj_y, gt_traj_x, gt_traj_y = [], [], [], []

    plot_dir = os.path.join(args_cli.dense_export_dir, "eval_plots")
    autosave_every = 2050

    # =========================
    # Dense mapping
    # =========================
    dense_map_manager = None
    if args_cli.enable_dense_mapping:
        dense_map_manager = DenseMapManager(
            DenseMapManagerCfg(
                camera_name="tilted_camera",
                voxel_length=0.02,
                sdf_trunc=0.05,
                depth_scale=1.0,
                depth_max=5.0,
                export_dir=args_cli.dense_export_dir,
                export_every_n_frames=args_cli.dense_export_every,
                use_color=True,
                width=320,
                height=240,
                fx=145.4545,
                fy=145.4545,
                cx=160.0,
                cy=120.0,
                cam_offset_pos=(0.4, 0.0, 0.0),
                cam_offset_quat_xyzw=(0.5, -0.5, 0.5, -0.5),
                pose_source="orb",
                debug_pose=True,
                enable_live_vis=args_cli.dense_live_vis,
                vis_update_every_n_frames=10,
                vis_as_mesh=args_cli.dense_vis_mesh,
                window_name="Spot Dense TSDF Map",
                width_vis=1280,
                height_vis=720,
            ),
            low_env.unwrapped,
        )

        print("[INFO] Dense TSDF enabled")

    # =========================
    # Load model
    # =========================
    high_level_ckpt = os.path.expanduser(args_cli.high_level_checkpoint)
    print("[INFO] Loading:", high_level_ckpt)
    model = PPO.load(high_level_ckpt, env=vec_env, device="cpu")

    obs = vec_env.reset()

    # =========================
    # performance logs
    # =========================
    sys_cpu_log, sys_ram_log, fps_log = [], [], []
    sys_disk_read_log, sys_disk_write_log = [], []

    last_time = time.time()
    last_disk_io = psutil.disk_io_counters()
    record_interval = 10

    # =========================
    # task metrics
    # =========================
    episode_collisions = 0
    collision_cooldown = 0

    device = low_env.unwrapped.scene["robot"].data.root_pos_w.torch.device
    target_pos = torch.tensor(
        [args_cli.target_x, args_cli.target_y, 0.0],
        device=device
    )

    start_time_exp2 = time.time()
    is_success = False
    time_to_goal = -1
    prev_pos = None

    print("\n[INFO] Start rollout")

    try:
        for step in range(args_cli.steps):

            action, _ = model.predict(obs, deterministic=args_cli.deterministic)
            obs, rewards, dones, infos = vec_env.step(action)

            # =========================
            # system log
            # =========================
            if step % record_interval == 0 and step > 0:
                now = time.time()
                io = psutil.disk_io_counters()

                elapsed = now - last_time
                fps = record_interval / elapsed

                sys_cpu_log.append(psutil.cpu_percent())
                sys_ram_log.append(psutil.virtual_memory().used / (1024**3))
                fps_log.append(fps)

                sys_disk_read_log.append(
                    (io.read_bytes - last_disk_io.read_bytes) / (1024*1024*elapsed)
                )
                sys_disk_write_log.append(
                    (io.write_bytes - last_disk_io.write_bytes) / (1024*1024*elapsed)
                )

                last_time = now
                last_disk_io = io

            # =========================
            # dense map
            # =========================
            if dense_map_manager is not None:
                dense_map_manager.update()

            # =========================
            # collision detection
            # =========================
            robot = low_env.unwrapped.scene["robot"]
            acc = robot.data.body_acc_w.torch[0, 0, :2]
            force = torch.norm(acc).item()

            if force > 10.0 and collision_cooldown <= 0:
                episode_collisions += 1
                collision_cooldown = 15

            if collision_cooldown > 0:
                collision_cooldown -= 1

            # =========================
            # respawn / teleport detection
            # =========================
            current_pos = robot.data.root_pos_w.torch[0][:2].detach().cpu()

            if prev_pos is not None:
                jump_dist = torch.norm(current_pos - prev_pos).item()

                if jump_dist > 2.0:
                    episode_collisions += 1

                    print(
                        f"[FAILURE] Respawn detected "
                        f"(jump distance = {jump_dist:.2f} m)"
                    )

                    print("[MISSION FAILED]")
                    break

            prev_pos = current_pos.clone()

            # =========================
            # success detection
            # =========================
            pos = robot.data.root_pos_w.torch[0][:2]
            dist = torch.norm(pos - target_pos[:2]).item()

            if (not is_success) and dist < args_cli.success_radius:
                is_success = True
                time_to_goal = time.time() - start_time_exp2
                print(f"[SUCCESS] step={step}, time={time_to_goal:.2f}s")

            # =========================
            # ORB vs GT
            # =========================
            res = orb_gt_comparator.compute_relative_error(low_env.unwrapped)

            if res is not None:
                orb = res["orb_rel"]
                gt = res["gt_rel"]
                err = res["err"]

                timesteps.append(step)
                pos_errors.append(np.linalg.norm(err[:2]))
                yaw_errors.append(np.degrees(err[2]))

                orb_traj_x.append(orb[0])
                orb_traj_y.append(orb[1])
                gt_traj_x.append(gt[0])
                gt_traj_y.append(gt[1])

            # =========================
            # print status
            # =========================
            info0 = infos[0] if len(infos) > 0 else {}
            print(f"[STEP {step}] reward={float(rewards[0]):.3f} done={bool(dones[0])}")

            if np.any(dones):
                print("[RESET] episode done")

    except KeyboardInterrupt:
        print("\n[STOP] interrupted")

    # =========================
    # summary
    # =========================
    print("\n====== RESULT ======")
    print("Success:", is_success)
    print("Collisions:", episode_collisions)
    print("Time to goal:", time_to_goal)
    print("====================\n")

    # =========================
    # save CSV (experiment log)
    # =========================
    os.makedirs(args_cli.dense_export_dir, exist_ok=True)

    pd.DataFrame([{
        "success": int(is_success),
        "collisions": episode_collisions,
        "time_to_goal": time_to_goal,
        "steps": args_cli.steps
    }]).to_csv(
        os.path.join(args_cli.dense_export_dir, "experiment_log.csv"),
        index=False
    )

    # =========================
    # save plots
    # =========================
    save_error_plots(
        plot_dir,
        timesteps,
        pos_errors,
        yaw_errors,
        orb_traj_x,
        orb_traj_y,
        gt_traj_x,
        gt_traj_y
    )

    # =========================
    # cleanup
    # =========================
    if dense_map_manager is not None:
        dense_map_manager.close()

    vec_env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()