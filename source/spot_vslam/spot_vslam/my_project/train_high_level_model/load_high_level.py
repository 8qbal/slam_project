import argparse
import os
import time
import psutil
import pandas as pd

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Load and play high-level PPO on top of frozen low-level policy.")
parser.add_argument("--low_level_task", type=str, required=True, help="Low-level Isaac task")
parser.add_argument("--low_level_checkpoint", type=str, required=True, help="Path to low-level rl-games .pth checkpoint")
parser.add_argument("--high_level_checkpoint", type=str, required=True, help="Path to high-level SB3 .zip checkpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Use 1 for visualization")
parser.add_argument("--steps", type=int, default=5000, help="Number of play steps")
parser.add_argument("--deterministic", action="store_true", help="Use deterministic high-level policy")
parser.add_argument("--enable_dense_mapping", action="store_true", help="Enable Open3D TSDF dense mapping")
parser.add_argument("--dense_export_every", type=int, default=1000, help="Export mesh/pcd every N frames")
parser.add_argument("--dense_export_dir", type=str, default="./dense_map_output", help="Dense map export directory")
parser.add_argument("--dense_live_vis", action="store_true", help="Live visualize TSDF map")
parser.add_argument("--dense_vis_mesh", action="store_true", help="Visualize mesh instead of point cloud")
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

from spot_vslam.my_project.manager.ros2_manager import Ros2Manager
from spot_vslam.my_project.test_high_level_model.high_level_vec_env import (
    HighLevelEnvCfg,
    HighLevelIsaacVecEnv,
)

from spot_vslam.my_project.manager.dense_map_manager import (
    DenseMapManager,
    DenseMapManagerCfg,
)


class OrbGtComparator:
    """
    Compare ORB-SLAM pose and Isaac Sim GT pose in a shared relative frame.

    Shared origin:
    - the first timestep when ORB tracking becomes valid.
    """

    def __init__(self):
        self.initialized = False
        self.orb_origin_xyyaw = None
        self.gt_origin_xyyaw = None

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
        w, x, y, z = quat_wxyz
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def get_gt_xyyaw(self, env_unwrapped) -> np.ndarray:
            # 1. 取得機器人本體 (Base) 的世界位置與四元數
            pos = env_unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
            quat = env_unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
            
            # 2. 計算機器人本體的 Yaw 角 (朝向)
            yaw = self.quat_wxyz_to_yaw(quat)
            
            # 3. 座標轉換：將 GT 基準點從「機器人中心」平移到「相機中心」
            # 相機在機器人本體座標系的 X 軸前方 0.4 公尺處
            cam_x = pos[0] + 0.4 * np.cos(yaw)
            cam_y = pos[1] + 0.4 * np.sin(yaw)
            
            # 回傳相機真實的 Ground Truth 座標與朝向
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

        orb_rel = orb_now - self.orb_origin_xyyaw
        gt_rel = gt_now - self.gt_origin_xyyaw

        orb_rel[2] = self.wrap_to_pi(orb_rel[2])
        gt_rel[2] = self.wrap_to_pi(gt_rel[2])

        err = orb_rel - gt_rel
        err[2] = self.wrap_to_pi(err[2])

        return {
            "orb_rel": orb_rel,
            "gt_rel": gt_rel,
            "err": err,
            "status": orb_status,
        }


def build_low_level():
    env_cfg = parse_env_cfg(
        args_cli.low_level_task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
    )

    agent_cfg = load_cfg_from_registry(args_cli.low_level_task, "rl_games_cfg_entry_point")

    resume_path = os.path.expanduser(args_cli.low_level_checkpoint)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Low-level checkpoint not found: {resume_path}")

    print("[INFO] Loading low-level checkpoint:", resume_path)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))

    env = gym.make(args_cli.low_level_task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
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


def save_error_plots(
    out_dir: str,
    timesteps,
    pos_errors,
    yaw_errors,
    orb_traj_x,
    orb_traj_y,
    gt_traj_x,
    gt_traj_y,
    suffix: str = "",
):
    os.makedirs(out_dir, exist_ok=True)

    if len(pos_errors) == 0:
        print("[WARN] No valid ORB tracking data. Plots were not generated.")
        return

    suffix = f"_{suffix}" if suffix else ""

    # Position error
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, pos_errors)
    plt.xlabel("Step")
    plt.ylabel("Position Error (m)")
    plt.title("ORB vs GT Position Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"position_error{suffix}.png"), dpi=200)
    plt.close()

    # Yaw error
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, yaw_errors)
    plt.xlabel("Step")
    plt.ylabel("Yaw Error (deg)")
    plt.title("ORB vs GT Yaw Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"yaw_error{suffix}.png"), dpi=200)
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
    plt.savefig(os.path.join(out_dir, f"trajectory_overlay{suffix}.png"), dpi=200)
    plt.close()

    print("[INFO] Saved evaluation plots to:", out_dir, f"(suffix='{suffix}')")
    print(f"[EVAL] Mean position error: {np.mean(pos_errors):.3f} m")
    print(f"[EVAL] Max position error:  {np.max(pos_errors):.3f} m")
    print(f"[EVAL] Mean yaw error:      {np.mean(yaw_errors):.2f} deg")
    print(f"[EVAL] Max yaw error:       {np.max(yaw_errors):.2f} deg")



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

    # 1. 畫 CPU 使用率 (左邊的 Y 軸)
    line1 = axs[0].plot(df['Step'], df['CPU_Usage_Percent'], color='tab:red', linewidth=2, label='CPU (%)')
    axs[0].set_title(f'System Performance Over Time ({mode_name_clean})')
    axs[0].set_ylabel('CPU Usage (%)')
    axs[0].grid(True, linestyle='--', alpha=0.7)

    # [新增] 建立雙 Y 軸來畫硬碟讀寫 (右邊的 Y 軸，單位 MB/s)
    if 'Disk_Read_MBs' in df.columns and 'Disk_Write_MBs' in df.columns:
        ax0_twin = axs[0].twinx()
        line2 = ax0_twin.plot(df['Step'], df['Disk_Read_MBs'], color='tab:orange', linewidth=1.5, linestyle=':', label='Read (MB/s)')
        line3 = ax0_twin.plot(df['Step'], df['Disk_Write_MBs'], color='tab:purple', linewidth=1.5, linestyle='--', label='Write (MB/s)')
        ax0_twin.set_ylabel('Disk I/O (MB/s)')
        
        # 合併左軸與右軸的圖例 (Legend)
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        axs[0].legend(lines, labels, loc='upper left')
    else:
        axs[0].legend(loc='upper left')

    # 2. 畫 RAM 使用量
    axs[1].plot(df['Step'], df['RAM_Usage_GB'], color='tab:blue', linewidth=2)
    axs[1].set_title('System RAM Usage Over Time')
    axs[1].set_ylabel('RAM Usage (GB)')
    axs[1].grid(True, linestyle='--', alpha=0.7)

    # 3. 畫 系統 FPS
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
    
    # if show_plot:
    #     plt.show() 
    
    # plt.close(fig) # 務必關閉，否則記憶體會爆掉


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

    # Error logs
    timesteps = []
    pos_errors = []
    yaw_errors = []
    orb_traj_x = []
    orb_traj_y = []
    gt_traj_x = []
    gt_traj_y = []

    plot_dir = os.path.join(args_cli.dense_export_dir, "eval_plots")
    autosave_every = 2050

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
                pose_source="orb",   # change to "gt" if you want to verify TSDF first
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
        print("[INFO] Dense TSDF mapping enabled")
        print("[INFO] Dense map export dir:", os.path.abspath(args_cli.dense_export_dir))

    high_level_ckpt = os.path.expanduser(args_cli.high_level_checkpoint)
    if not os.path.isfile(high_level_ckpt):
        raise FileNotFoundError(f"High-level checkpoint not found: {high_level_ckpt}")

    print("[INFO] Loading high-level checkpoint:", high_level_ckpt)
    model = PPO.load(high_level_ckpt, env=vec_env, device="cpu")

    obs = vec_env.reset()
    print("[INFO] Start rollout")
    print("[INFO] num_envs =", vec_env.num_envs)

    sys_cpu_log = []
    sys_ram_log = []
    fps_log = []
    last_time = time.time()
    record_interval = 10  # 每 10 個 step 記錄一次
    sys_disk_read_log = []   # [新增] 記錄硬碟讀取率
    sys_disk_write_log = []  # [新增] 記錄硬碟寫入率

    last_time = time.time()
    last_disk_io = psutil.disk_io_counters()  # [新增] 獲取初始的硬碟 I/O 狀態
    record_interval = 10  # 每 10 個 step 記錄一次

    try:
        for step in range(args_cli.steps):
            action, _ = model.predict(obs, deterministic=args_cli.deterministic)
            obs, rewards, dones, infos = vec_env.step(action)

            # ----------------------------------------------------
            # [新增] 3. 迴圈內的效能紀錄 (加在 vec_env.step 之後)
            if step % record_interval == 0 and step > 0:
                current_time = time.time()
                current_disk_io = psutil.disk_io_counters() # [新增] 取得當下硬碟 I/O 狀態
                
                # 取得 CPU 與 RAM 數據
                cpu_percent = psutil.cpu_percent(interval=None)
                ram_gb = psutil.virtual_memory().used / (1024 ** 3)
                
                # 計算這 10 步的平均 FPS 與經過時間
                elapsed = current_time - last_time
                fps = record_interval / elapsed
                
                # [新增] 計算硬碟讀寫率 (轉為 MB/s)
                # disk_io_counters 記錄的是開機以來的總位元組(Bytes)，所以要相減並除以時間與 1MB
                read_mbs = (current_disk_io.read_bytes - last_disk_io.read_bytes) / (1024 * 1024 * elapsed)
                write_mbs = (current_disk_io.write_bytes - last_disk_io.write_bytes) / (1024 * 1024 * elapsed)
                
                sys_cpu_log.append(cpu_percent)
                sys_ram_log.append(ram_gb)
                fps_log.append(fps)
                sys_disk_read_log.append(read_mbs)   # [新增]
                sys_disk_write_log.append(write_mbs) # [新增]
                
                # 更新上一次的狀態
                last_time = current_time
                last_disk_io = current_disk_io       # [新增]
            # ----------------------------------------------------

            if dense_map_manager is not None:
                dense_map_manager.update()

            orb_gt_info = orb_gt_comparator.compute_relative_error(low_env.unwrapped)
            if orb_gt_info is not None:
                orb_rel = orb_gt_info["orb_rel"]
                gt_rel = orb_gt_info["gt_rel"]
                err = orb_gt_info["err"]

                pos_err = np.linalg.norm(err[:2])
                yaw_err_deg = np.degrees(err[2])

                timesteps.append(step)
                pos_errors.append(pos_err)
                yaw_errors.append(yaw_err_deg)

                orb_traj_x.append(orb_rel[0])
                orb_traj_y.append(orb_rel[1])
                gt_traj_x.append(gt_rel[0])
                gt_traj_y.append(gt_rel[1])

                print(
                    "[ORB-vs-GT] "
                    f"pos_err={pos_err:.3f} m, "
                    f"yaw_err={yaw_err_deg:.2f} deg, "
                    f"orb_rel=({orb_rel[0]: .3f}, {orb_rel[1]: .3f}, {orb_rel[2]: .3f}), "
                    f"gt_rel=({gt_rel[0]: .3f}, {gt_rel[1]: .3f}, {gt_rel[2]: .3f})"
                )
            else:
                print("[ORB-vs-GT] waiting for ORB tracking initialization...")

            # --- 定期自動存檔區塊 (縮排修正) ---
            if step > 0 and step % autosave_every == 0:
                # 1. 存軌跡圖
                save_error_plots(
                    out_dir=plot_dir, timesteps=timesteps,
                    pos_errors=pos_errors, yaw_errors=yaw_errors,
                    orb_traj_x=orb_traj_x, orb_traj_y=orb_traj_y,
                    gt_traj_x=gt_traj_x, gt_traj_y=gt_traj_y,
                    suffix=f"step_{step:06d}"
                )

                # 2. 存效能圖 (只有在有紀錄時才存)
                if len(sys_cpu_log) > 0:
                    mode_name = "rl_slam_mapping" if args_cli.enable_dense_mapping else "rl_slam"
                    df = pd.DataFrame({
                        'Step': range(record_interval, record_interval * len(sys_cpu_log) + 1, record_interval),
                        'CPU_Usage_Percent': sys_cpu_log,
                        'RAM_Usage_GB': sys_ram_log,
                        'System_FPS': fps_log,
                        'Disk_Read_MBs': sys_disk_read_log,   # <- 請確認是否有這行
                        'Disk_Write_MBs': sys_disk_write_log  # <- 請確認是否有這行
                    })
                    temp_csv = os.path.join(args_cli.dense_export_dir, f"performance_log_{mode_name}_step_{step:06d}.csv")
                    df.to_csv(temp_csv, index=False)
                    save_performance_plots(temp_csv, plot_dir, suffix=f"step_{step:06d}", show_plot=False)
                # ----------------------------------------------------

            # 只印第一個 env，觀察最直觀
            info0 = infos[0] if len(infos) > 0 else {}
            cmd_vx = info0.get("cmd_vx", None)
            cmd_wz = info0.get("cmd_wz", None)
            progress = info0.get("progress", None)
            stuck = info0.get("stuck", None)
            body_contact = info0.get("body_contact", None)
            tracking_bad = info0.get("tracking_bad", None)

            print(
                f"[STEP {step:05d}] "
                f"reward={float(rewards[0]): .4f} "
                f"done={bool(dones[0])} "
                f"cmd_vx={cmd_vx if cmd_vx is not None else 'NA'} "
                f"cmd_wz={cmd_wz if cmd_wz is not None else 'NA'} "
                f"progress={progress if progress is not None else 'NA'} "
                f"stuck={stuck if stuck is not None else 'NA'} "
                f"body_contact={body_contact if body_contact is not None else 'NA'} "
                f"tracking_bad={tracking_bad if tracking_bad is not None else 'NA'}"
            )

            if np.any(dones):
                print(f"[INFO] Episode ended at step {step}, continuing after reset inside VecEnv.")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected. Stopping rollout safely...")

    print("[INFO] Finished rollout")

    # ----------------------------------------------------
    # [新增] 4. 迴圈結束後，將收集到的數據存成 CSV
    mode_name = "rl_slam_mapping" if args_cli.enable_dense_mapping else "rl_slam"
    if len(sys_cpu_log) > 0:
        final_csv = os.path.join(args_cli.dense_export_dir, f"performance_log_{mode_name}_final.csv")
        df = pd.DataFrame({
            'Step': range(record_interval, record_interval * len(sys_cpu_log) + 1, record_interval),
            'CPU_Usage_Percent': sys_cpu_log,
            'RAM_Usage_GB': sys_ram_log,
            'System_FPS': fps_log,
            'Disk_Read_MBs': sys_disk_read_log,   # [新增]
            'Disk_Write_MBs': sys_disk_write_log  # [新增]
        })
        df.to_csv(final_csv, index=False)
        save_performance_plots(final_csv, plot_dir, suffix="final", show_plot=True)
    # ----------------------------------------------------

    # Final save
    save_error_plots(
        out_dir=plot_dir,
        timesteps=timesteps,
        pos_errors=pos_errors,
        yaw_errors=yaw_errors,
        orb_traj_x=orb_traj_x,
        orb_traj_y=orb_traj_y,
        gt_traj_x=gt_traj_x,
        gt_traj_y=gt_traj_y,
        suffix="final",
    )

    if dense_map_manager is not None: dense_map_manager.close()

    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()