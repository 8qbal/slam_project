import argparse
import os
import time

import pandas as pd
import psutil

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Load and play high-level PPO on top of frozen low-level policy.")
parser.add_argument("--low_level_task", type=str, required=True, help="Low-level Isaac task")
parser.add_argument(
    "--low_level_checkpoint", type=str, required=True, help="Path to low-level rl-games .pth checkpoint"
)
parser.add_argument("--high_level_checkpoint", type=str, required=True, help="Path to high-level SB3 .zip checkpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Use 1 for visualization")
parser.add_argument("--steps", type=int, default=5000, help="Number of play steps")
parser.add_argument("--seed", type=int, default=None, help="Env seed (use a different one per trial)")
parser.add_argument("--deterministic_policy", action="store_true", help="Use deterministic high-level policy")
parser.add_argument("--enable_dense_mapping", action="store_true", help="Enable Open3D TSDF dense mapping")
parser.add_argument("--dense_export_every", type=int, default=1000, help="Export mesh/pcd every N frames")
parser.add_argument("--dense_export_dir", type=str, default="./output/dense_map", help="Dense map export directory")
parser.add_argument("--dense_live_vis", action="store_true", help="Live visualize TSDF map")
parser.add_argument("--dense_vis_mesh", action="store_true", help="Visualize mesh instead of point cloud")
parser.add_argument(
    "--dense_pose_source",
    choices=["orb", "gt"],
    default="orb",
    help="Camera pose for TSDF integration: ORB-SLAM3 estimate (integrates only tracked frames) or simulator GT",
)
parser.add_argument(
    "--paper",
    action="store_true",
    help="Paper policy (9-dim goal obs) on a Spot-Paper-high-level-Map*-Play-v0 task; the goal is the map's goal",
)
parser.add_argument(
    "--navigation_test",
    action="store_true",
    help="With --paper: one run from the map start to the goal (or 3000 steps), appended to --results_csv, "
    "then Table 9 is printed. Without it the policy keeps running for --steps (Tables 7-8).",
)
parser.add_argument(
    "--results_csv", type=str, default="./navigation_results.csv", help="--navigation_test: one row per run is appended"
)
parser.add_argument(
    "--log_every", type=int, default=1, help="Print the per-step [STEP] / [ORB-vs-GT] lines every N steps (0 = never)"
)
parser.add_argument(
    "--no_ros2", action="store_true", help="Don't start ROS 2 (no images published, no SLAM): the 'RL only' setup"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

import isaaclab_tasks  # noqa: F401

import spot_vslam.tasks  # noqa: F401
from spot_vslam.assets import PAPER_MAPS
from spot_vslam.legacy.high_level.high_level_vec_env import (
    HighLevelEnvCfg,
    HighLevelIsaacVecEnv,
)
from spot_vslam.legacy.high_level.low_level_runner import build_low_level as build_low_level_policy
from spot_vslam.legacy.high_level.paper_vec_env import PaperHighLevelEnvCfg, PaperHighLevelIsaacVecEnv
from spot_vslam.slam.dense_map import (
    DenseMapManager,
    DenseMapManagerCfg,
)
from spot_vslam.utils import yaw_from_quat_xyzw


class OrbGtComparator:
    """
    Compare ORB-SLAM pose and Isaac Sim GT pose in a shared relative frame.

    Shared origin:
    - the first timestep when ORB tracking becomes valid.

    Both trajectories are expressed in the body frame at that moment (translation rotated by -yaw0), because the
    ORB-SLAM3 node publishes poses relative to its first tracked camera pose (yaw 0), while GT yaw is in the world
    frame and is randomized on reset.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Forget the shared origin; call when the episode resets (the robot is teleported)."""
        self.initialized = False
        self.orb_origin_xyyaw = None
        self.gt_origin_xyyaw = None

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @classmethod
    def to_origin_frame(cls, xyyaw: np.ndarray, origin_xyyaw: np.ndarray) -> np.ndarray:
        dx, dy = xyyaw[:2] - origin_xyyaw[:2]
        c, s = np.cos(origin_xyyaw[2]), np.sin(origin_xyyaw[2])
        return np.array([c * dx + s * dy, -s * dx + c * dy, cls.wrap_to_pi(xyyaw[2] - origin_xyyaw[2])])

    def get_gt_xyyaw(self, env_unwrapped) -> np.ndarray:
        # 1. World position and quaternion of the robot base
        pos = env_unwrapped.scene["robot"].data.root_pos_w.torch[0].detach().cpu().numpy()
        quat = env_unwrapped.scene["robot"].data.root_quat_w.torch[0].detach().cpu().numpy()

        # 2. Base yaw (heading)
        yaw = yaw_from_quat_xyzw(quat)

        # 3. Move the GT reference point from the robot center to the camera center:
        # the camera is 0.4 m ahead along the base x axis
        cam_x = pos[0] + 0.4 * np.cos(yaw)
        cam_y = pos[1] + 0.4 * np.sin(yaw)

        # Return the camera's ground-truth position and heading
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

        orb_rel = self.to_origin_frame(orb_now, self.orb_origin_xyyaw)
        gt_rel = self.to_origin_frame(gt_now, self.gt_origin_xyyaw)

        err = orb_rel - gt_rel
        err[2] = self.wrap_to_pi(err[2])

        return {
            "orb_rel": orb_rel,
            "gt_rel": gt_rel,
            "err": err,
            "status": orb_status,
        }


def build_low_level():
    resume_path = os.path.expanduser(args_cli.low_level_checkpoint)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Low-level checkpoint not found: {resume_path}")
    return build_low_level_policy(
        args_cli.low_level_task, resume_path, args_cli.num_envs, seed=args_cli.seed, ros2=not args_cli.no_ros2
    )


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

    if len(pos_errors) == 0 or np.all(np.isnan(pos_errors)):
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
    # NaN entries separate episodes (see the reset handling in main)
    print(f"[EVAL] Mean position error: {np.nanmean(pos_errors):.3f} m")
    print(f"[EVAL] Max position error:  {np.nanmax(pos_errors):.3f} m")
    print(f"[EVAL] Mean |yaw error|:    {np.nanmean(np.abs(yaw_errors)):.2f} deg")
    print(f"[EVAL] Max |yaw error|:     {np.nanmax(np.abs(yaw_errors)):.2f} deg")


def save_performance_plots(csv_path, out_dir, suffix="", show_plot=False):
    """Read the CSV log and plot CPU, RAM, FPS and disk I/O over time."""
    import os

    import matplotlib.pyplot as plt
    import pandas as pd

    if not os.path.exists(csv_path):
        print(f"[WARN] CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    mode_name = os.path.basename(csv_path).replace(".csv", "")
    mode_name_clean = mode_name.split("_step_")[0].split("_final")[0]

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. CPU usage (left y axis)
    line1 = axs[0].plot(df["Step"], df["CPU_Usage_Percent"], color="tab:red", linewidth=2, label="CPU (%)")
    axs[0].set_title(f"System Performance Over Time ({mode_name_clean})")
    axs[0].set_ylabel("CPU Usage (%)")
    axs[0].grid(True, linestyle="--", alpha=0.7)

    # Second y axis for disk read/write (right, MB/s)
    if "Disk_Read_MBs" in df.columns and "Disk_Write_MBs" in df.columns:
        ax0_twin = axs[0].twinx()
        line2 = ax0_twin.plot(
            df["Step"], df["Disk_Read_MBs"], color="tab:orange", linewidth=1.5, linestyle=":", label="Read (MB/s)"
        )
        line3 = ax0_twin.plot(
            df["Step"], df["Disk_Write_MBs"], color="tab:purple", linewidth=1.5, linestyle="--", label="Write (MB/s)"
        )
        ax0_twin.set_ylabel("Disk I/O (MB/s)")

        # Merge the legends of both axes
        lines = line1 + line2 + line3
        labels = [line.get_label() for line in lines]
        axs[0].legend(lines, labels, loc="upper left")
    else:
        axs[0].legend(loc="upper left")

    # 2. RAM usage
    axs[1].plot(df["Step"], df["RAM_Usage_GB"], color="tab:blue", linewidth=2)
    axs[1].set_title("System RAM Usage Over Time")
    axs[1].set_ylabel("RAM Usage (GB)")
    axs[1].grid(True, linestyle="--", alpha=0.7)

    # 3. System FPS
    axs[2].plot(df["Step"], df["System_FPS"], color="tab:green", linewidth=2)
    axs[2].set_title("System Processing Speed (FPS)")
    axs[2].set_xlabel("Simulation Step")
    axs[2].set_ylabel("FPS (Hz)")
    axs[2].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    suffix_str = f"_{suffix}" if suffix else ""
    plot_path = os.path.join(out_dir, f"{mode_name_clean}_charts{suffix_str}.png")

    plt.savefig(plot_path, dpi=200)
    print(f"[INFO] Performance charts saved to {plot_path}")

    # if show_plot:
    #     plt.show()

    # plt.close(fig) # must close, or memory blows up


class CollisionCounter:
    """Counts collisions for Table 9: a collision starts when the body or a leg touches something (> 1 N)."""

    def __init__(self, env_unwrapped):
        self.sensor = env_unwrapped.scene.sensors["contact_forces"]
        self.body_ids, _ = self.sensor.find_bodies(["body", ".*leg"])
        self.in_contact = False
        self.count = 0

    def update(self):
        forces = self.sensor.data.net_forces_w_history.torch[0, :, self.body_ids]  # (history, bodies, 3)
        touching = bool((forces.norm(dim=-1) > 1.0).any())
        if touching and not self.in_contact:
            self.count += 1
        self.in_contact = touching


def record_navigation_run(map_name: str, success: bool, steps: int, time_s: float, collisions: int, pos_err: float):
    """Append one run to --results_csv, then print the navigation results (paper Table 9, mean +- std per map)."""
    row = {
        "map": map_name,
        "seed": args_cli.seed,
        "success": int(success),
        "steps": steps,
        "time_s": round(time_s, 2),
        "collisions": collisions,
        "mean_pos_err_m": round(pos_err, 3),
    }
    print("[NAV-TEST] run:", row)
    df = pd.DataFrame([row])
    df.to_csv(args_cli.results_csv, mode="a", header=not os.path.isfile(args_cli.results_csv), index=False)

    df = pd.read_csv(args_cli.results_csv)
    print(
        f"[NAV-TEST] Navigation test over {len(df)} runs in {args_cli.results_csv} (failed runs count as 3000 steps):"
    )
    print(f"[NAV-TEST] {'Map':<4} {'Runs':>4} {'Success %':>10} {'Collisions':>14} {'Steps':>18} {'Time (s)':>16}")
    for m, g in df.groupby("map"):
        ok = g[g.success == 1]
        t = f"{ok.time_s.mean():.2f} ± {ok.time_s.std(ddof=1):.2f}" if len(ok) else "—"
        print(
            f"[NAV-TEST] {m:<4} {len(g):>4} {100 * g.success.mean():>10.0f} "
            f"{g.collisions.mean():>6.2f} ± {g.collisions.std(ddof=1):<5.2f} "
            f"{g.steps.mean():>8.2f} ± {g.steps.std(ddof=1):<7.2f} {t:>16}"
        )


def main():
    if args_cli.navigation_test and not args_cli.paper:
        raise ValueError("--navigation_test needs --paper")
    low_env, low_agent = build_low_level()

    if args_cli.paper:
        map_name = getattr(low_env.unwrapped.cfg, "paper_map", None)
        if map_name is None:
            raise ValueError("--paper needs a Spot-Paper-high-level-Map*-Play-v0 task")
        paper_map = PAPER_MAPS[map_name]
        hl_cfg = PaperHighLevelEnvCfg(
            goal_w=paper_map.goal,
            goal_radius=paper_map.goal_radius,
            max_episode_hl_steps=3000,  # Table 9 timeout
        )
        vec_env = PaperHighLevelIsaacVecEnv(low_env, low_agent, hl_cfg)
        collisions = CollisionCounter(low_env.unwrapped)
        print(
            f"[INFO] Paper map {map_name}: start {paper_map.start} -> goal {paper_map.goal}"
            f" (r={paper_map.goal_radius} m)"
        )
    else:
        hl_cfg = HighLevelEnvCfg(
            hl_decimation=2,
            cmd_smoothing_alpha=0.25,
            vx_min=0.0,
            vx_max=0.5,
            wz_min=-0.25,
            wz_max=0.25,
            max_episode_hl_steps=3000,  # paper Table 9: evaluation timeout of 3000 steps (400 is the training value)
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
    os.makedirs(plot_dir, exist_ok=True)
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
                pose_source=args_cli.dense_pose_source,
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
    run_start_time = time.time()
    run_result = None
    print("[INFO] Start rollout")
    print("[INFO] num_envs =", vec_env.num_envs)

    sys_cpu_log = []
    sys_ram_log = []
    fps_log = []
    last_time = time.time()
    record_interval = 10  # log every 10 steps
    sys_disk_read_log = []  # disk read rate
    sys_disk_write_log = []  # disk write rate

    last_time = time.time()
    last_disk_io = psutil.disk_io_counters()  # initial disk I/O counters
    record_interval = 10  # log every 10 steps

    try:
        for step in range(args_cli.steps):
            log_now = args_cli.log_every > 0 and step % args_cli.log_every == 0
            action, _ = model.predict(obs, deterministic=args_cli.deterministic_policy)
            obs, rewards, dones, infos = vec_env.step(action)

            # ----------------------------------------------------
            # 3. Performance logging in the loop (after vec_env.step)
            if step % record_interval == 0 and step > 0:
                current_time = time.time()
                current_disk_io = psutil.disk_io_counters()  # current disk I/O counters

                # CPU and RAM
                cpu_percent = psutil.cpu_percent(interval=None)
                ram_gb = psutil.virtual_memory().used / (1024**3)

                # Mean FPS and elapsed time over these 10 steps
                elapsed = current_time - last_time
                fps = record_interval / elapsed

                # Disk read/write rate (MB/s):
                # disk_io_counters counts bytes since boot, so take the difference and divide by the time and 1 MB
                read_mbs = (current_disk_io.read_bytes - last_disk_io.read_bytes) / (1024 * 1024 * elapsed)
                write_mbs = (current_disk_io.write_bytes - last_disk_io.write_bytes) / (1024 * 1024 * elapsed)

                sys_cpu_log.append(cpu_percent)
                sys_ram_log.append(ram_gb)
                fps_log.append(fps)
                sys_disk_read_log.append(read_mbs)
                sys_disk_write_log.append(write_mbs)

                # Remember the current state
                last_time = current_time
                last_disk_io = current_disk_io
            # ----------------------------------------------------

            if dense_map_manager is not None:
                dense_map_manager.update()

            if args_cli.navigation_test:
                collisions.update()
                if dones[0]:
                    success = bool(infos[0].get("is_success", False))
                    run_result = dict(
                        success=success,
                        steps=step + 1 if success else 3000,
                        time_s=time.time() - run_start_time,
                        collisions=collisions.count,
                    )
                    print(
                        f"[NAV-TEST] {'SUCCESS' if success else 'FAIL'} at step {step + 1},"
                        f" {collisions.count} collisions"
                    )
                    break

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

                if log_now:
                    print(
                        "[ORB-vs-GT] "
                        f"pos_err={pos_err:.3f} m, "
                        f"yaw_err={yaw_err_deg:.2f} deg, "
                        f"orb_rel=({orb_rel[0]: .3f}, {orb_rel[1]: .3f}, {orb_rel[2]: .3f}), "
                        f"gt_rel=({gt_rel[0]: .3f}, {gt_rel[1]: .3f}, {gt_rel[2]: .3f})"
                    )
            elif log_now:
                print("[ORB-vs-GT] waiting for ORB tracking initialization...")

            # --- Periodic autosave ---
            if step > 0 and step % autosave_every == 0:
                # 1. Trajectory plot
                save_error_plots(
                    out_dir=plot_dir,
                    timesteps=timesteps,
                    pos_errors=pos_errors,
                    yaw_errors=yaw_errors,
                    orb_traj_x=orb_traj_x,
                    orb_traj_y=orb_traj_y,
                    gt_traj_x=gt_traj_x,
                    gt_traj_y=gt_traj_y,
                    suffix=f"step_{step:06d}",
                )

                # 2. Performance plot (only when something was logged)
                if len(sys_cpu_log) > 0:
                    mode_name = "rl_slam_mapping" if args_cli.enable_dense_mapping else "rl_slam"
                    df = pd.DataFrame(
                        {
                            "Step": range(record_interval, record_interval * len(sys_cpu_log) + 1, record_interval),
                            "CPU_Usage_Percent": sys_cpu_log,
                            "RAM_Usage_GB": sys_ram_log,
                            "System_FPS": fps_log,
                            "Disk_Read_MBs": sys_disk_read_log,
                            "Disk_Write_MBs": sys_disk_write_log,
                        }
                    )
                    temp_csv = os.path.join(
                        args_cli.dense_export_dir, f"performance_log_{mode_name}_step_{step:06d}.csv"
                    )
                    df.to_csv(temp_csv, index=False)
                    save_performance_plots(temp_csv, plot_dir, suffix=f"step_{step:06d}", show_plot=False)
                # ----------------------------------------------------

            # Print only the first env
            info0 = infos[0] if len(infos) > 0 else {}
            cmd_vx = info0.get("cmd_vx", None)
            cmd_wz = info0.get("cmd_wz", None)
            progress = info0.get("progress", None)
            stuck = info0.get("stuck", None)
            body_contact = info0.get("body_contact", None)
            tracking_bad = info0.get("tracking_bad", None)

            if log_now:
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
                # the robot is teleported on reset: restart the ORB/GT alignment and break the plotted lines
                orb_gt_comparator.reset()
                for log in (timesteps, pos_errors, yaw_errors, orb_traj_x, orb_traj_y, gt_traj_x, gt_traj_y):
                    log.append(np.nan)
                print(f"[INFO] Episode ended at step {step}, continuing after reset inside VecEnv.")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected. Stopping rollout safely...")

    print("[INFO] Finished rollout")

    if args_cli.navigation_test:
        if run_result is None:  # stopped early (Ctrl+C or --steps < 3000): not a complete run
            print("[NAV-TEST] Run did not finish (no goal, no timeout); nothing recorded.")
        else:
            pos_err = float(np.nanmean(pos_errors)) if np.any(np.isfinite(pos_errors)) else float("nan")
            record_navigation_run(map_name, pos_err=pos_err, **run_result)

    # ----------------------------------------------------
    # 4. After the loop, save the collected data as CSV
    mode_name = "rl_slam_mapping" if args_cli.enable_dense_mapping else "rl_slam"
    if len(sys_cpu_log) > 0:
        final_csv = os.path.join(args_cli.dense_export_dir, f"performance_log_{mode_name}_final.csv")
        df = pd.DataFrame(
            {
                "Step": range(record_interval, record_interval * len(sys_cpu_log) + 1, record_interval),
                "CPU_Usage_Percent": sys_cpu_log,
                "RAM_Usage_GB": sys_ram_log,
                "System_FPS": fps_log,
                "Disk_Read_MBs": sys_disk_read_log,
                "Disk_Write_MBs": sys_disk_write_log,
            }
        )
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

    if dense_map_manager is not None:
        print(
            f"[INFO] TSDF integrated frames: {dense_map_manager.frame_count} (pose_source={args_cli.dense_pose_source})"
        )
        dense_map_manager.close()

    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
