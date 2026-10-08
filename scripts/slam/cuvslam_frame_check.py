"""Frame check for in-process cuVSLAM (docs/cuvslam.md): drive Spot on Map A with fixed commands and print the
cuVSLAM pose next to the ground-truth pose, both in the start frame. No high-level policy, no ROS 2.

    SPOT_VSLAM_LOW_LEVEL_CKPT=<low-level .pth> uv run scripts/slam/cuvslam_frame_check.py

Phases (high-level steps of 0.08 s): stand, straight, turn left, turn right. x must grow forward, yaw must turn the
same way as GT. The trajectories are written to output/cuvslam_frame_check/ (TUM-like: t x y yaw).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Spot-Nav-MapA-cuVSLAM-Play-v0")
parser.add_argument("--print_every", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import math
import os

import gymnasium as gym
import torch

import spot_vslam.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# (steps, vx m/s, wz rad/s)
PHASES = [(25, 0.0, 0.0), (100, 0.5, 0.0), (80, 0.3, 0.4), (80, 0.3, -0.4)]
VX_RANGE, WZ_RANGE = (0.0, 1.0), (-0.4, 0.4)
SLAM_KEYS = ("pose_xyyaw", "status", "num_features", "track_time_ms")


def to_action(vx, wz):
    a = lambda v, r: 2.0 * (v - r[0]) / (r[1] - r[0]) - 1.0  # noqa: E731
    return torch.tensor([[a(vx, VX_RANGE), a(wz, WZ_RANGE)]])


cfg = parse_env_cfg(args.task, num_envs=1)
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
goal = env.command_manager.get_term("goal")

rows = []
step = 0
robot = env.scene["robot"]
term = env.action_manager.get_term("pre_trained_policy")
print(
    f"{'step':>5} {'cmd vx wz':>10} | {'GT x':>6} {'y':>6} {'yaw':>6} | {'SLAM x':>6} {'y':>6} {'yaw':>6} | ok feat  ms"
    " | z roll pitch | ll vx wz | body vx wz"
)
for n, vx, wz in PHASES:
    action = to_action(vx, wz).to(env.device)
    for _ in range(n):
        env.step(action)
        step += 1
        gt = goal.gt_pose_start_frame()[0].tolist()
        res = getattr(env, "slam_res", None) or {k: torch.zeros(1, 3 if k == "pose_xyyaw" else 1) for k in SLAM_KEYS}  # GT-Play: no SLAM
        sl = res["pose_xyyaw"][0].tolist()
        ok, feat, ms = float(res["status"][0]), float(res["num_features"][0]), float(res["track_time_ms"][0])
        rows.append((step, *gt, *sl, ok, feat, ms))
        z = float(robot.data.root_pos_w.torch[0, 2])
        qx, qy, qz, qw = robot.data.root_quat_w.torch[0].tolist()
        roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
        ll = term.processed_actions[0].tolist()
        bv = [float(robot.data.root_lin_vel_b.torch[0, 0]), float(robot.data.root_ang_vel_b.torch[0, 2])]
        if step % args.print_every == 0:
            print(
                f"{step:5d} {vx:4.1f} {wz:4.1f} | {gt[0]:6.2f} {gt[1]:6.2f} {math.degrees(gt[2]):6.1f} |"
                f" {sl[0]:6.2f} {sl[1]:6.2f} {math.degrees(sl[2]):6.1f} | {ok:.0f} {feat:4.0f} {ms:5.1f}"
                f" | {z:.2f} {math.degrees(roll):5.1f} {math.degrees(pitch):5.1f}"
                f" | {ll[0]:.2f} {ll[2]:5.2f} | {bv[0]:5.2f} {bv[1]:5.2f}"
            )

t = torch.tensor(rows)
err_xy = torch.linalg.norm(t[:, 1:3] - t[:, 4:6], dim=1)
err_yaw = torch.rad2deg(torch.atan2(torch.sin(t[:, 3] - t[:, 6]), torch.cos(t[:, 3] - t[:, 6]))).abs()
print(
    f"\ntracked {t[:, 7].mean() * 100:.0f}% of steps, features {t[:, 8].mean():.0f}, track {t[:, 9].mean():.1f} ms"
    f"\nxy error: mean {err_xy.mean():.3f} m, final {err_xy[-1]:.3f} m; yaw error mean {err_yaw.mean():.1f} deg"
)
out = "output/cuvslam_frame_check"
os.makedirs(out, exist_ok=True)
dt = env.step_dt
with open(f"{out}/gt.txt", "w") as f_gt, open(f"{out}/cuvslam.txt", "w") as f_sl:
    for r in rows:
        f_gt.write(f"{r[0] * dt:.3f} {r[1]:.4f} {r[2]:.4f} {r[3]:.4f}\n")
        f_sl.write(f"{r[0] * dt:.3f} {r[4]:.4f} {r[5]:.4f} {r[6]:.4f}\n")
print(f"wrote {out}/gt.txt, {out}/cuvslam.txt")
env.close()
app.close()
