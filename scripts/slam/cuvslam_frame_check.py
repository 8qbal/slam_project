"""Frame check for in-process cuVSLAM (docs/cuvslam.md): drive Spot on Map A with fixed commands and print the
cuVSLAM pose next to the ground-truth pose, both in the start frame. No high-level policy, no ROS 2.

    SPOT_VSLAM_LOW_LEVEL_CKPT=<low-level .pth> uv run scripts/slam/cuvslam_frame_check.py

Phases (high-level steps of 0.08 s): stand, straight, turn left, turn right. x must grow forward, yaw must turn the
same way as GT. Odometry and loop-closure poses are compared separately; trajectories go to
output/cuvslam_frame_check/ (t x y yaw).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Spot-Nav-MapA-cuVSLAM-Play-v0")
parser.add_argument("--print_every", type=int, default=10)
parser.add_argument("--no_loop_closure", action="store_true", help="cuVSLAM odometry only")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import math
import os

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import spot_vslam.tasks  # noqa: F401

# (steps, vx m/s, wz rad/s)
PHASES = [(25, 0.0, 0.0), (100, 0.5, 0.0), (80, 0.3, 0.4), (80, 0.3, -0.4)]
VX_RANGE, WZ_RANGE = (0.0, 1.0), (-0.4, 0.4)
EST = ("pose_xyyaw", "odom_xyyaw", "slam_xyyaw")  # used by the policy, odometry only, loop closure only
NO_SLAM = {
    **{k: torch.zeros(1, 3) for k in EST},
    **{k: torch.zeros(1, 1) for k in ("status", "num_features", "track_time_ms")},
}


def to_action(vx, wz):
    a = lambda v, r: 2.0 * (v - r[0]) / (r[1] - r[0]) - 1.0  # noqa: E731
    return torch.tensor([[a(vx, VX_RANGE), a(wz, WZ_RANGE)]])


cfg = parse_env_cfg(args.task, num_envs=1)
if args.no_loop_closure and cfg.cuvslam is not None:
    cfg.cuvslam.loop_closure = False
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
goal = env.command_manager.get_term("goal")

rows = []  # step, GT (3), pose (3), odom (3), slam (3), ok, features, ms
step = 0
robot = env.scene["robot"]
print(
    f"{'step':>5} {'cmd vx wz':>10} | {'GT x':>6} {'y':>6} {'yaw':>6} | {'odom x':>6} {'y':>6} {'yaw':>6} |"
    f" {'loop x':>6} {'y':>6} {'yaw':>6} | ok feat  ms | z roll"
)
for n, vx, wz in PHASES:
    action = to_action(vx, wz).to(env.device)
    for _ in range(n):
        env.step(action)
        step += 1
        gt = goal.gt_pose_start_frame()[0].tolist()
        res = getattr(env, "slam_res", None) or NO_SLAM  # GT-Play: no SLAM
        est = [res.get(k, torch.full((1, 3), float("nan")))[0].tolist() for k in EST]
        ok, feat, ms = float(res["status"][0]), float(res["num_features"][0]), float(res["track_time_ms"][0])
        rows.append((step, *gt, *est[0], *est[1], *est[2], ok, feat, ms))
        if step % args.print_every == 0:
            z = float(robot.data.root_pos_w.torch[0, 2])
            qx, qy, qz, qw = robot.data.root_quat_w.torch[0].tolist()
            roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
            od, lc = est[1], est[2]
            print(
                f"{step:5d} {vx:4.1f} {wz:4.1f} | {gt[0]:6.2f} {gt[1]:6.2f} {math.degrees(gt[2]):6.1f} |"
                f" {od[0]:6.2f} {od[1]:6.2f} {math.degrees(od[2]):6.1f} |"
                f" {lc[0]:6.2f} {lc[1]:6.2f} {math.degrees(lc[2]):6.1f} | {ok:.0f} {feat:4.0f} {ms:5.1f}"
                f" | {z:.2f} {math.degrees(roll):5.1f}"
            )

t = torch.tensor(rows)
gt_t = t[:, 1:4]
print(
    f"\ntracked {t[:, 13].mean() * 100:.0f}% of steps, features {t[:, 14].mean():.0f}, track {t[:, 15].mean():.1f} ms"
)
for j, name in enumerate(EST):
    e = t[:, 4 + 3 * j : 7 + 3 * j]
    valid = ~torch.isnan(e[:, 0])
    if not valid.any():
        print(f"{name}: no estimate")
        continue
    err_xy = torch.linalg.norm(gt_t[valid, :2] - e[valid, :2], dim=1)
    d_yaw = gt_t[valid, 2] - e[valid, 2]
    err_yaw = torch.rad2deg(torch.atan2(torch.sin(d_yaw), torch.cos(d_yaw))).abs()
    jumps = (torch.linalg.norm(e[valid][1:, :2] - e[valid][:-1, :2], dim=1) > 0.5).sum()
    print(
        f"{name}: {int(valid.sum())} steps, xy error mean {err_xy.mean():.3f} m, max {err_xy.max():.3f} m,"
        f" final {err_xy[-1]:.3f} m; yaw error mean {err_yaw.mean():.1f} deg; jumps > 0.5 m/step: {int(jumps)}"
    )
out = "output/cuvslam_frame_check"
os.makedirs(out, exist_ok=True)
dt = env.step_dt
for j, name in enumerate(("gt",) + EST):
    with open(f"{out}/{name.removesuffix('_xyyaw')}.txt", "w") as f:
        for r in rows:
            x, y, yaw = r[1 + 3 * j : 4 + 3 * j]
            f.write(f"{r[0] * dt:.3f} {x:.4f} {y:.4f} {yaw:.4f}\n")
print(f"wrote {out}/{{gt,pose,odom,slam}}.txt (t x y yaw)")
env.close()
app.close()
