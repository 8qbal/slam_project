"""cuVSLAM with N parallel envs (one tracker each): env-steps per second, GPU memory, trackers that return a pose.

    SPOT_VSLAM_LOW_LEVEL_CKPT=policies/walls_soft_mirror_turn/model.pth \
        uv run python scripts/slam/cuvslam_multi_env.py --num_envs 16
"""

import argparse
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--steps", type=int, default=150)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import spot_vslam.tasks  # noqa: E402,F401

T = "Spot-Nav-MapA-cuVSLAM-Play-v0"
cfg = parse_env_cfg(T, device="cuda:0", num_envs=a.num_envs)
env = gym.make(T, cfg=cfg).unwrapped
env.reset()
act = torch.zeros(a.num_envs, 2, device=env.device)
act[:, 0] = 0.0  # high-level action 0 = vx 0.5 m/s, wz 0
for _ in range(20):  # warm up
    env.step(act)
torch.cuda.synchronize()
t0 = time.time()
for i in range(a.steps):
    env.step(act)
torch.cuda.synchronize()
dt = time.time() - t0
res = env.slam_res
status = res["status"].flatten()
free, total = torch.cuda.mem_get_info()
print(
    f"CUV envs {a.num_envs}: {a.steps * a.num_envs / dt:.1f} env-steps/s ({a.steps / dt:.1f} steps/s), "
    f"tracking ok {int(status.sum())}/{a.num_envs}, features {res['num_features'].flatten().tolist()[:4]}, "
    f"GPU used {(total - free) / 2**30:.1f} GiB of {total / 2**30:.1f}",
    flush=True,
)
env.close()
app.close()
