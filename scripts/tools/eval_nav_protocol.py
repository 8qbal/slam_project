"""Table 9 protocol on a high-level checkpoint: one run per env, up to 3000 HL steps, contacts counted (not ending the
run), success = goal reached. Prints success, collisions, steps and time per run.

Ground-truth pose by default; use ``Spot-Nav-Map{A,B}-GT-Play-v0`` (fixed start facing the goal, deterministic, so
the runs are nearly identical) or ``Spot-Nav-Map{A,B}-v0`` (training start: random heading, +-0.5 m, a push) with
``--num_envs 16``.

    SPOT_VSLAM_LOW_LEVEL_CKPT=policies/walls_soft_mirror_turn/model.pth \
        uv run python scripts/tools/eval_nav_protocol.py --task Spot-Nav-MapA-GT-Play-v0 --checkpoint <high-level .pth>

``--jitter_xy 0.5 --jitter_yaw 0.5 --run_seed N`` gives a random start (position and heading around the goal
direction, no push) to any Play task, the same for GT, cuVSLAM and ORB-SLAM3.

With ``Spot-Nav-Map{A,B}-cuVSLAM-Play-v0`` the high level gets the cuVSLAM pose, and the script also prints the position
error against ground truth (start frame) and the share of steps without a valid pose.
"""

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task")
p.add_argument("--checkpoint")
p.add_argument("--num_envs", type=int, default=5)
p.add_argument("--max_steps", type=int, default=3000)
p.add_argument("--run_seed", type=int, default=1)
p.add_argument("--trace_npz", default=None, help="save per-step SLAM error and quality signals (cuVSLAM) to this file")
p.add_argument("--jitter_xy", type=float, default=0.0, help="Play tasks: random start offset, +-m in x and y")
p.add_argument("--jitter_yaw", type=float, default=0.0, help="Play tasks: random start heading, +-rad around the goal")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import spot_vslam.tasks  # noqa: E402,F401

cfg = parse_env_cfg(a.task, device="cuda:0", num_envs=a.num_envs)
cfg.seed = a.run_seed
if a.jitter_xy > 0 or a.jitter_yaw > 0:  # random start for the fixed-start Play tasks (no push, so no velocity range)
    cfg.events.reset_base.params["pose_range"] = {
        "x": (-a.jitter_xy, a.jitter_xy),
        "y": (-a.jitter_xy, a.jitter_xy),
        "yaw": (-a.jitter_yaw, a.jitter_yaw),
    }
cfg.episode_length_s = (a.max_steps + 1) * 0.08  # timeout only after max_steps HL steps
cfg.terminations.body_contact = None  # contacts are counted, not ending the run
cfg.terminations.leg_contact = None
cfg.rewards.fall = None
agent_cfg = load_cfg_from_registry(a.task, "rl_games_cfg_entry_point")
clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))
env = RlGamesVecEnvWrapper(gym.make(a.task, cfg=cfg), "cuda:0", clip_obs, clip_actions)
vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
agent_cfg["params"]["load_checkpoint"] = True
agent_cfg["params"]["load_path"] = a.checkpoint
agent_cfg["params"]["config"]["num_actors"] = a.num_envs
runner = Runner()
runner.load(agent_cfg)
agent = runner.create_player()
agent.restore(a.checkpoint)
agent.reset()

u = env.unwrapped
sensor = u.scene["contact_forces"]
body_ids, _ = sensor.find_bodies(["body", ".*leg"])
n = a.num_envs
obs = env.reset()
obs = obs["obs"] if isinstance(obs, dict) else obs
_ = agent.get_batch_size(obs, 1)
touching = torch.zeros(n, dtype=torch.bool, device=u.device)
collisions = torch.zeros(n, device=u.device)
finished = torch.zeros(n, dtype=torch.bool, device=u.device)
success = torch.zeros(n, dtype=torch.bool, device=u.device)
steps = torch.full((n,), float(a.max_steps), device=u.device)
cmd = u.command_manager.get_term("goal")
err_sum = torch.zeros(n, device=u.device)
err_max = torch.zeros(n, device=u.device)
err_n = torch.zeros(n, device=u.device)
lost = torch.zeros(n, device=u.device)
trace = {k: [] for k in ("err", "features", "odom_slam", "track_ms", "dist", "live")}
t = 0
while not finished.all() and t < a.max_steps:
    with torch.inference_mode():
        act = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
        obs, _, dones, _ = env.step(act)
    t += 1
    hist = sensor.data.net_forces_w_history[:, :, body_ids]  # (N, history, bodies, 3)
    now = hist.norm(dim=-1).amax(dim=(1, 2)) > 1.0
    collisions += (now & ~touching & ~finished).float()
    touching = now
    reached = u.termination_manager.get_term("goal_reached").bool() & ~finished
    success |= reached
    steps = torch.where(reached, torch.full_like(steps, t), steps)
    slam = getattr(u, "slam_res", None) or getattr(u, "orb_slam_res", None)  # cuVSLAM or ORB-SLAM3
    if slam is not None:  # position error of the SLAM pose against ground truth, both in the start frame
        e = (slam["pose_xyyaw"][:, :2] - cmd.gt_pose_start_frame()[:, :2]).norm(dim=1)
        live = ~finished & ~dones.bool()
        err_sum += torch.where(live, e, torch.zeros_like(e))
        err_max = torch.where(live, torch.maximum(err_max, e), err_max)
        err_n += live.float()
        lost += (live & (slam["status"].flatten() < 0.5)).float()
        if a.trace_npz and "num_features" in slam:
            gap = (slam["odom_xyyaw"][:, :2] - slam["slam_xyyaw"][:, :2]).norm(dim=1)  # NaN until a loop closure pose
            for k, v in zip(
                trace,
                (e, slam["num_features"].flatten(), gap, slam["track_time_ms"].flatten(), cmd.distance(), live.float()),
            ):
                trace[k].append(v.cpu().numpy())
    finished |= dones.bool()
    if slam is not None and t % 100 == 0 and n == 1:  # one-env trace: where GT and SLAM disagree
        g, q = cmd.gt_pose_start_frame()[0], slam["pose_xyyaw"][0]
        ok = float(slam["status"].flatten()[0])
        print(f"PROTO trace {t}: gt ({g[0]:.1f}, {g[1]:.1f}) slam ({q[0]:.1f}, {q[1]:.1f}) status {ok:.0f}", flush=True)
    if t % 500 == 0:
        print(f"PROTO step {t}: finished {int(finished.sum())}/{n}, success {int(success.sum())}", flush=True)
sec = steps * u.step_dt
print(
    "PROTO RESULT",
    a.task,
    a.checkpoint.split("/")[-2],
    f"runs {n}",
    f"success {100 * success.float().mean():.0f}%",
    f"collisions {collisions.mean():.2f}+-{collisions.std():.2f}",
    f"steps {steps.mean():.0f}+-{steps.std():.0f}",
    f"time {sec.mean():.1f}s",
    "per-run success",
    success.int().tolist(),
    "steps",
    steps.int().tolist(),
    "coll",
    collisions.int().tolist(),
)
if (getattr(u, "slam_res", None) or getattr(u, "orb_slam_res", None)) is not None:
    print(
        "PROTO SLAM mean error",
        [round(v, 2) for v in (err_sum / err_n.clamp(min=1)).tolist()],
        "max error",
        [round(v, 2) for v in err_max.tolist()],
        "steps without a pose",
        [round(v, 3) for v in (lost / err_n.clamp(min=1)).tolist()],
    )
if a.trace_npz and trace["err"]:
    import numpy as np

    np.savez(a.trace_npz, **{k: np.stack(v) for k, v in trace.items()})  # each (steps, envs)
env.close()
app.close()
