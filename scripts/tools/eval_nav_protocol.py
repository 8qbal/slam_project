"""Table 9 protocol on a high-level checkpoint: one run per env, up to 3000 HL steps, contacts counted (not ending the
run), success = goal reached. Prints success, collisions, steps and time per run.

Ground-truth pose by default; use ``Spot-Nav-Map{A,B}-GT-Play-v0`` (fixed start facing the goal, deterministic, so
the runs are nearly identical) or ``Spot-Nav-Map{A,B}-v0`` (training start: random heading, +-0.5 m, a push) with
``--num_envs 16``.

    SPOT_VSLAM_LOW_LEVEL_CKPT=policies/walls_soft_mirror_turn/model.pth \
        uv run python scripts/tools/eval_nav_protocol.py --task Spot-Nav-MapA-GT-Play-v0 --checkpoint <high-level .pth>

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
cfg.seed = 1
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
    slam = getattr(u, "slam_res", None)
    if slam is not None:  # position error of the SLAM pose against ground truth, both in the start frame
        e = (slam["pose_xyyaw"][:, :2] - cmd.gt_pose_start_frame()[:, :2]).norm(dim=1)
        live = ~finished & ~dones.bool()
        err_sum += torch.where(live, e, torch.zeros_like(e))
        err_max = torch.where(live, torch.maximum(err_max, e), err_max)
        err_n += live.float()
        lost += (live & (slam["status"].flatten() < 0.5)).float()
    finished |= dones.bool()
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
if getattr(u, "slam_res", None) is not None:
    print(
        "PROTO SLAM mean error",
        [round(v, 2) for v in (err_sum / err_n.clamp(min=1)).tolist()],
        "max error",
        [round(v, 2) for v in err_max.tolist()],
        "steps without a pose",
        [round(v, 3) for v in (lost / err_n.clamp(min=1)).tolist()],
    )
env.close()
app.close()
