"""Run a low-level checkpoint at fixed commands and print gait statistics: tracked speed, contact pattern, step
rate, foot lift, body motion and the per-term reward (docs/low_level_reward.md). No training, no ROS 2.

    python scripts/tools/gait_check.py --task Spot-Paper-v0 --checkpoint <low-level .pth>
    python scripts/tools/gait_check.py --task Spot-Vslam-Depth-v0 \
        --checkpoint baselines/rl_vslam_v0/policies/low_level/spot_rough_ppo.pth

Envs 0-7 get fixed commands (CMDS / WZ); the first 50 steps (2 s) are skipped. "trot" = diagonal pairs in contact
together and opposite to the other pair, "4-down" = all feet on the ground, "0-down" = all feet in the air,
"swing dz" = mean per-foot height range, "|mu|>1" = share of policy outputs outside [-1, 1] (clipped only when the
agent cfg clips).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Spot-Paper-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import spot_vslam.tasks  # noqa: F401
from spot_vslam.tasks.navigation.mdp.actions import RlGamesPolicy

CMDS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 0.5, 0.5]  # vx per env, wz = 0 (last two: wz = +0.4 / -0.4)
WZ = [0, 0, 0, 0, 0, 0, 0.4, -0.4]
N = len(CMDS)

cfg = parse_env_cfg(args.task, num_envs=N)
cfg.observations.policy.enable_corruption = False
cfg.episode_length_s = 1000.0
cfg.events.push_robot = None
cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
cfg.commands.base_velocity.rel_standing_envs = 0.0
env = gym.make(args.task, cfg=cfg).unwrapped
agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
obs, _ = env.reset()
obs_dim = obs["policy"].shape[1]
policy = RlGamesPolicy(args.checkpoint, agent_cfg, obs_dim, 12).to(env.device)

cmd_term = env.command_manager.get_term("base_velocity")
cmd = torch.tensor([[v, 0.0, w] for v, w in zip(CMDS, WZ)], device=env.device)
robot = env.scene["robot"]
sensor = env.scene.sensors["contact_forces"]
feet = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]
foot_s = [sensor.body_names.index(f) for f in feet]
foot_b = [robot.body_names.index(f) for f in feet]
terms = env.reward_manager.active_terms

log = {k: [] for k in ["vx", "wz", "contact", "foot_z", "base_z", "pitch_rate", "roll_rate", "vz", "sat", "rew"]}
resets = torch.zeros(N, device=env.device)
for t in range(args.steps):
    cmd_term.vel_command_b[:] = cmd
    o = obs["policy"]
    with torch.inference_mode():
        mu = policy.model.a2c_network({"obs": policy.model.norm_obs(o)})[0]
    act = policy(o)  # clipped as in training
    obs, _, term, trunc, _ = env.step(act)
    resets += (term | trunc).float()
    if t < 50:
        continue
    lin = robot.data.root_lin_vel_b.torch
    ang = robot.data.root_ang_vel_b.torch
    log["vx"].append(lin[:, 0].clone())
    log["vz"].append(lin[:, 2].abs().clone())
    log["wz"].append(ang[:, 2].clone())
    log["pitch_rate"].append(ang[:, 1].abs().clone())
    log["roll_rate"].append(ang[:, 0].abs().clone())
    log["contact"].append((sensor.data.current_contact_time.torch[:, foot_s] > 0).clone())
    log["foot_z"].append(robot.data.body_pos_w.torch[:, foot_b, 2].clone())
    log["base_z"].append(robot.data.root_pos_w.torch[:, 2].clone())
    log["sat"].append((mu.abs() > 1.0).float().mean(dim=1).clone())
    log["rew"].append(env.reward_manager._step_reward.clone())

L = {k: torch.stack(v) for k, v in log.items()}  # [T, N, ...]
T = L["vx"].shape[0]
dt = env.step_dt
c = L["contact"]  # [T, N, 4] fl fr hl hr
fl, fr, hl, hr = c.unbind(-1)
trot = (fl == hr) & (fr == hl) & (fl != fr)
n_down = c.sum(-1)
flight = n_down == 0
pronk = n_down == 4
touchdowns = ((~c[:-1]) & c[1:]).sum(0).float()  # [N, 4]
step_hz = touchdowns.mean(-1) / (T * dt)
swing_h = []
for e in range(N):
    fz = L["foot_z"][:, e]
    swing_h.append((fz.max(0).values - fz.min(0).values).mean().item())

print(f"\n== {args.checkpoint}\n   dt={dt:.3f}s, {T} steps measured, mean |mu|>1 fraction = {L['sat'].mean():.2f}")
print(" cmd vx  wz |   vx    wz | trot  4-down 0-down | step Hz | swing dz | base z std | vz | pitch rate | resets")
for e in range(N):
    print(
        f" {CMDS[e]:5.2f} {WZ[e]:4.1f} | {L['vx'][:, e].mean():5.2f} {L['wz'][:, e].mean():5.2f} |"
        f" {trot[:, e].float().mean():.2f}  {pronk[:, e].float().mean():.2f}   {flight[:, e].float().mean():.2f}  |"
        f"  {step_hz[e]:4.1f}   |  {swing_h[e]:.3f}  |   {L['base_z'][:, e].std():.3f}    |"
        f" {L['vz'][:, e].mean():.2f} |"
        f"   {L['pitch_rate'][:, e].mean():.2f}     | {int(resets[e])}"
    )
print("\n mean per-step reward (weighted, /dt) per term, all envs:")
r = L["rew"].mean((0, 1))
for name, v in sorted(zip(terms, r.tolist()), key=lambda x: -abs(x[1])):
    print(f"   {name:24s} {v:8.3f}")
env.close()
app.close()
