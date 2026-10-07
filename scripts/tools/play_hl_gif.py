"""Play the trained high-level PPO on top of the frozen low-level policy and record a chase-cam GIF."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Spot-Vslam-high-level-Play-v0")
parser.add_argument("--low_level_checkpoint", required=True)
parser.add_argument("--high_level_checkpoint", required=True)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--out", required=True)
parser.add_argument("--const_cmd", action="store_true")
parser.add_argument("--spawn", type=float, nargs=2, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from stable_baselines3 import PPO

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import spot_vslam.tasks  # noqa: F401
from spot_vslam.legacy.high_level.high_level_vec_env import HighLevelEnvCfg, HighLevelIsaacVecEnv
from spot_vslam.slam.ros2_manager import Ros2Manager

env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
if args_cli.spawn is not None:
    env_cfg.scene.robot.init_state.pos = (args_cli.spawn[0], args_cli.spawn[1], env_cfg.scene.robot.init_state.pos[2])
# workaround: a depth-only camera in the scene makes every camera's RGB output black (Isaac Lab 3.0 RTX)
env_cfg.scene.train_camera.data_types = ["rgb", "distance_to_image_plane"]
# third-person chase camera: 2.5 m behind, 1.5 m above, pitched ~25 deg down
env_cfg.scene.chase_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/body/chase_cam",
    update_period=0.0,
    height=360,
    width=480,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, horizontal_aperture=20.955, clipping_range=(0.1, 100.0)),
    offset=CameraCfg.OffsetCfg(pos=(-2.5, 0.0, 1.5), rot=(0.0, 0.2164, 0.0, 0.9763), convention="world"),
)

agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
env = gym.make(args_cli.task, cfg=env_cfg)
env = RlGamesVecEnvWrapper(
    env,
    agent_cfg["params"]["config"]["device"],
    agent_cfg["params"]["env"].get("clip_observations", float("inf")),
    agent_cfg["params"]["env"].get("clip_actions", float("inf")),
)
vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
agent_cfg["params"]["load_checkpoint"] = True
agent_cfg["params"]["load_path"] = args_cli.low_level_checkpoint
agent_cfg["params"]["config"]["num_actors"] = 1
runner = Runner()
runner.load(agent_cfg)
agent = runner.create_player()
agent.restore(args_cli.low_level_checkpoint)
agent.reset()

if getattr(env.unwrapped.cfg, "ros2", None) is not None:
    env.unwrapped.ros2_manager = Ros2Manager(env.unwrapped.cfg.ros2, env.unwrapped)

# same high-level cfg as scripts/high_level/train_high_level_ppo.py
hl_cfg = HighLevelEnvCfg(
    hl_decimation=2,
    cmd_smoothing_alpha=0.25,
    vx_min=0.0,
    vx_max=float(os.environ.get("VX_MAX", 1.0)),
    wz_min=-float(os.environ.get("WZ_MAX", 0.4)),
    wz_max=float(os.environ.get("WZ_MAX", 0.4)),
    max_episode_hl_steps=400,
)
vec_env = HighLevelIsaacVecEnv(env, agent, hl_cfg)
model = PPO.load(args_cli.high_level_checkpoint, env=vec_env, device="cpu")

robot = env.unwrapped.scene["robot"]
cam = env.unwrapped.scene.sensors["chase_cam"]
obs = vec_env.reset()
u = env.unwrapped
print("[DIAG] env_origins:", u.scene.env_origins.cpu().numpy().tolist(), flush=True)
print(
    "[DIAG] terrain cfg size/rows/cols:",
    u.scene.terrain.cfg.terrain_generator.size if u.scene.terrain.cfg.terrain_generator else None,
    getattr(u.scene.terrain.cfg.terrain_generator, "num_rows", None),
    getattr(u.scene.terrain.cfg.terrain_generator, "num_cols", None),
    flush=True,
)
print("[DIAG] root pos after reset:", robot.data.root_pos_w.torch[0].cpu().numpy().tolist(), flush=True)
print("[DIAG] termination terms:", u.termination_manager.active_terms, flush=True)
frames, traj, ep_ret, contacts = [], [], 0.0, 0
for step in range(args_cli.steps):
    action, _ = model.predict(obs, deterministic=True)
    if args_cli.const_cmd:
        action = np.zeros_like(action)
    obs, rew, done, info = vec_env.step(action)
    ep_ret += float(rew[0])
    contacts += int(info[0]["body_contact"] > 0)
    pos = robot.data.root_pos_w.torch[0].cpu().numpy()
    traj.append(pos[:2].copy())
    if step < 3:
        for nm in ["chase_cam", "tilted_camera"]:
            t = u.scene.sensors[nm].data.output["rgb"].torch
            print(
                f"[DIAG] {nm} rgb dtype={t.dtype} shape={tuple(t.shape)} min={t.min().item()} max={t.max().item()} mean={t.float().mean().item():.3f}",
                flush=True,
            )
    rgb = cam.data.output["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8)
    img = Image.fromarray(rgb)
    ImageDraw.Draw(img).text(
        (8, 8),
        f"t={step * 0.08:5.1f}s  vx={info[0]['cmd_vx']:.2f}  wz={info[0]['cmd_wz']:+.2f}  x={pos[0]:+.1f} y={pos[1]:+.1f}",
        fill=(255, 255, 0),
    )
    frames.append(np.asarray(img))
    if step % 25 == 0:
        print(
            f"[PLAY] step={step} pos=({pos[0]:+.2f},{pos[1]:+.2f}) cmd_vx={info[0]['cmd_vx']:.2f} "
            f"cmd_wz={info[0]['cmd_wz']:+.2f} ret={ep_ret:.2f}",
            flush=True,
        )
    fired = [n for n in u.termination_manager.active_terms if bool(u.termination_manager.get_term(n)[0])]
    if args_cli.steps <= 60:
        print(f"[DIAG] step={step} pos={pos.round(2).tolist()} fired={fired}", flush=True)
    if done[0]:
        print(f"[PLAY] episode done at step {step}: {info[0].get('episode')}", flush=True)
        break

traj = np.array(traj)
dist = np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()
print(
    f"[RESULT] steps={len(frames)} return={ep_ret:.2f} path_len={dist:.2f}m "
    f"start={traj[0]} end={traj[-1]} net_disp={np.linalg.norm(traj[-1] - traj[0]):.2f}m body_contact_steps={contacts}"
)
os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
np.save(args_cli.out.replace(".gif", "_traj.npy"), traj)
imageio.mimsave(args_cli.out, frames[::2], duration=0.16, loop=0)  # every 2nd HL step, real-time speed
print("[RESULT] gif saved:", args_cli.out)
simulation_app.close()
