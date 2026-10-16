# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Talk simply.** Use short, plain sentences and avoid jargon and long explanations, unless the user asks for more detail.

**Answer briefly.** Get to the point: no long intros, recaps or rambling. Use a few bullets or a small table at most.

## Project

Isaac Lab 3.0 environments for running visual SLAM (ORB-SLAM3 over ROS 2, with Nav2) on the Boston Dynamics Spot
quadruped. The project was migrated from Isaac Lab 2.x to 3.0 on 2026-09-26 (see "Isaac Lab 3.0 migration notes"
below). `docs/ROADMAP.md` lists what to fix and build next (fixes, measurements, cuVSLAM, tracking-quality A/B/C,
manager-based high level, refactor backlog) — read it before making structural changes. `docs/SYSTEM.md` has the
system diagram and each policy's inputs/outputs. `scripts/README.md` has the run commands.

## Paper reproduction

The project reproduces `references/main.pdf` (Li & Jeong, IJCAS 2026); `references/README.md` maps paper settings to
code. The original author repo (github.com/BernieMHao/slam_project) has no USD files, and its code is an earlier
version than the paper (no goal in the high level, 245-dim low-level obs).

Two policy sets exist side by side. Never change one in a way that breaks the other:

- **`rl_vslam_v0`** (baseline, git tag `baseline/rl-vslam-v0`, files in `baselines/rl_vslam_v0/`): the old tasks
  (`Spot-Vslam-Depth-v0`, `Spot-Vslam-high-level-*`, config `locomotion/config/spot/baseline_v0_env_cfg.py`),
  245-dim low-level obs, 11-dim SB3 high level with no goal (`legacy/high_level/high_level_vec_env.py`).
- **Paper version**: low level `Spot-Paper-*` (`locomotion/config/spot/paper_env_cfg.py`): 353-dim obs (80x60
  depth), Table 3 reward weights on Isaac Lab's Spot reward functions plus the Eq. 5 wall term, PPO settings of
  Isaac Lab's Spot (no action clipping; the literal Eqs. 2-6 made the policy trot in place, see
  `docs/low_level_reward.md`). High level: 9-dim obs with goal (Table 5), [256, 128] ELU, Table 6
  PPO, as the manager-based `Spot-Nav-*` (`tasks/navigation/`) and as the older SB3 wrapper
  (`legacy/high_level/paper_vec_env.py`, `Spot-Paper-high-level-*`). The paper gives no high-level reward or
  training map, so those are our own (goal progress + success bonus; trained on Map A or B).

Paper maps (`PAPER_MAPS` in `assets/maps.py`; each map has one fixed start and goal, never random goals):

| Map | USD | Start → goal (radius) |
| --- | --- | --- |
| A "Room" | `Simple_Warehouse/warehouse.usd` (`WAREHOUSE_USD_PATH`) | (-9, -10) → (-4, 15.5), 1 m |
| B "Warehouse" | `Simple_Warehouse/warehouse_multiple_shelves.usd` | (-9, 17) → (7, 0), 2 m |
| C "Corridor" | `corrider_map.usd` — lost | — |

Map B's original start (-22, 3.5) lies outside the asset, so the start is the farthest free corner. The high-level
training maze (`flat_maze.usd`) is lost too. In map tasks every env has its own copy of the map, 50 m apart.

**The user runs all training themselves.** Prepare and smoke-test the code (a few epochs/steps, then delete the
output), then hand over the command.

## Commands

```bash
# Install into the Isaac Lab venv (this machine: ~/IsaacLab, venv at ~/IsaacLab/.venv)
~/IsaacLab/isaaclab.sh -p -m pip install -e source/spot_vslam
# or, with uv from this directory:
uv sync

# Source ROS 2 before running anything that touches the ROS 2 / SLAM managers
source /opt/ros/jazzy/setup.bash

# Train / play any task with rl_games. Isaac Lab 3.0 runs headless by default: every task has a depth camera, so
# pass --cameras (`isaaclab train` has no camera flag), and --viz kit for a window. train.py prints the reward every
# epoch (--quiet_rewards turns it off, --reward_terms_every N sets the per-term breakdown interval).
python scripts/rl_games/train.py --task Spot-Paper-v0 --num_envs 512 --cameras --max_iterations 8000  # paper low level (reward flat after ~8000)
python scripts/rl_games/play.py  --task Spot-Paper-Play-v0 --cameras --checkpoint <.pth> --viz kit
SPOT_VSLAM_LOW_LEVEL_CKPT=<low-level .pth> python scripts/rl_games/train.py --task Spot-Nav-MapA-v0 --cameras
python scripts/rl_games/play.py --task Spot-Nav-MapA-GT-Play-v0 --cameras --checkpoint <high-level .pth> --viz kit

# SB3 high level (rl_vslam_v0 and the current paper high-level checkpoint)
python scripts/high_level/train_high_level_ppo.py --paper --task Spot-Paper-high-level-MapA-v0 \
  --low_level_task Spot-Paper-high-level-MapA-v0 --checkpoint logs/rl_games/spot_paper_ppo/<run>/nn/spot_paper_ppo.pth
python scripts/slam/test_vslam.py

# Paper tests (each starts ORB-SLAM3 itself; set LOW_CKPT and HIGH_CKPT)
bash scripts/tools/run_navigation_test.sh   # Table 9: success %, collisions, steps, time per map
bash scripts/tools/run_system_load_test.sh  # Tables 7-8: FPS, CPU, RAM, disk for RL / +VSLAM / +TSDF
python scripts/tools/summarize_system_load.py output/system_load_test

# Nav2
ros2 launch source/spot_vslam/launch/start_nav2_slam.launch.py

# Lint / format (ruff + codespell, configured in .pre-commit-config.yaml)
pre-commit run --all-files
```

Use `~/IsaacLab/isaaclab.sh -p` or `uv run` instead of bare `python` whenever the Isaac Lab venv isn't already active.

There is no test suite; "verification" for this project means loading task configs and running a short headless
episode (a few `env.step`s, or `scripts/rl_games/train.py --max_iterations 2`, then delete the output). When
restructuring configs, compare each task's resolved config (`load_cfg_from_registry(...).to_dict()`, term order
included) before and after: a trained policy must keep loading under its task id.

Assets (`SPOT_USD_PATH`, `WAREHOUSE_USD_PATH` in `spot_vslam/assets/__init__.py`) come from the local Isaac Sim
asset pack via `$ISAACSIM_ASSET_ROOT`; the low-level locomotion checkpoint comes from `--checkpoint` or
`$SPOT_VSLAM_LOW_LEVEL_CKPT` (final ones are in `policies/`, e.g. `policies/walls_soft/model.pth`). Neither should be hardcoded back to machine-specific paths.

## Architecture

### Layout

```
docs/              ROADMAP.md (what to fix / build next), SYSTEM.md (diagram, policy inputs/outputs)
scripts/           README.md = run commands
  rl_games/        train.py, play.py for any task (low level, navigation)
  high_level/      SB3 high level on a frozen low level: train_high_level_ppo.py, load_high_level.py (plays both
                   policies with ORB-SLAM3, records load / SLAM error / TSDF)
  slam/            ORB-SLAM3 / Nav2 test loops on Spot-test180-v0
  tools/           paper tables (run_navigation_test.sh, run_system_load_test.sh, summarize_system_load.py), GIF
  legacy/          old scripts nothing calls
output/            git-ignored run outputs: checkpoints/high_level{,_paper}, tensorboard/, dense_map/, test results
baselines/         frozen policy snapshots + results (rl_vslam_v0)
policies/          final paper-version checkpoints (low levels, SB3 high level) + their yaml; README has sources
references/        the paper (main.pdf), its bibliography, and paper-vs-code notes
source/spot_vslam/
  config/          Isaac Sim extension manifest
  launch/          ROS 2 launch file + nav2_params.yaml
  ros2/            orbslam3_rgbd_node / orbslam3_stereo_node (ORB-SLAM3 wrappers, C++, one source)
  spot_vslam/
    assets/        USD paths, spot.py (Spot cfg for slam_test), maps.py (PAPER_MAPS: start/goal per map)
    envs/          ManagerBasedRLEnv subclass with the ROS 2 / ORB-SLAM3 managers wired in
    slam/          Ros2Manager, OrbSlamSubscriberManager, DenseMapManager (TSDF)
    tasks/         Isaac Lab layout: <task>/<task>_env_cfg.py (shared MDP), <task>/mdp/ (own terms, re-exports
                   isaaclab.envs.mdp), <task>/config/spot/ (variants, gym.register, agents/*.yaml)
      locomotion/  low level. config/spot/paper_env_cfg.py (353-dim), baseline_v0_env_cfg.py (245-dim, frozen)
      navigation/  high level, manager-based. mdp/actions.py runs the frozen low level (rl_games .pth)
      slam_test/   SLAM test env
    legacy/        high_level/: SB3 wrappers (high_level_vec_env.py, paper_vec_env.py) for existing checkpoints,
                   low_level_runner.py (frozen rl_games low level for the SB3 scripts)
    utils.py       shared helpers (yaw_from_quat_xyzw, step_obs_info)
```

### Env / manager wiring

`spot_vslam.envs.ManagerBasedRLEnv` (`envs/rl_env.py`) is a thin subclass of Isaac Lab's
`ManagerBasedRLEnv` that adds two *optional* managers, constructed from the env cfg and updated every step
right after the command manager and before interval events/observations:

- `cfg.ros2` → `Ros2Manager` (publishes camera images / TF to ROS 2)
- `cfg.slam_subscriber` → `OrbSlamSubscriberManager` (receives ORB-SLAM3 pose estimates back into the sim)

Task configs that don't set `ros2`/`slam_subscriber` run as plain locomotion envs; the ROS 2 machinery only
activates when a task config opts in (the `_PLAY` / `-Play-v0` configs set `ros2`). Scripts must not create a second
`Ros2Manager`; `scripts/high_level/load_high_level.py` still does (the rl_vslam_v0 tables were measured that way).

`Ros2Manager` builds ROS 2 image messages itself via `numpy_to_imgmsg()` (in `slam/ros2_manager.py`) instead
of `cv_bridge`: the ROS 2 Jazzy `cv_bridge` binary is built against NumPy 1.x and segfaults under the NumPy 2.x
that Isaac Sim / Isaac Lab 3.0 require.

### Task registration

Each task package registers its ids in `tasks/<task>/config/spot/__init__.py` (`tasks/__init__.py` imports them
via `import_packages`). Task ids of trained policies never change.

| Task ID | Config (under `tasks/`) |
| --- | --- |
| `Spot-Paper-v0` / `-Play-v0` | `locomotion/config/spot/paper_env_cfg.py` (paper low level) |
| `Spot-Paper-Rough-v0`, `Spot-Paper-Obstacles-v0` | same reward / PPO, other training terrain (Isaac Lab Spot terrain; pillars) |
| `Spot-Paper-Walls{,Soft,SoftNoise,SoftSym,SoftTrot}-v0` | wall terrain; Soft: contact costs instead of ending; Noise: sensor noise; Sym: + even-diagonal penalty; Trot: + per-foot duty and minimum swing time |
| `Spot-Paper-high-level-Map{A,B}-v0` / `-Play-v0` | same; low-level env under the SB3 high level |
| `Spot-Nav-Map{A,B}-v0` / `-Play-v0` / `-Stereo-Play-v0` / `-cuVSLAM-Play-v0` / `-GT-Play-v0` | `navigation/config/spot/navigation_env_cfg.py` (train: GT pose; Play: ORB-SLAM3 RGB-D; Stereo-Play: ORB-SLAM3 stereo; cuVSLAM-Play: cuVSLAM stereo; GT-Play: GT pose, no ROS 2). Play tasks start at the map start facing the goal |
| `Spot-Vslam-Depth-v0` / `-Play-v0` | `locomotion/config/spot/baseline_v0_env_cfg.py` (rl_vslam_v0 low level) |
| `Spot-Vslam-high-level-v0` / `-Play-v0` / `-Warehouse-Play-v0` | same (rl_vslam_v0 SB3 high level; Warehouse = Map A, spawn (0, 0)) |
| `Spot-test180-v0` | `slam_test/slam_test_env_cfg.py` |

### Isaac Lab 3.0 migration notes

The project was ported from Isaac Lab 2.x to 3.0 (`~/IsaacLab`, branch `release/3.0.0`) on 2026-09-26. Key
changes to be aware of when touching env/asset/manager code:

- Quaternions are now XYZW (was WXYZ) — this also means policies trained pre-migration need retraining.
- ProxyArray accesses need `.torch`; write APIs go through `*_to_sim_index`.
- `ViewerCfg` was replaced by `sim.default_visualizer_cfg`; `RecordVideo` by `apply_video_recording`.
- The robot is the stock Isaac Sim Spot (`SPOT_USD_PATH`) with cameras spawned via `PinholeCameraCfg` — the
  original custom USDs (`spot_with_camera.usd`, `flat_maze.usd`, `corrider_map.usd`, ...) are lost.
- The arena is `Simple_Warehouse/warehouse.usd` (`WAREHOUSE_USD_PATH`, prim `/World/Maze`) — this is paper
  Map A, not Map B.
- The local asset pack has no `IsaacLab/Environments/` folder, so `GroundPlaneCfg` and `terrain_type="plane"` fail:
  use a one-tile flat `TerrainGeneratorCfg` instead.
- **Known crash:** setting `events = None` or `rewards = None` on a `ManagerBasedRLEnvCfg` makes `sim.reset()`
  raise `AttributeError: 'NoneType' object has no attribute '__dict__'` from
  `ManagerBase._resolve_terms_callback`, even though the docstring says `None` is allowed. Leave `events` at
  its default and give `rewards` an empty `@configclass` instead. `curriculum = None` still works fine.
- **Known bug (open, docs/ROADMAP.md fixes):** in terrain-generator envs (5×5 tiles × 8 m), robots spawn at tile
  origins up to ±16 m from center, but the warehouse only spans x∈[-12,12], y∈[-18,20.8] — some envs spawn
  outside it. The paper map tasks avoid this (one map copy per env); fix it if a terrain-grid task gets an arena.

Verified 2026-10-07 (after the restructure): all task configs load; `Spot-Paper-v0` (353), `Spot-Vslam-Depth-v0`
(245), `Spot-Paper-high-level-MapB-v0`, `Spot-test180-v0`, `Spot-Nav-MapA-v0` / `-GT-Play-v0` step headless;
`Spot-Nav-MapA-v0` trains (~100 FPS, 16 envs) and plays; the SB3 `load_high_level.py --paper` path runs.
