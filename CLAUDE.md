# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Talk simply.** Use short, plain sentences and avoid jargon and long explanations, unless the user asks for more detail.

## Project

Isaac Lab 3.0 environments for running visual SLAM (ORB-SLAM3 over ROS 2, with Nav2) on the Boston Dynamics Spot
quadruped. The project was migrated from Isaac Lab 2.x to 3.0 on 2026-09-26 (see "Isaac Lab 3.0 migration notes"
below) and `PLAN.md` at the repo root tracks the remaining refactor work — read it before making structural changes
to the task configs.

## Paper reproduction

The project reproduces `references/main.pdf` (Li & Jeong, IJCAS 2026); `references/README.md` maps paper settings to
code. The original author repo (github.com/BernieMHao/slam_project) has no USD files, and its code is an earlier
version than the paper (no goal in the high level, 245-dim low-level obs).

Two policy sets exist side by side. Never change one in a way that breaks the other:

- **`rl_vslam_v0`** (baseline, git tag `baseline/rl-vslam-v0`, files in `baselines/rl_vslam_v0/`): the old tasks
  (`Spot-Vslam-Depth-v0`, `Spot-Vslam-high-level-*`), 245-dim low-level obs, 11-dim high level with no goal.
- **Paper version** (`tasks/spot_vslam/paper_cfg.py`, `high_level/paper_vec_env.py`, `mdp/paper_rewards.py`):
  `Spot-Paper-*` tasks. Low level: 353-dim obs (80x60 depth), reward Eqs. 2-6 with Table 3 weights. High level:
  9-dim obs with goal (Table 5), [256, 128] ELU, Table 6 PPO. The paper gives no high-level reward or training map,
  so those are our own (goal progress + success bonus; trained on Map A or B).

Paper maps (`PAPER_MAPS` in `paper_cfg.py`; each map has one fixed start and goal, never random goals):

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

# Train / play via Isaac Lab 3.0's unified entry points (tasks come from the package's gym entry point)
isaaclab train --rl_library rl_games --task Spot-Vslam-v0
isaaclab play  --rl_library rl_games --task Spot-Vslam-Play-v0 --checkpoint latest

# Project scripts (these attach the ROS 2 manager during play). Isaac Lab 3.0 runs headless by default:
# camera tasks need --cameras (train.py) and a window needs --viz kit.
python scripts/low_level/train.py --task Spot-Vslam-v0
python scripts/low_level/play.py  --task Spot-Vslam-Play-v0 --checkpoint /path/to/model.pth
python scripts/slam/test_vslam.py

# Paper policies (~13 h low level, then the high level). train.py prints the reward every epoch
# (--quiet_rewards turns it off, --reward_terms_every N sets the per-term breakdown interval).
python scripts/low_level/train.py --task Spot-Paper-v0 --num_envs 512 --cameras
python scripts/high_level/train_high_level_ppo.py --paper --task Spot-Paper-high-level-MapA-v0 \
  --low_level_task Spot-Paper-high-level-MapA-v0 --checkpoint logs/rl_games/spot_paper_ppo/<run>/nn/spot_paper_ppo.pth

# Paper tests (each starts ORB-SLAM3 itself; set LOW_CKPT and HIGH_CKPT)
bash scripts/high_level/run_navigation_test.sh   # Table 9: success %, collisions, steps, time per map
bash scripts/high_level/run_system_load_test.sh  # Tables 7-8: FPS, CPU, RAM, disk for RL / +VSLAM / +TSDF
python scripts/high_level/summarize_system_load.py output/system_load_test

# Nav2
ros2 launch source/spot_vslam/launch/start_nav2_slam.launch.py

# Lint / format (ruff + codespell, configured in .pre-commit-config.yaml)
pre-commit run --all-files
```

Use `~/IsaacLab/isaaclab.sh -p` or `uv run` instead of bare `python` whenever the Isaac Lab venv isn't already active.

There is no test suite; "verification" for this project means loading task configs and running a short headless
episode (e.g. `isaaclab play --task <id> --headless` or one of `scripts/slam/test_*.py`).

Assets (`SPOT_USD_PATH`, `WAREHOUSE_USD_PATH` in `spot_vslam/assets/__init__.py`) come from the local Isaac Sim
asset pack via `$ISAACSIM_ASSET_ROOT`; the low-level locomotion checkpoint comes from `--checkpoint` or
`$SPOT_VSLAM_LOW_LEVEL_CKPT`. Neither should be hardcoded back to machine-specific paths.

## Architecture

### Layout

```
scripts/
  low_level/       train.py, play.py (low-level locomotion + Vslam tasks, rl_games; play.py attaches ROS 2)
  slam/            ORB-SLAM3 / Nav2 test loops (test_vslam.py, test_rotate_180.py, test_nav2_loop.py, ...)
  orb_aware/       rule-based ORB-aware navigation
  high_level/      high-level navigation policy (SB3 PPO on a frozen low-level policy): train / evaluate;
                   load_high_level.py plays both policies with ORB-SLAM3 and records load / SLAM error / TSDF
  tools/           USD helpers (add_light_to_spot_usd.py)
baselines/         frozen policy snapshots + results (rl_vslam_v0)
references/        the paper (main.pdf), its bibliography, and paper-vs-code notes
source/spot_vslam/
  config/          Isaac Sim extension manifest
  launch/          ROS 2 launch file for Nav2
  spot_vslam/
    assets/        Spot-with-camera ArticulationCfg, textures, usd/ (USD files, git-ignored)
    envs/          ManagerBasedRLEnv subclass with the ROS 2 / ORB-SLAM3 managers wired in
    managers/      Ros2Manager, OrbSlamSubscriberManager, DenseMapManager, ROS 2 command/observation terms
    mdp/           custom reward / observation functions
    high_level/    high-level policy wrappers (vec env, RL env)
    tasks/spot_vslam/
      __init__.py  gym.register(...) for every task id
      agents/      rl_games configs
      params/      nav2_params.yaml
      *_cfg.py     per-task env configs (each ~200-800 lines, largely copy-pasted — see PLAN.md step 1)
```

### Env / manager wiring

`spot_vslam.envs.ManagerBasedRLEnv` (`envs/rl_env.py`) is a thin subclass of Isaac Lab's
`ManagerBasedRLEnv` that adds two *optional* managers, constructed from the env cfg and updated every step
right after the command manager and before interval events/observations:

- `cfg.ros2` → `Ros2Manager` (publishes camera images / TF to ROS 2)
- `cfg.slam_subscriber` → `OrbSlamSubscriberManager` (receives ORB-SLAM3 pose estimates back into the sim)

Task configs that don't set `ros2`/`slam_subscriber` run as plain locomotion envs; the ROS 2 machinery only
activates when a task config opts in, which is why `scripts/low_level/play.py` (ROS 2 attached) and
`isaaclab play` (no ROS 2) can run the same task ids.

`Ros2Manager` builds ROS 2 image messages itself via `numpy_to_imgmsg()` (in `managers/ros2_manager.py`) instead
of `cv_bridge`: the ROS 2 Jazzy `cv_bridge` binary is built against NumPy 1.x and segfaults under the NumPy 2.x
that Isaac Sim / Isaac Lab 3.0 require.

### Task registration

Every task id is registered in `tasks/spot_vslam/__init__.py` via `gym.register(...)`, pointing at a
`*_cfg.py` module + class and an `agents/*.yaml` rl_games config. Task id → config:

| Task ID | Config |
| --- | --- |
| `Spot-v0` / `Spot-Play-v0` | `rough_env_cfg_v0_rough` (low-level locomotion) |
| `Spot-Vslam-v0` / `-Play-v0` | `simulate_vslam_env_cfg` |
| `Spot-Vslam-Deploy-v0` / `-Play-v0` | `walk_model_run_on_vslam` |
| `Spot-Vslam-Depth-v0` / `-Play-v0` | `use_depth_training` |
| `Spot-Vslam-Depth-2-v0` / `-Play-v0` | `use_depth_training_2` |
| `Spot-Vslam-Depth-3-v0` / `-Play-v0` | `vslam_training` |
| `Spot-Vslam-high-level-v0` / `-Play-v0` | `use_depth_training` (high-level) |
| `Spot-Vslam-high-level-Warehouse-Play-v0` | `use_depth_training` (rl_vslam_v0 tests, Map A, spawn (0, 0)) |
| `Spot-Paper-v0` / `-Play-v0` | `paper_cfg` (paper low level) |
| `Spot-Paper-high-level-MapA-v0` / `-Play-v0` | `paper_cfg` (paper high level, Map A) |
| `Spot-Paper-high-level-MapB-v0` / `-Play-v0` | `paper_cfg` (paper high level, Map B) |
| `Spot-test180-v0`, `Isaac-Velocity-Rough-Spot-v0` | `spot_vslam_test_cfg` |

These configs currently duplicate `SpotRewardsCfg`, `SpotObservationsCfg`, `COBBLESTONE_ROAD_CFG` (7 files),
`SpotTerminationsCfg` (6 files), `get_orb_slam_pose`/`get_orb_slam_status` (4 files), and
`SpotEventCfg`/`SpotActionsCfg` (3 files) — PLAN.md step 1 covers consolidating these into a shared
`tasks/spot_vslam/base_cfg.py` and `mdp/`, without changing any task's observation layout (a policy trained
under one task id must keep loading under that id).

### Isaac Lab 3.0 migration notes

The project was ported from Isaac Lab 2.x to 3.0 (`~/IsaacLab`, branch `release/3.0.0`) on 2026-09-26. Key
changes to be aware of when touching env/asset/manager code:

- Quaternions are now XYZW (was WXYZ) — this also means policies trained pre-migration need retraining.
- ProxyArray accesses need `.torch`; write APIs go through `*_to_sim_index`.
- `ViewerCfg` was replaced by `sim.default_visualizer_cfg`; `RecordVideo` by `apply_video_recording`.
- The robot is the stock Isaac Sim Spot (`SPOT_USD_PATH`) with cameras spawned via `PinholeCameraCfg` — the
  original custom USDs (`spot_with_camera.usd`, `flat_maze.usd`, `corrider_map.usd`, ...) are lost.
- The arena is `Simple_Warehouse/warehouse.usd` (`WAREHOUSE_USD_PATH`, prim `/World/Maze`) — this is paper
  Map A, not Map B. Frustum raycasters use `MultiMeshRayCasterCfg` over the arena root.
- **Known crash:** setting `events = None` or `rewards = None` on a `ManagerBasedRLEnvCfg` makes `sim.reset()`
  raise `AttributeError: 'NoneType' object has no attribute '__dict__'` from
  `ManagerBase._resolve_terms_callback`, even though the docstring says `None` is allowed. Leave `events` at
  its default and give `rewards` an empty `@configclass` instead. `curriculum = None` still works fine.
- **Known bug (open, PLAN.md step 2):** in terrain-generator envs (5×5 tiles × 8 m), robots spawn at tile
  origins up to ±16 m from center, but the warehouse only spans x∈[-12,12], y∈[-18,20.8] — some envs spawn
  outside it (`Spot-Vslam-Deploy-Play-v0` shows open sky). The fix is to shrink the spawn area to fit the
  arena, not to scale the warehouse.

The original 15 task configs load; `Spot-test180-v0` and `Spot-Vslam-Deploy-Play-v0` are verified to run headless.
All `Spot-Paper-*` tasks build and step (353-dim obs; per-env map copies checked on Map A and B).
