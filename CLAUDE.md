# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Isaac Lab 3.0 environments for running visual SLAM (ORB-SLAM3 over ROS 2, with Nav2) on the Boston Dynamics Spot
quadruped. The project was migrated from Isaac Lab 2.x to 3.0 on 2026-09-26 (see "Isaac Lab 3.0 migration notes"
below) and `PLAN.md` at the repo root tracks the remaining refactor work — read it before making structural changes
to the task configs.

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

# Project scripts (these attach the ROS 2 manager during play)
python scripts/low_level/train.py --task Spot-Vslam-v0
python scripts/low_level/play.py  --task Spot-Vslam-Play-v0 --checkpoint /path/to/model.pth
python scripts/slam/test_vslam.py

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
  high_level/      high-level navigation policy (SB3 PPO on a frozen low-level policy): train / evaluate
  tools/           USD helpers (add_light_to_spot_usd.py)
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
- The arena is `Simple_Warehouse` (`WAREHOUSE_USD_PATH`, prim `/World/Maze`); frustum raycasters use
  `MultiMeshRayCasterCfg` over the arena root.
- **Known crash:** setting `events = None` or `rewards = None` on a `ManagerBasedRLEnvCfg` makes `sim.reset()`
  raise `AttributeError: 'NoneType' object has no attribute '__dict__'` from
  `ManagerBase._resolve_terms_callback`, even though the docstring says `None` is allowed. Leave `events` at
  its default and give `rewards` an empty `@configclass` instead. `curriculum = None` still works fine.
- **Known bug (open, PLAN.md step 2):** in terrain-generator envs (5×5 tiles × 8 m), robots spawn at tile
  origins up to ±16 m from center, but the warehouse only spans x∈[-12,12], y∈[-18,20.8] — some envs spawn
  outside it (`Spot-Vslam-Deploy-Play-v0` shows open sky). The fix is to shrink the spawn area to fit the
  arena, not to scale the warehouse.

15/15 task configs load; `Spot-test180-v0` and `Spot-Vslam-Deploy-Play-v0` are verified to run headless.
