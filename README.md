# Spot Visual SLAM (Isaac Lab 3.0)

Isaac Lab environments for running visual SLAM (ORB-SLAM3 over ROS 2, with Nav2) on the Boston Dynamics Spot quadruped.

## Layout

```
scripts/
  rl_games/        train.py, play.py (rl_games with the ROS 2 manager attached)
  slam/            ORB-SLAM3 / Nav2 test loops (test_vslam.py, test_rotate_180.py, test_nav2_loop.py, ...)
  orb_aware/       rule-based ORB-aware navigation
  high_level/      high-level navigation policy training / evaluation
  tools/           USD helpers (add_light_to_spot_usd.py)
source/spot_vslam/
  pyproject.toml   package metadata + `isaaclab.tasks` entry point
  config/          Isaac Sim extension manifest
  launch/          ROS 2 launch file for Nav2
  spot_vslam/
    assets/        Spot-with-camera ArticulationCfg, textures, usd/ (USD files, git-ignored)
    envs/          ManagerBasedRLEnv with the ROS 2 / ORB-SLAM3 managers
    managers/      Ros2Manager, OrbSlamSubscriberManager, DenseMapManager, ROS 2 terms
    mdp/           custom rewards / observations
    high_level/    high-level policy wrappers (vec env, RL env)
    tasks/spot_vslam/
      __init__.py  gym.register(...) for every task below
      agents/      rl_games configs
      params/      nav2_params.yaml
      *_cfg.py     env configs
```

## Installation

1. Install Isaac Lab 3.0 (this machine: `~/IsaacLab`, venv at `~/IsaacLab/.venv`).
2. Install this package into that environment:

   ```bash
   ~/IsaacLab/isaaclab.sh -p -m pip install -e source/spot_vslam
   # or, with uv from this directory:
   uv sync
   ```

3. Copy the USD files (robot + maps) into `source/spot_vslam/spot_vslam/assets/usd/`, or export
   `SPOT_VSLAM_USD_DIR=/path/to/usd`. They are git-ignored; see `assets/usd/README.md` for the list.
4. Source ROS 2 before running anything that uses the ROS 2 / SLAM managers:

   ```bash
   source /opt/ros/jazzy/setup.bash
   ```

## Tasks

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

## Running

```bash
# Isaac Lab 3.0 unified entry points (tasks are discovered through the package entry point)
isaaclab train --rl_library rl_games --task Spot-Vslam-v0
isaaclab play  --rl_library rl_games --task Spot-Vslam-Play-v0 --checkpoint latest

# Project scripts (ROS 2 manager attached during play)
python scripts/rl_games/train.py --task Spot-Vslam-v0
python scripts/rl_games/play.py  --task Spot-Vslam-Play-v0 --checkpoint /path/to/model.pth
python scripts/slam/test_vslam.py

# Nav2
ros2 launch source/spot_vslam/launch/start_nav2_slam.launch.py
```

Use `~/IsaacLab/isaaclab.sh -p` instead of `python` if the Isaac Lab venv is not active.

## Development

```bash
pre-commit run --all-files
```
