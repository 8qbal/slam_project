# Spot Visual SLAM (Isaac Lab 3.0)

Isaac Lab environments for visual SLAM (ORB-SLAM3 over ROS 2) with hierarchical RL navigation on the Boston Dynamics
Spot quadruped. Reproduces and extends `references/main.pdf`.

## Layout

```
docs/              ROADMAP.md (what to fix / build next), SYSTEM.md (diagram, policy inputs/outputs)
scripts/           see scripts/README.md for how to train and run the policies
  rl_games/        train.py, play.py: any task (low level or navigation)
  high_level/      SB3 high-level training / play (rl_vslam_v0 and the current paper high level)
  slam/            ORB-SLAM3 / Nav2 test loops
  tools/           paper tables (Tables 7-9), GIF
  legacy/          old scripts, not part of the pipeline
baselines/         frozen policy snapshots + results (rl_vslam_v0)
references/        the paper (main.pdf) and paper-vs-code notes
output/            (git-ignored) run outputs: checkpoints/, tensorboard/, dense_map/, test results
source/spot_vslam/
  launch/          ROS 2 launch file + params for Nav2
  ros2/            orbslam3_rgbd_node (ORB-SLAM3 RGB-D wrapper, C++)
  spot_vslam/
    assets/        USD paths, Spot config, paper maps (PAPER_MAPS)
    envs/          ManagerBasedRLEnv with optional ROS 2 / ORB-SLAM3 managers
    slam/          Ros2Manager, OrbSlamSubscriberManager, DenseMapManager (TSDF)
    tasks/         Isaac Lab task layout: <task>/<task>_env_cfg.py, <task>/mdp/, <task>/config/spot/
      locomotion/  low-level policy: walk at a commanded velocity, seeing depth ahead
      navigation/  high-level policy: reach the map goal by commanding the frozen low level
      slam_test/   SLAM test env for scripts/slam
    legacy/        SB3 high-level wrappers, kept so existing high-level checkpoints still run
```

## Installation

1. Install Isaac Lab 3.0 (this machine: `~/IsaacLab`).
2. Install this package into that environment: `uv sync` (or `~/IsaacLab/isaaclab.sh -p -m pip install -e source/spot_vslam`).
3. Assets come from the local Isaac Sim asset pack (`ISAACSIM_ASSET_ROOT`, e.g. `~/isaacsim_assets/Assets/Isaac/6.0`).
4. Source ROS 2 before running anything that uses ROS 2 / SLAM: `source /opt/ros/jazzy/setup.bash`.

## Tasks

| Task ID | What | Config |
| --- | --- | --- |
| `Spot-Paper-v0` / `-Play-v0` | paper low level (353-dim obs) | `locomotion/config/spot/paper_env_cfg.py` |
| `Spot-Nav-Map{A,B}-v0` | paper high level, manager-based (GT pose, condition A) | `navigation/config/spot/navigation_env_cfg.py` |
| `Spot-Nav-Map{A,B}-Play-v0` | evaluate with the ORB-SLAM3 pose | same |
| `Spot-Nav-Map{A,B}-GT-Play-v0` | evaluate with the GT pose, no ROS 2 | same |
| `Spot-Paper-high-level-Map{A,B}-v0` / `-Play-v0` | low-level env under the SB3 high-level wrapper | `locomotion/config/spot/paper_env_cfg.py` |
| `Spot-Vslam-Depth-v0` / `-Play-v0` | rl_vslam_v0 low level (245-dim obs) | `locomotion/config/spot/baseline_v0_env_cfg.py` |
| `Spot-Vslam-high-level-v0` / `-Play-v0` / `-Warehouse-Play-v0` | rl_vslam_v0 SB3 high level | same |
| `Spot-test180-v0` | SLAM test env | `slam_test/slam_test_env_cfg.py` |

How to run them: [`scripts/README.md`](scripts/README.md).

## Development

```bash
pre-commit run --all-files
```
