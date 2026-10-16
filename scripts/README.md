# Scripts

Run everything from the repo root. Use `uv run --no-sync python` (or `~/IsaacLab/isaaclab.sh -p`) if the Isaac Lab
venv is not active. Isaac Lab 3.0 is headless by default: add `--viz kit` for a window. Every task has a depth camera,
so pass `--cameras` (`isaaclab train` has no camera flag; use `scripts/rl_games/train.py`).

## Setup (for anything with ORB-SLAM3 / ROS 2)

```bash
source /opt/ros/jazzy/setup.bash
source ~/orbslam3/ws/install/setup.bash
export ORB_SLAM3_ROOT=~/orbslam3/ORB_SLAM3
# start ORB-SLAM3 in a second terminal before playing a policy with SLAM
ros2 launch source/spot_vslam/ros2/spot_orbslam3/launch/orbslam3_rgbd.launch.py robot_index:=0
# or stereo (left + right image, tasks Spot-Nav-Map{A,B}-Stereo-Play-v0):
# ros2 launch source/spot_vslam/ros2/spot_orbslam3/launch/orbslam3_stereo.launch.py robot_index:=0
```

## Paper policies

```bash
# 1. Low level (~13 h). Output: logs/rl_games/spot_paper_ppo/<run>/nn/spot_paper_ppo.pth
python scripts/rl_games/train.py --task Spot-Paper-v0 --num_envs 512 --cameras
python scripts/rl_games/play.py  --task Spot-Paper-Play-v0 --cameras --checkpoint <low-level .pth> --viz kit

# 2. High level (manager-based, 16 envs, ~500k steps). The frozen low level comes from SPOT_VSLAM_LOW_LEVEL_CKPT.
#    Output: logs/rl_games/spot_navigation_ppo/<run>/nn/
export SPOT_VSLAM_LOW_LEVEL_CKPT=policies/walls_soft_mirror_turn/model.pth  # the low level LOW_LEVEL_TASK is set for
python scripts/rl_games/train.py --task Spot-Nav-MapA-v0 --cameras

# 3. Play the high level: with ORB-SLAM3 (start it first, see Setup) or with the GT pose
python scripts/rl_games/play.py --task Spot-Nav-MapA-Play-v0    --cameras --checkpoint <high-level .pth> --viz kit
python scripts/rl_games/play.py --task Spot-Nav-MapA-GT-Play-v0 --cameras --checkpoint <high-level .pth> --viz kit
```

### SB3 high level (the current paper high-level checkpoint, and the paper tables)

```bash
python scripts/high_level/train_high_level_ppo.py --paper --task Spot-Paper-high-level-MapA-v0 \
  --low_level_task Spot-Paper-high-level-MapA-v0 --checkpoint <low-level .pth>
# --no_ros2 = no SLAM (GT pose); --enable_dense_mapping --dense_pose_source gt = TSDF map; --navigation_test = stop at goal
python scripts/high_level/load_high_level.py --paper --viz kit \
  --low_level_task Spot-Paper-high-level-MapA-Play-v0 --low_level_checkpoint <low-level .pth> \
  --high_level_checkpoint output/checkpoints/high_level_paper/high_level_policy.zip --deterministic_policy --steps 3000
```

## Baseline `rl_vslam_v0`

Policies and the Tables 7-8 runner are in `baselines/rl_vslam_v0/` (see its README):

```bash
python scripts/high_level/load_high_level.py --viz kit --low_level_task Spot-Vslam-high-level-Warehouse-Play-v0 \
  --low_level_checkpoint baselines/rl_vslam_v0/policies/low_level/spot_rough_ppo.pth \
  --high_level_checkpoint baselines/rl_vslam_v0/policies/high_level/high_level_policy.zip --deterministic_policy
```

## Tables and plots (`tools/`)

These use the SB3 high level and start ORB-SLAM3 themselves. Set `LOW_CKPT` and `HIGH_CKPT`.

```bash
bash scripts/tools/run_navigation_test.sh    # Table 9: success %, collisions, steps, time (Map A and B)
bash scripts/tools/run_system_load_test.sh   # Tables 7-8: FPS, CPU, RAM, disk for RL / +VSLAM / +TSDF
python scripts/tools/summarize_system_load.py output/system_load_test   # re-print Tables 7-8
python scripts/tools/play_hl_gif.py --low_level_checkpoint <.pth> --high_level_checkpoint <.zip> --out run.gif
```

## Folders

| Folder | What |
| --- | --- |
| `rl_games/` | train / play any task with rl_games (low level, navigation) |
| `high_level/` | SB3 high level on a frozen low level (rl_vslam_v0 and the current paper checkpoint) |
| `slam/` | ORB-SLAM3 / Nav2 test loops (`Spot-test180-v0`) |
| `tools/` | tables and plots |
| `legacy/` | old scripts, not part of the pipeline (see its README) |
