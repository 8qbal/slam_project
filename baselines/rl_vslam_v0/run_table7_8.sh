#!/usr/bin/env bash
# Paper Tables 7-8 (system load) in the warehouse: 3 configs x 5 trials, 2000 high-level steps each.
# Usage (from the repo root): bash baselines/rl_vslam_v0/run_table7_8.sh   (set OUT=... to change the output dir)
cd "$(dirname "$0")/../.."
source /opt/ros/jazzy/setup.bash
source ~/orbslam3/ws/install/setup.bash
export ORB_SLAM3_ROOT=~/orbslam3/ORB_SLAM3
set -u  # after sourcing: the ROS setup scripts reference unset variables

TASK=Spot-Vslam-high-level-Warehouse-Play-v0
LOW_CKPT=baselines/rl_vslam_v0/policies/low_level/spot_rough_ppo.pth
OUT=${OUT:-output/baseline_rl_vslam_v0_rerun}
STEPS=2000

run() {  # run <config> <seed> <use_orb 0|1> [extra args...]
  local cfg=$1 seed=$2 use_orb=$3; shift 3
  local dir=$OUT/${cfg}/seed${seed}
  rm -rf "$dir"; mkdir -p "$dir"
  local lpid=""
  if [ "$use_orb" = 1 ]; then
    ros2 launch source/spot_vslam/ros2/spot_orbslam3/launch/orbslam3_rgbd.launch.py robot_index:=0 > "$dir/orbslam3.log" 2>&1 &
    lpid=$!
    sleep 8
  fi
  echo "[$(date +%T)] $cfg seed=$seed"
  uv run --no-sync python scripts/high_level/load_high_level.py --low_level_task $TASK \
    --low_level_checkpoint $LOW_CKPT --high_level_checkpoint baselines/rl_vslam_v0/policies/high_level/high_level_policy.zip --num_envs 1 \
    --steps $STEPS --deterministic_policy --seed "$seed" --dense_export_dir "$dir" "$@" > "$dir/run.log" 2>&1
  echo "  exit=$?"
  if [ -n "$lpid" ]; then
    kill -INT $lpid; sleep 5
    kill $(pgrep -x orbslam3_rgbd_n) 2>/dev/null
  fi
}

for seed in 1 2 3 4 5; do
  run rl_only "$seed" 0
  run rl_vslam "$seed" 1
  run rl_vslam_tsdf "$seed" 1 --enable_dense_mapping --dense_pose_source gt
done
echo "[$(date +%T)] all done"
