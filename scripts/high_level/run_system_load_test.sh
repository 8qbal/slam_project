#!/usr/bin/env bash
# Paper Tables 7-8 (system load) with the paper policies: 3 configs x 5 runs, 2000 high-level steps each.
#   rl_only        both policies, no ROS 2 and no ORB-SLAM3 (the high level gets the ground-truth pose)
#   rl_vslam       + ORB-SLAM3 (the high level gets the ORB-SLAM3 pose)
#   rl_vslam_tsdf  + TSDF dense mapping with ground-truth poses (paper Sec. 3)
# The robot drives from the map start to the goal; when it gets there it is put back at the start and keeps going.
# At the end the tables are printed by summarize_system_load.py.
#
# Usage (from the repo root):
#   LOW_CKPT=logs/rl_games/spot_paper_ppo/<run>/nn/spot_paper_ppo.pth \
#   HIGH_CKPT=checkpoints_high_level_paper/high_level_policy.zip \
#   bash scripts/high_level/run_system_load_test.sh
# Optional: MAP=A (default) or B, TRIALS=5, STEPS=2000, OUT=output/system_load_test
cd "$(dirname "$0")/../.."
source /opt/ros/jazzy/setup.bash
source ~/orbslam3/ws/install/setup.bash
export ORB_SLAM3_ROOT=~/orbslam3/ORB_SLAM3
set -u  # after sourcing: the ROS setup scripts reference unset variables

: "${LOW_CKPT:?set LOW_CKPT to the paper low-level .pth}"
: "${HIGH_CKPT:?set HIGH_CKPT to the paper high-level .zip}"
MAP=${MAP:-A}
TRIALS=${TRIALS:-5}
STEPS=${STEPS:-2000}
OUT=${OUT:-output/system_load_test}
TASK=Spot-Paper-high-level-Map${MAP}-Play-v0

run() {  # run <config> <seed> <use_orb 0|1> [extra args...]
  local cfg=$1 seed=$2 use_orb=$3; shift 3
  local dir=$OUT/${cfg}/seed${seed}
  rm -rf "$dir"; mkdir -p "$dir"
  # psutil measures disk I/O for the whole system: let the previous run's TSDF meshes finish writing first
  sync; sleep 10
  local lpid=""
  if [ "$use_orb" = 1 ]; then
    ros2 launch source/spot_vslam/ros2/spot_orbslam3/launch/orbslam3_rgbd.launch.py robot_index:=0 > "$dir/orbslam3.log" 2>&1 &
    lpid=$!
    sleep 8
  fi
  echo "[$(date +%T)] $cfg seed=$seed (map $MAP)"
  uv run --no-sync python scripts/high_level/load_high_level.py --paper --low_level_task "$TASK" \
    --low_level_checkpoint "$LOW_CKPT" --high_level_checkpoint "$HIGH_CKPT" --num_envs 1 \
    --steps "$STEPS" --deterministic_policy --seed "$seed" --dense_export_dir "$dir" "$@" > "$dir/run.log" 2>&1
  echo "  exit=$?"
  if [ -n "$lpid" ]; then
    kill -INT $lpid; sleep 5
    kill $(pgrep -x orbslam3_rgbd_n) 2>/dev/null
  fi
}

for seed in $(seq 1 "$TRIALS"); do
  run rl_only "$seed" 0 --no_ros2
  run rl_vslam "$seed" 1
  run rl_vslam_tsdf "$seed" 1 --enable_dense_mapping --dense_pose_source gt
done
echo "[$(date +%T)] all done"
uv run --no-sync python scripts/high_level/summarize_system_load.py "$OUT"
