#!/usr/bin/env bash
# Paper Table 9: paper policies on Map A and Map B, 5 runs each (seeds 1-5), ORB-SLAM3 restarted for every run.
# Each run goes from the map start to the goal (or 3000 steps) and appends one row to $OUT/navigation_results.csv;
# the Table 9 summary is printed at the end of every run.
#
# Usage (from the repo root):
#   LOW_CKPT=logs/rl_games/spot_paper_ppo/<run>/nn/spot_paper_ppo.pth \
#   HIGH_CKPT=checkpoints_high_level_paper/high_level_policy.zip \
#   bash scripts/high_level/run_navigation_test.sh
# Optional: MAPS="A B" (default), TRIALS=5, OUT=output/navigation_test, TSDF=1 (also run GT-pose TSDF mapping as in the paper)
cd "$(dirname "$0")/../.."
source /opt/ros/jazzy/setup.bash
source ~/orbslam3/ws/install/setup.bash
export ORB_SLAM3_ROOT=~/orbslam3/ORB_SLAM3
set -u  # after sourcing: the ROS setup scripts reference unset variables

: "${LOW_CKPT:?set LOW_CKPT to the paper low-level .pth}"
: "${HIGH_CKPT:?set HIGH_CKPT to the paper high-level .zip}"
MAPS=${MAPS:-"A B"}
TRIALS=${TRIALS:-5}
OUT=${OUT:-output/navigation_test}
TSDF=${TSDF:-0}
mkdir -p "$OUT"

for map in $MAPS; do
  for seed in $(seq 1 "$TRIALS"); do
    dir=$OUT/map${map}/seed${seed}
    rm -rf "$dir"; mkdir -p "$dir"
    ros2 launch source/spot_vslam/ros2/spot_orbslam3/launch/orbslam3_rgbd.launch.py robot_index:=0 > "$dir/orbslam3.log" 2>&1 &
    lpid=$!
    sleep 8
    extra=()
    if [ "$TSDF" = 1 ]; then extra=(--enable_dense_mapping --dense_pose_source gt); fi
    echo "[$(date +%T)] map $map seed $seed"
    uv run --no-sync python scripts/high_level/load_high_level.py --paper --navigation_test \
      --low_level_task "Spot-Paper-high-level-Map${map}-Play-v0" \
      --low_level_checkpoint "$LOW_CKPT" --high_level_checkpoint "$HIGH_CKPT" --num_envs 1 \
      --deterministic_policy --seed "$seed" --steps 3000 \
      --dense_export_dir "$dir" --results_csv "$OUT/navigation_results.csv" "${extra[@]}" > "$dir/run.log" 2>&1
    echo "  exit=$?  $(grep -E '^\[NAV-TEST\] (SUCCESS|FAIL)' "$dir/run.log")"
    kill -INT $lpid; sleep 5
    kill $(pgrep -x orbslam3_rgbd_n) 2>/dev/null
  done
done
sed -n '/Navigation test over/,$p' "$dir/run.log" | grep '^\[NAV-TEST\]'
echo "[$(date +%T)] all done; per-run rows in $OUT/navigation_results.csv"
