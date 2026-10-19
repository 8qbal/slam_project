#!/usr/bin/env bash
# Table 9 protocol with ORB-SLAM3 in the loop for an rl_games high level (condition B with ORB-SLAM3).
# ORB-SLAM3 keeps its origin after a reset, so it is restarted for every run: one env, one run, one node.
#
# Usage (from the repo root):
#   SPOT_VSLAM_LOW_LEVEL_CKPT=policies/walls_soft_mirror_turn/model.pth \
#   HIGH_CKPT=policies/nav_hall_racks/model.pth bash scripts/tools/run_nav_protocol_orbslam.sh
# Optional: MAPS="A B", TRIALS=5, MAX_STEPS=3000, VARIANT=Play (RGB-D) or Stereo-Play, OUT=output/nav_protocol_orbslam
cd "$(dirname "$0")/../.."
source /opt/ros/jazzy/setup.bash
source ~/orbslam3/ws/install/setup.bash
export ORB_SLAM3_ROOT=~/orbslam3/ORB_SLAM3
set -u

: "${SPOT_VSLAM_LOW_LEVEL_CKPT:?set the low-level .pth}"
: "${HIGH_CKPT:?set the high-level rl_games .pth}"
MAPS=${MAPS:-"A B"}
TRIALS=${TRIALS:-5}
VARIANT=${VARIANT:-Play}
LAUNCH=orbslam3_rgbd.launch.py; NODE=orbslam3_rgbd_n
if [ "$VARIANT" = Stereo-Play ]; then LAUNCH=orbslam3_stereo.launch.py; NODE=orbslam3_stereo; fi
OUT=${OUT:-output/nav_protocol_orbslam_${VARIANT}}
mkdir -p "$OUT"

for map in $MAPS; do
  for run in $(seq 1 "$TRIALS"); do
    log=$OUT/map${map}_run${run}.log
    ros2 launch source/spot_vslam/ros2/spot_orbslam3/launch/$LAUNCH robot_index:=0 > "$OUT/orbslam3_map${map}_run${run}.log" 2>&1 &
    lpid=$!
    sleep 8
    echo "[$(date +%T)] map $map run $run"
    uv run --no-sync python scripts/tools/eval_nav_protocol.py --task "Spot-Nav-Map${map}-${VARIANT}-v0" \
      --checkpoint "$HIGH_CKPT" --num_envs 1 --max_steps "${MAX_STEPS:-3000}" > "$log" 2>&1
    echo "  exit=$?  $(grep -E '^PROTO (RESULT|SLAM)' "$log" | cut -c1-200)"
    kill -INT $lpid; sleep 5
    kill $(pgrep -x "$NODE") 2>/dev/null
  done
done
echo "[$(date +%T)] done; logs in $OUT"
