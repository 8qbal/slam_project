#!/usr/bin/env bash
# Record one sensor dataset (docs/cuvslam.md "Benchmark", ROADMAP section 2): a fixed drive from the map start to the goal
# (cuvslam_frame_check.py --to_goal drives on ground truth, so no SLAM and no policy is needed), with the topics
# Ros2Manager publishes: left + right image, depth, camera_info, and /tf (odom -> base_link is the GT pose of the robot).
# Stamps are Ros2Manager's fixed 30 Hz steps (one per published frame), not sim time (see docs/cuvslam.md "Dataset").
#
# Usage (from the repo root):  SPOT_VSLAM_LOW_LEVEL_CKPT=<low-level .pth> bash scripts/slam/record_dataset.sh [A|B] [name]
# Output: output/datasets/<name>/ (bag, gt.tum from the frame check, run.log). Default name: map<MAP>_stereo.
cd "$(dirname "$0")/../.."
source /opt/ros/jazzy/setup.bash
MAP=${1:-A}
NAME=${2:-map${MAP}_stereo}
OUT=output/datasets/$NAME
: "${SPOT_VSLAM_LOW_LEVEL_CKPT:?set SPOT_VSLAM_LOW_LEVEL_CKPT to the low-level .pth}"
rm -rf "$OUT"; mkdir -p "$OUT"

# no ORB-SLAM3 node: nothing but the sensors is recorded
ros2 bag record -o "$OUT/bag" --compression-mode file --compression-format zstd \
  /spot0/camera/image_raw /spot0/camera/right/image_raw /spot0/camera/depth /spot0/camera/camera_info /tf /tf_static \
  > "$OUT/bag.log" 2>&1 &
BAG=$!
sleep 5
rm -rf output/cuvslam_frame_check
uv run --no-sync python scripts/slam/cuvslam_frame_check.py --task "Spot-Nav-Map${MAP}-Stereo-Play-v0" --to_goal > "$OUT/run.log" 2>&1
echo "drive exit=$?"
sleep 3
kill -TERM $BAG; wait $BAG 2>/dev/null  # SIGINT is ignored by background jobs of a script
cp output/cuvslam_frame_check/gt.tum output/cuvslam_frame_check/gt.txt "$OUT/" 2>/dev/null
ros2 bag info "$OUT/bag" | tee "$OUT/bag_info.txt"
