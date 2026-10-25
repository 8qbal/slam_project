#!/usr/bin/env bash
# Everything that needs a long evaluation after the 2026-10-25 fixes, one after the other (one Isaac Sim at a time:
# 15 GB RAM). No training. Every step has a timeout and a log; a failing step does not stop the next one.
#   1. terrain / noise study            run_terrain_eval.sh (20 gait_check runs)
#   2. Tables 7-8, 3 configs x 5 runs   run_system_load_test.sh ("RL only" was measured before the --no_ros2 fix)
#   3. A / B (2 seeds) with GT pose      eval_nav_protocol.py on Spot-Nav-Map{A,B}-GT-Play-v0, 8 envs
#   4. A / B (2 seeds) with ORB-SLAM3    eval_nav_protocol.py on Spot-Nav-Map{A,B}-Play-v0, 1 env per run, node restarted
#   5. cuVSLAM vs. ORB-SLAM3 stereo      cuvslam_frame_check.py --to_goal, 3 drives each on Map A and B, evo APE
# C (12-dim) is not evaluated with ORB-SLAM3 or GT: its quality inputs come from cuVSLAM.
#
# Usage (from the repo root):  bash scripts/tools/run_overnight_eval.sh
# Optional: OUT=output/overnight_eval  ONLY="1 2 3 4 5"  ORB_SEEDS=3
cd "$(dirname "$0")/../.."
source /opt/ros/jazzy/setup.bash
source ~/orbslam3/ws/install/setup.bash
export ORB_SLAM3_ROOT=~/orbslam3/ORB_SLAM3
set -u
OUT=${OUT:-output/overnight_eval}
ONLY=${ONLY:-"1 2 3 4 5"}
ORB_SEEDS=${ORB_SEEDS:-3}
mkdir -p "$OUT"
LOG=$OUT/progress.log
export SPOT_VSLAM_LOW_LEVEL_CKPT=policies/walls_soft_mirror_turn/model.pth
NAV=logs/rl_games/spot_navigation_ppo
declare -A POL=(
  [A]=policies/nav_hall_racks/model.pth
  [B_s1]=$NAV/2026-10-21_02-48-36/nn/last_spot_navigation_ppo_ep_8800_rew_31.86478.pth
  [B_s2]=$NAV/2026-10-21_22-32-20/nn/last_spot_navigation_ppo_ep_8800_rew_33.05179.pth
)
JITTER=(--jitter_xy 0.5 --jitter_yaw 0.5)

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
orb_start() {  # orb_start <log>
  ros2 launch source/spot_vslam/ros2/spot_orbslam3/launch/$ORB_LAUNCH robot_index:=0 use_viewer:=false > "$1" 2>&1 &
  ORB_PID=$!
  sleep 8
}
orb_stop() {
  kill -INT "$ORB_PID" 2>/dev/null; sleep 5
  kill "$(pgrep -x orbslam3_rgbd_n)" 2>/dev/null
  kill "$(pgrep -x orbslam3_stereo)" 2>/dev/null
  sleep 2
}
want() { [[ " $ONLY " == *" $1 "* ]]; }

if want 1; then
  say "1 terrain / noise study"
  OUT=$OUT/terrain_eval timeout 7200 bash scripts/tools/run_terrain_eval.sh >> "$LOG" 2>&1
  say "1 done (exit $?)"
fi

if want 2; then
  say "2 system load (Tables 7-8), WallsSoft + SB3 high level as in the 2026-10-13 run"
  LOW_CKPT=policies/walls_soft/model.pth HIGH_CKPT=output/checkpoints/high_level_paper/high_level_policy.zip \
    OUT=$OUT/system_load timeout 14400 bash scripts/tools/run_system_load_test.sh > "$OUT/system_load.log" 2>&1
  say "2 done (exit $?)"
fi

if want 3; then
  for map in A B; do
    for p in A B_s1 B_s2; do
      say "3 GT pose: policy $p on Map $map"
      timeout 2400 uv run --no-sync python scripts/tools/eval_nav_protocol.py --task "Spot-Nav-Map${map}-GT-Play-v0" \
        --checkpoint "${POL[$p]}" --num_envs 8 --run_seed 2 "${JITTER[@]}" > "$OUT/gt_${p}_map${map}.log" 2>&1
      say "  exit $?"
    done
  done
fi

if want 4; then
  ORB_LAUNCH=orbslam3_rgbd.launch.py
  for map in A B; do
    for p in A B_s1 B_s2; do
      for s in $(seq 1 "$ORB_SEEDS"); do
        say "4 ORB-SLAM3 RGB-D: policy $p on Map $map, run $s"
        orb_start "$OUT/orb_${p}_map${map}_run${s}.orbslam.log"
        timeout 1800 uv run --no-sync python scripts/tools/eval_nav_protocol.py --task "Spot-Nav-Map${map}-Play-v0" \
          --checkpoint "${POL[$p]}" --num_envs 1 --run_seed "$s" "${JITTER[@]}" > "$OUT/orb_${p}_map${map}_run${s}.log" 2>&1
        say "  exit $?"
        orb_stop
      done
    done
  done
fi

if want 5; then
  for map in A B; do
    for s in 1 2 3; do
      for sys in cuvslam orb; do
        d=$OUT/slam_compare/map${map}_${sys}_run${s}; mkdir -p "$d"
        if [ "$sys" = cuvslam ]; then
          task=Spot-Nav-Map${map}-cuVSLAM-Play-v0
        else
          task=Spot-Nav-Map${map}-Stereo-Play-v0; ORB_LAUNCH=orbslam3_stereo.launch.py; orb_start "$d/orbslam.log"
        fi
        say "5 $sys stereo, Map $map, drive $s"
        rm -rf output/cuvslam_frame_check
        timeout 1500 uv run --no-sync python scripts/slam/cuvslam_frame_check.py --task "$task" --to_goal > "$d/run.log" 2>&1
        say "  exit $?"
        [ "$sys" = orb ] && orb_stop
        cp -r output/cuvslam_frame_check/. "$d/" 2>/dev/null
        est=pose.tum; [ -f "$d/$est" ] && uvx --from evo evo_ape tum "$d/gt.tum" "$d/$est" -a > "$d/evo_ape.txt" 2>&1
      done
    done
  done
fi
say "all done"
