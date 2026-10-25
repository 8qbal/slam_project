#!/usr/bin/env bash
# Low-level terrain and noise study (docs/ROADMAP.md section 1, comparisons 2 and 3): every trained policy on every
# evaluation terrain with gait_check.py (tracked speed, turns, falls), then WallsSoft vs. WallsSoftNoise with and
# without observation noise. Log per run in $OUT; no training, no ROS 2. One Isaac Sim at a time (about 1-2 min a run).
#
# Usage (from the repo root):  bash scripts/tools/run_terrain_eval.sh
# Optional: OUT=output/terrain_eval
cd "$(dirname "$0")/../.."
set -u
OUT=${OUT:-output/terrain_eval}
mkdir -p "$OUT"
OBSTACLES=logs/rl_games/spot_paper_ppo/2026-10-10_07-00-46/nn/last_spot_paper_ppo_ep_8000_rew_161.45201.pth

# comparison 2: trained on (rows) x evaluated on (columns)
declare -A POLICY=(
  [paper]=policies/paper_isaaclab_reward_reproduce/model.pth
  [rough]=policies/rough/model.pth
  [obstacles]=$OBSTACLES
  [walls_soft]=policies/walls_soft/model.pth
)
for name in paper rough obstacles walls_soft; do
  for task in Spot-Paper-v0 Spot-Paper-Rough-v0 Spot-Paper-Obstacles-v0 Spot-Paper-WallsSoft-v0; do
    echo "[$(date +%T)] $name on $task"
    uv run --no-sync python scripts/tools/gait_check.py --task "$task" --checkpoint "${POLICY[$name]}" \
      > "$OUT/${name}__${task}.log" 2>&1
  done
done

# comparison 3: noise (the task has sensor noise; --noise keeps it on)
for name in walls_soft walls_soft_noise; do
  for flag in "" "--noise"; do
    echo "[$(date +%T)] $name on Spot-Paper-WallsSoftNoise-v0 $flag"
    uv run --no-sync python scripts/tools/gait_check.py --task Spot-Paper-WallsSoftNoise-v0 \
      --checkpoint "policies/$name/model.pth" $flag > "$OUT/${name}__noise${flag:+_on}.log" 2>&1
  done
done
echo "[$(date +%T)] done; logs in $OUT"
