#!/usr/bin/env bash
# The 23-run check of a high-level checkpoint on the small maze and the warehouse (jittered starts), one Isaac Sim
# per run, about 40 min. One Isaac Sim at a time: do not run it while a training runs.
#
#   CKPT=<.pth> POSE=GT|cuVSLAM|cuVSLAM-Quality bash scripts/tools/run_eval_set.sh
#
# Small maze seeds 1, 2 x 5 random starts; warehouse seeds 0, 1, 2 x 3 random starts; 4 unjittered warehouse runs.
# Last line: successes out of runs. Per-run lines are in $OUT (default output/eval_set_<name>.log).
set -u
cd "$(dirname "$0")/../.."
export SPOT_VSLAM_LOW_LEVEL_CKPT=${SPOT_VSLAM_LOW_LEVEL_CKPT:-policies/walls_soft_mirror_turn/model.pth}
: "${CKPT:?set CKPT=<high-level .pth>}"
case "${POSE:-GT}" in
  GT) V=GT-Depth ;;
  cuVSLAM) V=cuVSLAM-Depth ;;
  cuVSLAM-Quality) V=cuVSLAM-QualityDepth ;;
  *) echo "POSE must be GT, cuVSLAM or cuVSLAM-Quality"; exit 1 ;;
esac
OUT=${OUT:-output/eval_set_$(basename "$(dirname "$(dirname "$CKPT")")")_${POSE:-GT}.log}
mkdir -p "$(dirname "$OUT")"; : > "$OUT"
ok=0; n=0
run() { # label task jitter-args
  local label=$1 task=$2; shift 2
  local line
  line=$(uv run --no-sync python scripts/tools/eval_nav_protocol.py --task "$task" --num_envs 1 --max_steps 900 \
    --checkpoint "$CKPT" "$@" 2>&1 | grep "PROTO RESULT" | tail -1)
  n=$((n + 1)); case "$line" in *"success 100%"*) ok=$((ok + 1)) ;; esac
  echo "$label: ${line:-NO RESULT}" | tee -a "$OUT"
}
for ms in 1 2; do for rs in 1 2 3 4 5; do
  SPOT_VSLAM_MAZE_SEED=$ms run "maze$ms r$rs" "Spot-Nav-SmallMaze-$V-Play-v0" --jitter_xy 0.5 --jitter_yaw 0.5 --run_seed $rs
done; done
for ws in 0 1 2; do for rs in 1 2 3; do
  SPOT_VSLAM_WAREHOUSE_SEED=$ws run "warehouse$ws r$rs" "Spot-Nav-Warehouse-$V-Play-v0" --jitter_xy 0.5 --jitter_yaw 0.5 --run_seed $rs
done; done
for c in "0 1" "1 1" "2 1" "0 0"; do set -- $c
  SPOT_VSLAM_WAREHOUSE_SEED=$1 SPOT_VSLAM_WAREHOUSE_LATERAL=$2 run "warehouse$1 lateral$2 fixed" "Spot-Nav-Warehouse-$V-Play-v0"
done
echo "EVAL SET $CKPT $V: $ok / $n" | tee -a "$OUT"
