#!/bin/bash
# S8 hanchan evaluation of the final SL-A checkpoint on held-out seeds.
# Seeds 3,000,000+ (key 0x2333) are disjoint from RL selection seeds (900,000+) and rollout seeds (10M+, key 0x5eed).
# Usage: bash code/run_s8.sh [checkpoint]   (default ckpt/sl_a/best.pth, snapshotted first)
set -euo pipefail
M=/root/gpufree-data/mj; source $M/env.sh; cd $M
SRC=${1:-ckpt/sl_a/best.pth}
SNAP=ckpt/sl_a/s8_candidate.pth
cp "$SRC" "$SNAP"
OUT=logs/evals_hanchan.jsonl
L=logs/s8_hanchan.log
echo "$(date -Is) S8 start candidate=$SRC sha=$(sha256sum $SNAP | cut -c1-16)" | tee -a $L
for champ in rule ckpt/sl_a/backup_step.pth ckpt/sl_a/pre_auxw.pth; do
  RAYON_NUM_THREADS=8 OMP_NUM_THREADS=1 python -u code/mjtf/eval.py --length 8 \
    --challenger "$SNAP" --champion "$champ" --seeds 2500 --seed_start 3000000 --out $OUT \
    | grep EVAL_RESULT | tee -a $L
done
touch ckpt/.stage_hanchan/eval_sl
echo "$(date -Is) S8_DONE" | tee -a $L
