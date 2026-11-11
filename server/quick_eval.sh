#!/bin/bash
# Quick sanity eval: snapshot best.pth, play N seeds vs a champion, print the result without per-seed list.
# Usage: bash code/quick_eval.sh <champion> <seeds> [seed_start]
set -euo pipefail
M=/root/gpufree-data/mj; source $M/env.sh; cd $M
cp ckpt/sl_a/best.pth /tmp/best_snap.pth
RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1 python -u code/mjtf/eval.py --length 8 \
  --challenger /tmp/best_snap.pth --champion "$1" --seeds "$2" --seed_start "${3:-900000}" \
  | grep EVAL_RESULT | cut -d' ' -f2- \
  | python -c 'import sys,json; d=json.load(sys.stdin); d.pop("seed_pt"); print(json.dumps(d))'
