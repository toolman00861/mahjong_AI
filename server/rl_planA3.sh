#!/bin/bash
# Continue fixed plan A (ckpt/rl_hanchan_a2, temp 0.5, 500 seeds/iter, lr 1e-5) for ~3 h (2026-11-11):
# 3 blocks of 10 iterations; after each block a held-out eval (seeds 3,000,000+) of the snapshot, paired vs SL.
# Output: logs/rl_planA3.log, last line PLANA3_DONE.
set -uo pipefail
M=/root/gpufree-data/mj; source $M/env.sh; cd $M
L=logs/rl_planA3.log
R=ckpt/rl_hanchan_a2
SL=ckpt/sl_a/best.pth
CFG=ckpt/.stage_hanchan/rl_planA2.json
log(){ echo "$(date -Is) $*" >> $L; }
log PLANA3_START
for target in 20 30 40; do
  python - $target <<'PY'
import json,sys
p='/root/gpufree-data/mj/ckpt/.stage_hanchan/rl_planA2.json'
c=json.load(open(p)); c['max_iters']=int(sys.argv[1]); json.dump(c,open(p,'w'),indent=1)
PY
  OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 python -u code/mjtf/rl.py --cfg $CFG >> logs/rl_hanchan_a2.log 2>&1
  rc=$?
  it=$(grep -o '"iter": [0-9]*' $R/train.jsonl | tail -1 | grep -o '[0-9]*$')
  log "BLOCK target=$target rc=$rc iter=$it $(tail -n 1 $R/train.jsonl | cut -c1-200)"
  [[ $rc != 0 || $it != $target ]] && { log ABORT; break; }
  cp $R/last.pth $R/snap_it$it.pth
  OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 python -u code/mjtf/eval.py --length 8 --challenger $R/snap_it$it.pth --champion $SL \
    --seeds 2500 --seed_start 3000000 --workers 4 --bf16 --out logs/evals_heldout.jsonl > /dev/null
  log "HELDOUT it$it $(python code/paired_heldout.py | grep snap_it$it)"
done
log PLANA3_DONE

