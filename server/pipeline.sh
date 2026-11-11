#!/bin/bash
# Hanchan pipeline. Completion of SL is a reporting/review gate by default.
set -euo pipefail
M=/root/gpufree-data/mj
source "$M/env.sh"
cd "$M"
mkdir -p ckpt/.stage_hanchan
L=logs/pipeline_hanchan.log
stage=ckpt/.stage_hanchan
step(){ echo "$(date -Is) $*" | tee -a "$L"; }
step 'Waiting for SL-A completion'
until python - <<'PY'
import torch,json
from pathlib import Path
m=Path('/root/gpufree-data/mj')
p=m/'ckpt/sl_a/last.pth'
if not p.exists(): raise SystemExit(1)
s=torch.load(p,map_location='cpu',weights_only=False)
cfg=json.load(open(m/'code/mjtf/cfg/sl_a.json'))
raise SystemExit(0 if s['step'] >= cfg['max_steps'] else 1)
PY
do
  if ! tmux has-session -t sl 2>/dev/null; then
    step 'SL-A not running and not complete; inspect logs before resuming'
    exit 1
  fi
  sleep 120
done
step 'SL-A complete: report validation results before proceeding'
if [[ ${CONTINUE_AFTER_SL:-0} != 1 ]]; then exit 0; fi
SL=ckpt/sl_a/best.pth
if [[ ${RUN_SL_B:-0} == 1 ]]; then
  # The staged hanchan indices must have been reviewed and published first.
  for year in 2022 2023 2024 2025; do
    test -e "data/index_hanchan/majsoul-jade-$year.tsv.done"
    # published by publish_hanchan.py (quality-filtered subset of the staged index)
    grep -q '"published"' "data/index/majsoul-jade-$year.tsv.done"
  done
  if [[ ! -e $stage/sl_b ]]; then
    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u code/mjtf/train_sl.py --cfg code/mjtf/cfg/sl_b.json >> logs/sl_b_hanchan.log 2>&1
    touch "$stage/sl_b"
  fi
  SL=ckpt/sl_b/best.pth
fi
if [[ ! -e $stage/eval_sl ]]; then
  RAYON_NUM_THREADS=8 OMP_NUM_THREADS=1 python -u code/mjtf/eval.py --length 8 --challenger "$SL" --champion rule --seeds 2500 --out logs/evals_hanchan.jsonl | tee -a "$L"
  RAYON_NUM_THREADS=8 OMP_NUM_THREADS=1 python -u code/mjtf/eval.py --length 8 --challenger "$SL" --champion ckpt/sl_a/backup_step.pth --seeds 1000 --out logs/evals_hanchan.jsonl | tee -a "$L"
  touch "$stage/eval_sl"
fi
step 'SL evaluation complete; review full reward metrics and RL smoke tests'
if [[ ${RUN_RL:-0} != 1 ]]; then exit 0; fi
# PLAN S9: GRP-based per-kyoku reward (return_mode=grp) needs a hanchan GRP first.
if [[ ! -e $stage/grp ]]; then
  (cd code/mjtf && python -u grpnet.py majsoul-throne-2023,majsoul-throne-2024,majsoul-jade-2019 --kind hanchan --output $M/ckpt/grp_hanchan.pth) >> logs/grp_hanchan.log 2>&1
  touch "$stage/grp"
  step "grp: $(tail -1 logs/grp_hanchan.log)"
fi
python - "$SL" <<'PY'
import json,sys
from pathlib import Path
m=Path('/root/gpufree-data/mj')
c=json.load(open(m/'code/mjtf/cfg/rl.json'))
c['sl_ckpt']=str(m/sys.argv[1]); c['kind']='hanchan'
json.dump(c,open(m/'ckpt/.stage_hanchan/rl_active.json','w'),indent=1)
PY
RAYON_NUM_THREADS=8 OMP_NUM_THREADS=1 python -u code/mjtf/rl.py --cfg ckpt/.stage_hanchan/rl_active.json >> logs/rl_hanchan.log 2>&1
step PIPELINE_DONE