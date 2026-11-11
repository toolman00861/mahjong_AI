#!/bin/bash
# Plan A re-run with the KL temperature fix, sized for a 1-hour budget (2026-11-11):
# temp 0.5, 500 seeds/iter, lr 1e-5, 10 iterations, no in-run selection evals (eval_seeds=2 baseline only),
# then a held-out eval (seeds 3,000,000+) of the final policy. Output: logs/rl_planA2.log, last line PLANA2_DONE.
set -uo pipefail
M=/root/gpufree-data/mj; source $M/env.sh; cd $M
L=logs/rl_planA2.log
R=ckpt/rl_hanchan_a2
SL=ckpt/sl_a/best.pth
log(){ echo "$(date -Is) $*" >> $L; }
log PLANA2_START
python - <<'PY'
import json
c=json.load(open('/root/gpufree-data/mj/ckpt/.stage_hanchan/rl_active.json'))
c.update(run='rl_hanchan_a2', temp=0.5, seeds_per_iter=500, lr=1e-5, max_iters=10, patience=99,
         eval_seeds=2, eval_every=1000)
json.dump(c,open('/root/gpufree-data/mj/ckpt/.stage_hanchan/rl_planA2.json','w'),indent=1)
PY
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 python -u code/mjtf/rl.py --cfg ckpt/.stage_hanchan/rl_planA2.json >> logs/rl_hanchan_a2.log 2>&1
log "RL_EXIT rc=$?"
python3 - >> $L <<'PY'
import json
for l in open('/root/gpufree-data/mj/ckpt/rl_hanchan_a2/train.jsonl'):
    t=json.loads(l); print('TRAIN', t['iter'], t['opp'], t['gen_mean_pt'], t['kl_sl'], t['ent'], t['clipfrac'], t['t_total'])
PY
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 python -u code/mjtf/eval.py --length 8 --challenger $R/last.pth --champion $SL \
  --seeds 2500 --seed_start 3000000 --workers 4 --bf16 --out logs/evals_heldout.jsonl \
  | grep EVAL_RESULT | cut -d' ' -f2- \
  | python -c "import sys,json; d=json.load(sys.stdin); print('HELDOUT planA2_last', json.dumps({k:d[k] for k in ('avg_rank','mean_pt','se_pt','rank_dist')}), json.dumps(d['style']))" >> $L
log PLANA2_DONE

