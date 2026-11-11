#!/bin/bash
# Large confirmation (2026-11-11, fast version): 10,000 fresh seeds (20,000,000+, key 0x2333; never used),
# 40,000 games of RL it40 vs SL. SL-vs-SL control skipped for time (symmetric, measured -0.21 +-0.24 earlier),
# so the gain is the challenger's mean_pt with its seed-clustered SE.
set -uo pipefail
M=/root/gpufree-data/mj; source $M/env.sh; cd $M
L=logs/confirm_rl.log; O=logs/evals_confirm.jsonl; : > $O
SL=ckpt/sl_a/best.pth
RL=ckpt/rl_hanchan_a2/snap_it40.pth
echo "$(date -Is) CONFIRM_START" >> $L
OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8 python -u code/mjtf/eval.py --length 8 --challenger $RL --champion $SL \
  --seeds 10000 --seed_start 20000000 --workers 4 --bf16 --out $O > /dev/null 2>> $L
python - >> $L <<'PY'
import json
x = json.loads(open('/root/gpufree-data/mj/logs/evals_confirm.jsonl').readline())
print(f"RESULT games={x['games']} mean_pt={x['mean_pt']:+.3f} se={x['se_pt']:.3f} z={x['mean_pt']/x['se_pt']:.2f} "
      f"avg_rank={x['avg_rank']} rank_dist={x['rank_dist']} style={json.dumps(x['style'])} seconds={x['seconds']}")
PY
echo "$(date -Is) CONFIRM_DONE" >> $L

