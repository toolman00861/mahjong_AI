#!/bin/bash
# One full-size RL iteration (250 seeds, production model) in a throwaway run dir, to measure
# host RSS peak, GPU memory peak and per-iteration time before formal RL. Does not touch ckpt/rl_hanchan.
# Usage: bash code/probe_rl_mem.sh <sl_ckpt>
set -euo pipefail
M=/root/gpufree-data/mj; source $M/env.sh; cd $M
SL=${1:?sl ckpt}
rm -rf ckpt/rl_probe
python - "$SL" <<'PY'
import json,sys
c=json.load(open('code/mjtf/cfg/rl.json'))
c.update(run='rl_probe', sl_ckpt=sys.argv[1], max_iters=1, eval_seeds=2, eval_every=1000)
json.dump(c,open('/tmp/rl_probe.json','w'),indent=1)
PY
L=logs/probe_rl_mem.log
: > $L
RAYON_NUM_THREADS=8 OMP_NUM_THREADS=1 python -u code/mjtf/rl.py --cfg /tmp/rl_probe.json >> $L 2>&1 &
PID=$!
peak=0; gpk=0
while kill -0 $PID 2>/dev/null; do
  # RSS of the RL process tree (rl.py plus its eval subprocess), in MiB
  r=$(ps -o rss= --ppid $PID -p $PID 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}')
  g=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$PID -F', ' '$1==p {print $2}')
  [[ -n $r && $r -gt $peak ]] && peak=$r
  [[ -n $g && $g -gt $gpk ]] && gpk=$g
  sleep 2
done
wait $PID && rc=0 || rc=$?
echo "PROBE rc=$rc rss_peak_MiB=$peak gpu_peak_MiB=$gpk" | tee -a $L
tail -n 3 $L
rm -rf ckpt/rl_probe
