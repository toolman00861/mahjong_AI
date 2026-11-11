"""Paired seed-cluster comparison of held-out evals (logs/evals_heldout.jsonl) against the SL self-play control."""
import json, numpy as np
rows = [json.loads(l) for l in open('/root/gpufree-data/mj/logs/evals_heldout.jsonl')]
ctrl = next(r for r in rows if r['challenger'].endswith('sl_a/best.pth'))
b = np.asarray(ctrl['seed_pt'])
for r in rows:
    if r is ctrl or r['seed_start'] != ctrl['seed_start'] or len(r['seed_pt']) != len(b): continue
    d = np.asarray(r['seed_pt']) - b
    print(f"{r['challenger'].split('ckpt/')[-1]:40s} mean_pt={r['mean_pt']:+.2f} paired_gain={d.mean():+.2f} se={d.std(ddof=1)/np.sqrt(len(d)):.2f}")

