"""Generate East-only self-play games with a checkpoint (all 4 seats driven by the same model, sampling) for GRP training."""
import os, sys, argparse
sys.path.insert(0, '/root/gpufree-data/mj/code'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libriichi.arena import OneVsThree
from eval import load_engine
ap = argparse.ArgumentParser(); ap.add_argument('--ckpt'); ap.add_argument('--out'); ap.add_argument('--seeds', type=int, default=5000); ap.add_argument('--seed_start', type=int, default=50_000_000)
ap.add_argument('--length', type=int, choices=(4, 8), default=8)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
ch = load_engine(a.ckpt, 'challenger', sample=True, temp=1.0)
cp = load_engine(a.ckpt, 'champion', sample=True, temp=1.0)
env = OneVsThree(disable_progress_bar=True, log_dir=a.out, game_length=a.length)
print('GEN_DONE', env.py_vs_py(challenger=ch, champion=cp, seed_start=(a.seed_start, 0x6e6), seed_count=a.seeds), flush=True)
