"""Global Reward Predictor for hanchan: predicts final rank distribution from (grand_kyoku, honba, kyotaku, scores)
at the start of each kyoku, from the perspective of every player. Used to turn the final game reward into per-kyoku rewards."""
import os, sys, json, random, time, zlib
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, '/root/gpufree-data/mj/code')
from libriichi.dataset import Grp
M = '/root/gpufree-data/mj'

class GRPNet(nn.Module):
    def __init__(self, h=256, rounds=12):
        super().__init__()
        # input: kyoku one-hot(12: E1..W4) + honba/10 + kyotaku/4 + 4 relative scores/1e5 + 4 score-rank one-hot*... (simple)
        self.net = nn.Sequential(nn.Linear(rounds + 2 + 4 + 6, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 4))
    def forward(self, x):
        return self.net(x)

def featurize(feat, pid, rounds=12):
    """feat: (K,7) rows [grand_kyoku, honba, kyotaku, s0..s3 /1e4]; returns (K, rounds+12) from pid's perspective"""
    K = feat.shape[0]
    gk = feat[:, 0].astype(int)
    if np.any((gk < 0) | (gk >= rounds)):
        raise ValueError(f'Round outside GRP encoding range 0..{rounds - 1}')
    oh = np.zeros((K, rounds)); oh[np.arange(K), gk] = 1
    sc = feat[:, 3:7] * 1e4
    rel = np.roll(sc, -pid, axis=1)          # me first, then shimocha, toimen, kamicha
    # dealer seat relative to me: grand_kyoku % 4 is absolute oya seat
    oya_rel = (gk % 4 - pid) % 4
    oy = np.zeros((K, 4)); oy[np.arange(K), oya_rel] = 1
    diff = (rel[:, :1] - rel[:, 1:]) / 3e4     # 3 diffs to others
    x = np.concatenate([oh, feat[:, 1:2] / 5., feat[:, 2:3] / 4., rel / 5e4, oy[:, :1], diff, oy[:, 1:2], oy[:, 2:3]], axis=1)
    return x.astype(np.float32)

def load(files):
    X, Y = [], []
    for i in range(0, len(files), 256):
        try:
            gs = Grp.load_gz_log_files(files[i:i + 256])
        except Exception:
            continue
        for g in gs:
            feat = g.take_feature(); rk = g.take_rank_by_player()
            for pid in range(4):
                x = featurize(feat, pid); X.append(x); Y.append(np.full(len(x), rk[pid]))
    return np.concatenate(X), np.concatenate(Y)

if __name__ == '__main__':
    import glob as _glob
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('sources')
    ap.add_argument('directories', nargs='*')
    ap.add_argument('--kind', choices=('hanchan', 'tonpuu'), default='hanchan')
    ap.add_argument('--output')
    args = ap.parse_args()
    srcs = [s for s in args.sources.split(',') if s]
    files = []
    for d in args.directories:
        files += _glob.glob(f'{d}/*.json.gz')
    for s in srcs:
        bad = set(l.strip() for l in open(f'{M}/data/index/{s}.bad')) if os.path.exists(f'{M}/data/index/{s}.bad') else set()
        for l in open(f'{M}/data/index/{s}.tsv'):
            f, _, kind = l.split('\t')[:3]
            if kind == args.kind and f not in bad: files.append(f)
    random.Random(0).shuffle(files)
    val = [f for f in files if zlib.crc32(os.path.basename(f).encode()) % 100 == 0]
    tr = [f for f in files if zlib.crc32(os.path.basename(f).encode()) % 100 != 0]
    print(args.kind, 'games', len(tr), len(val), flush=True)
    if not tr or not val:
        raise ValueError('GRP requires nonempty train and validation splits')
    Xt, Yt = load(tr); Xv, Yv = load(val)
    print('samples', len(Xt), len(Xv), flush=True)
    dev = 'cuda'
    Xt, Yt, Xv, Yv = [torch.tensor(a).to(dev) for a in (Xt, Yt, Xv, Yv)]
    net = GRPNet().to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    best = 9
    for ep in range(30):
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 4096):
            idx = perm[i:i + 4096]
            loss = F.cross_entropy(net(Xt[idx]), Yt[idx].long())
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            vl = sum(F.cross_entropy(net(Xv[i:i+4096]), Yv[i:i+4096].long(), reduction='sum').item() for i in range(0, len(Xv), 4096)) / len(Xv)
            # first kyoku should be ~uniform: check
        print('epoch', ep, 'train', round(loss.item(), 4), 'val', round(vl, 4), flush=True)
        if vl < best:
            best = vl; torch.save({'model': net.state_dict(), 'kind': args.kind, 'rounds': 12}, args.output or f'{M}/ckpt/grp_{args.kind}.pth')
    print('GRP_DONE best', best)