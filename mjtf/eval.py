"""Duplicate one-vs-three evaluation in hanchan games by default.
Each seed is played 4 times with the challenger rotated through all seats (libriichi OneVsThree)."""
import os, sys, json, argparse, gzip, time, math, tempfile
from pathlib import Path
from collections import defaultdict
import numpy as np, torch
sys.path.insert(0, '/root/gpufree-data/mj/code'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libriichi.arena import OneVsThree
from model import TileTransformer
from engine import TFEngine, RuleEngine
from data import UMA, PLACE_BONUS

def load_engine(path, name, device='cuda', bf16=False, **kw):
    if path == 'rule':
        return RuleEngine(name)
    st = torch.load(path, map_location='cpu', weights_only=False)
    m = TileTransformer(**st['cfg']['model']).to(device)
    m.load_state_dict(st['model'])
    if bf16: m = m.to(torch.bfloat16)
    return TFEngine(m, device=device, name=name, **kw)


def play(a, log_dir, seed_start, seeds):
    ch = load_engine(a.challenger, 'challenger', device=a.device, bf16=a.bf16)
    cp = load_engine(a.champion, 'champion', device=a.device, bf16=a.bf16)
    env = OneVsThree(disable_progress_bar=True, log_dir=log_dir, game_length=a.length)
    return list(map(int, env.py_vs_py(challenger=ch, champion=cp, seed_start=(seed_start, a.seed_key), seed_count=seeds)))


def play_parallel(a, log_dir):
    """Split the seed range over a.workers processes sharing one log dir; returns summed arena ranks."""
    import subprocess
    w = max(1, min(a.workers, a.seeds)); base, extra = divmod(a.seeds, w)
    procs, start = [], a.seed_start
    for k in range(w):
        n = base + (k < extra)
        cmd = [sys.executable, '-u', os.path.abspath(__file__), '--challenger', a.challenger, '--champion', a.champion,
               '--seeds', str(n), '--seed_start', str(start), '--seed_key', str(a.seed_key), '--length', str(a.length),
               '--device', a.device, '--play_only', log_dir] + (['--bf16'] if a.bf16 else [])
        env = dict(os.environ, OMP_NUM_THREADS='1', RAYON_NUM_THREADS='1')
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env))
        start += n
    total, errs = [0] * 4, []
    for p in procs:
        out, err = p.communicate()
        r = next((json.loads(l[len('PLAY_RANKS '):]) for l in out.splitlines() if l.startswith('PLAY_RANKS ')), None)
        if p.returncode != 0 or r is None: errs.append(err[-3000:])
        else: total = [x + y for x, y in zip(total, r)]
    if errs: raise RuntimeError('eval worker failure:\n' + '\n'.join(errs))
    return total

def _read_game(path, challenger):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    if not events or events[0].get('type') != 'start_game':
        raise ValueError(f'{path}: missing leading start_game')
    names, seed = events[0].get('names'), events[0].get('seed')
    if not isinstance(names, list) or len(names) != 4 or names.count(challenger) != 1:
        raise ValueError(f'{path}: malformed names or challenger seat')
    if not isinstance(seed, (list, tuple)) or len(seed) != 2:
        raise ValueError(f'{path}: missing (seed, key) in start_game')
    seed, key = map(int, seed)
    me = names.index(challenger)
    score, last_kyotaku, kyoku_count = None, None, 0
    in_kyoku, has_result = False, False
    # Per-kyoku play-style flags for the challenger seat (each counted at most once per kyoku).
    style = dict(kyoku=0, agari=0, tsumo=0, houjuu=0, riichi=0, fuuro=0)
    flags = set()
    def close_flags():
        for k in flags: style[k] += 1
        flags.clear()
    for e in events[1:]:
        typ = e.get('type')
        if typ in ('chi', 'pon', 'daiminkan') and int(e.get('actor', -1)) == me:
            flags.add('fuuro')
        elif typ == 'hora':
            actor, target = int(e.get('actor', -1)), int(e.get('target', -1))
            if actor == me:
                flags.add('agari')
                if target == me: flags.add('tsumo')
            elif target == me:
                flags.add('houjuu')
        if typ == 'start_kyoku':
            if in_kyoku: raise ValueError(f'{path}: previous kyoku not closed')
            in_kyoku, has_result = True, False
            style['kyoku'] += 1
            if not isinstance(e.get('scores'), list) or len(e['scores']) != 4:
                raise ValueError(f'{path}: malformed start_kyoku scores')
            score = list(map(int, e['scores']))
            last_kyotaku = int(e.get('kyotaku', 0))
            kyoku_count += 1
        elif typ == 'reach_accepted':
            if not in_kyoku or has_result: raise ValueError(f'{path}: reach outside active kyoku')
            actor = int(e['actor'])
            if actor not in range(4): raise ValueError(f'{path}: invalid riichi actor')
            score[actor] -= 1000
            if actor == me: flags.add('riichi')
        elif typ in ('hora', 'ryukyoku'):
            if not in_kyoku or not isinstance(e.get('deltas'), list) or len(e['deltas']) != 4:
                raise ValueError(f'{path}: invalid result event/deltas')
            score = [x + int(y) for x, y in zip(score, e['deltas'])]
            has_result = True
        elif typ == 'end_kyoku':
            if not in_kyoku or not has_result:
                raise ValueError(f'{path}: kyoku closed without result')
            in_kyoku = False
            close_flags()
    if in_kyoku: raise ValueError(f'{path}: unfinished terminal kyoku')
    if score is None or last_kyotaku is None: raise ValueError(f'{path}: no completed kyoku')
    types = [e.get('type') for e in events]
    if types[-1:] != ['end_game'] or types.count('end_game') != 1:
        raise ValueError(f'{path}: missing unique terminal end_game')
    if len(types) < 3 or types[-2] != 'end_kyoku' or not any(t in ('hora', 'ryukyoku') for t in types):
        raise ValueError(f'{path}: incomplete terminal kyoku/result sequence')
    # Rust arena awards unclaimed kyotaku to the highest score (first seat on ties).
    total = sum(score)
    if total > 100000 or (100000-total) % 1000: raise ValueError(f'{path}: invalid score total {total}')
    leftover = (100000-total)//1000
    if leftover: score[max(range(4), key=lambda i: score[i])] += leftover*1000
    order = sorted(range(4), key=lambda i: -score[i])  # stable: seat order breaks ties
    ranks = [order.index(i) for i in range(4)]
    return seed, key, me, score, ranks[me], kyoku_count, style


def summarize_logs(log_dir, challenger, seed_start, seed_key, seed_count, length, arena_ranks):
    files = sorted(Path(log_dir).glob('*.json.gz'))
    if len(files) != seed_count*4:
        raise ValueError(f'expected {seed_count*4} game logs, found {len(files)}')
    games, grouped = [], defaultdict(list)
    for path in files:
        g = _read_game(path, challenger)
        if not seed_start <= g[0] < seed_start+seed_count or g[1] != seed_key:
            raise ValueError(f'{path}: log seed/key outside requested range')
        games.append(g); grouped[(g[0], g[1])].append(g)
    expected = {(s, seed_key) for s in range(seed_start, seed_start+seed_count)}
    if set(grouped) != expected or any(len(v) != 4 for v in grouped.values()):
        raise ValueError('logs do not contain exactly four games for every requested seed')
    for pair, group in grouped.items():
        if sorted(g[2] for g in group) != [0, 1, 2, 3]:
            raise ValueError(f'seed {pair}: challenger did not occupy each seat exactly once')
    log_ranks = [sum(g[4] == r for g in games) for r in range(4)]
    if list(map(int, arena_ranks)) != log_ranks:
        raise ValueError(f'arena/log rank mismatch: arena={list(arena_ranks)}, logs={log_ranks}')
    kind = 'tonpuu' if length == 4 else 'hanchan'
    seed_pt, cluster_rank, all_pt = [], [], []
    hist = [0]*4
    for (seed, key), group in sorted(grouped.items()):
        p = [float(UMA[kind][g[4]] + PLACE_BONUS[g[4]] + (g[3][g[2]]-25000)/1000.) for g in group]
        seed_pt.append(sum(p)/4)
        cluster_rank.append(sum(g[4]+1 for g in group)/4)
        all_pt.extend(p)
        for g in group: hist[g[4]] += 1
    rank_dist = np.asarray(hist, dtype=float)/len(games)
    tot = defaultdict(int)
    for g in games:
        for k, v in g[6].items(): tot[k] += v
    nk = max(tot['kyoku'], 1)
    # Rates are per kyoku of the challenger seat; agari includes tsumo, houjuu excludes own tsumo.
    style = dict(kyoku=tot['kyoku'], **{f'{k}_rate': round(tot[k]/nk, 4)
                                        for k in ('agari', 'tsumo', 'houjuu', 'riichi', 'fuuro')})
    return dict(kind=kind, game_length=length, games=len(games), ranks=hist, style=style,
                rank_dist=rank_dist.round(4).tolist(),
                avg_rank=round(float(np.mean([g[4]+1 for g in games])), 4),
                se_rank=(float(np.std(cluster_rank, ddof=1)/math.sqrt(seed_count)) if seed_count > 1 else None),
                mean_pt=float(np.mean(all_pt)),
                se_pt=(float(np.std(seed_pt, ddof=1)/math.sqrt(seed_count)) if seed_count > 1 else None),
                pt_placement=round(float((rank_dist*(UMA[kind]+PLACE_BONUS)).sum()), 2),
                seed_start=seed_start, seed_key=seed_key, seed_count=seed_count,
                seed_pt=seed_pt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--challenger', required=True)
    ap.add_argument('--champion', required=True)
    ap.add_argument('--seeds', type=int, default=2500)   # games = 4 * seeds
    ap.add_argument('--seed_key', type=int, default=0x2333)
    ap.add_argument('--seed_start', type=int, default=100000)
    ap.add_argument('--length', type=int, choices=(4, 8), default=8)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    ap.add_argument('--out', default=None)
    ap.add_argument('--workers', type=int, default=1, help='parallel arena processes sharing the GPU')
    ap.add_argument('--bf16', action='store_true', help='bf16 weights (fused inference fast path)')
    ap.add_argument('--play_only', default=None, help=argparse.SUPPRESS)  # internal: worker log dir
    a = ap.parse_args()
    if a.seeds < 1: ap.error('--seeds must be positive')
    if a.play_only:
        print('PLAY_RANKS', json.dumps(play(a, a.play_only, a.seed_start, a.seeds)), flush=True)
        return
    t = time.time()
    with tempfile.TemporaryDirectory(prefix='mjtf-eval-') as log_dir:
        ranks = play_parallel(a, log_dir) if a.workers > 1 else play(a, log_dir, a.seed_start, a.seeds)
        out = summarize_logs(log_dir, 'challenger', a.seed_start, a.seed_key, a.seeds, a.length, ranks)
    out.update(challenger=a.challenger, champion=a.champion, seconds=round(time.time() - t),
               workers=a.workers, bf16=a.bf16)
    print('EVAL_RESULT', json.dumps(out), flush=True)
    if a.out:
        with open(a.out, 'a') as f: f.write(json.dumps(out) + '\n')

if __name__ == '__main__':
    main()
