"""Replay review / 何切る tool.
Usage: python review.py <log.json[.gz]> <player_id> [--ckpt path] [--top 3] [--tonpuu]
For every decision of <player_id> prints the actual action and the model's top-k actions with probabilities."""
import sys, os, json, gzip, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from libriichi.state import PlayerState
from bot import load, DEFAULT_CKPT

TILES = ['%dm' % i for i in range(1, 10)] + ['%dp' % i for i in range(1, 10)] + ['%ds' % i for i in range(1, 10)] + list('ESWNPFC') + ['5mr', '5pr', '5sr']
NAMES = TILES + ['riichi', 'chi_low', 'chi_mid', 'chi_high', 'pon', 'kan', 'agari', 'ryukyoku', 'pass']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log'); ap.add_argument('player_id', type=int)
    ap.add_argument('--ckpt', default=DEFAULT_CKPT); ap.add_argument('--top', type=int, default=3)
    length = ap.add_mutually_exclusive_group(); length.add_argument('--hanchan', action='store_true'); length.add_argument('--tonpuu', action='store_true'); ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    dev = 'cpu' if a.cpu or not torch.cuda.is_available() else 'cuda'
    model = load(a.ckpt, dev)
    op = gzip.open if a.log.endswith('.gz') else open
    evs = [json.loads(l) for l in op(a.log, 'rt') if l.strip()]
    ps = PlayerState(a.player_id)
    agree = total = 0
    for i, e in enumerate(evs):
        if e['type'] == 'start_kyoku' and a.tonpuu: e['tonpuu'] = True
        if e['type'] == 'start_kyoku':
            print(f"\n=== {e['bakaze']}{e['kyoku']}-{e['honba']}  scores {e['scores']}")
        if e['type'] in ('start_game', 'end_game', 'end_kyoku'): 
            continue
        cans = ps.update(json.dumps(e))
        if not cans.can_act or i + 1 >= len(evs): continue
        obs, mask = ps.encode_obs(6, False)
        if mask.sum() <= 1: continue
        with torch.inference_mode():
            out = model(torch.as_tensor(obs[None], device=dev), torch.as_tensor(mask[None], device=dev))
        p = out['logits'][0].softmax(-1).cpu().numpy()
        nxt = evs[i + 1]
        actual = None
        if nxt.get('actor') == a.player_id:
            t = nxt['type']
            actual = {'dahai': lambda: nxt['pai'], 'reach': lambda: 'riichi', 'pon': lambda: 'pon', 'chi': lambda: 'chi',
                      'hora': lambda: 'agari', 'ryukyoku': lambda: 'ryukyoku', 'ankan': lambda: 'kan', 'kakan': lambda: 'kan', 'daiminkan': lambda: 'kan'}.get(t, lambda: t)()
        else:
            actual = 'pass' if mask[45] else '?'
        top = np.argsort(-p)[:a.top]
        best = NAMES[top[0]]
        ok = actual == best or (actual == 'chi' and best.startswith('chi'))
        agree += ok; total += 1
        print(f"turn {ps.at_turn:2d} | shanten {ps.shanten} | actual {actual:8s} | model " + '  '.join(f'{NAMES[j]} {p[j]*100:.1f}%' for j in top) + ('' if ok else '   <-- differs'))
    print(f'\nagreement {agree}/{total} = {agree/max(total,1):.3f}')

if __name__ == '__main__':
    main()
