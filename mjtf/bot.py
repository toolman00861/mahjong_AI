"""mjai-protocol bot for the trained model.
Usage: python bot.py <player_id> [--ckpt path] [--tonpuu] [--cpu]
Reads mjai events (one JSON per line) from stdin, writes reactions to stdout.
For East-only games, start_kyoku events get the "tonpuu":true extension injected only with --tonpuu (default: hanchan)."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from libriichi.mjai import Bot
from model import TileTransformer
from engine import TFEngine

DEFAULT_CKPT = os.environ.get('MJTF_CKPT', '/root/gpufree-data/mj/ckpt/final/model.pth')

def load(ckpt, device):
    st = torch.load(ckpt, map_location='cpu', weights_only=False)
    m = TileTransformer(**st['cfg']['model']).to(device)
    m.load_state_dict(st['model']); m.eval()
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('player_id', type=int)
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    length = ap.add_mutually_exclusive_group(); length.add_argument('--hanchan', action='store_true'); length.add_argument('--tonpuu', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    dev = 'cpu' if a.cpu or not torch.cuda.is_available() else 'cuda'
    eng = TFEngine(load(a.ckpt, dev), device=dev, name='mjtf', enable_amp=(dev == 'cuda'))
    bot = Bot(eng, a.player_id)
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        if a.tonpuu and '"start_kyoku"' in line:
            e = json.loads(line); e['tonpuu'] = True; line = json.dumps(e)
        r = bot.react(line)
        if r:
            print(r, flush=True)

if __name__ == '__main__':
    main()
