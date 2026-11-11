"""RL rollout worker: self-play one seed range with bf16 copies of the policy/opponent, then build the
PPO batch for its own games (GRP rewards) and save it, so generation and batch building scale across
processes. Launched by rl.generate_and_build; prints one WORKER_RESULT json line."""
import os, sys, json, argparse, torch
sys.path.insert(0, '/root/gpufree-data/mj/code'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl
from grpnet import GRPNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True); ap.add_argument('--opponent', required=True)
    ap.add_argument('--gen_dir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--seed_start', type=int, required=True); ap.add_argument('--seeds', type=int, required=True)
    ap.add_argument('--temp', type=float, default=1.); ap.add_argument('--kind', default='hanchan')
    ap.add_argument('--return_mode', default='grp'); ap.add_argument('--grp_ckpt', default=None)
    ap.add_argument('--max_batch_bytes', type=int, default=24 * 1024**3)
    a = ap.parse_args()
    dev = torch.device('cuda'); torch.backends.cuda.matmul.allow_tf32 = True
    policy = rl.load_model(a.policy, dev)[0].to(torch.bfloat16)
    opponent = policy if a.opponent == a.policy else rl.load_model(a.opponent, dev)[0].to(torch.bfloat16)
    ranks = rl.generate(policy, opponent, a.gen_dir, a.seeds, a.seed_start, dev, a.temp, kind=a.kind)
    del policy, opponent
    grp = None
    if a.return_mode == 'grp':
        gs = torch.load(a.grp_ckpt, map_location='cpu', weights_only=False)
        grp = GRPNet().to(dev); grp.load_state_dict(gs['model']); grp.eval()
    batch, mean_pt, n_games = rl.build_batch(a.gen_dir, grp=grp, dev=dev, kind=a.kind,
                                             max_batch_bytes=a.max_batch_bytes, return_mode=a.return_mode,
                                             with_count=True)
    torch.save(batch, a.out + '.tmp'); os.replace(a.out + '.tmp', a.out)
    print('WORKER_RESULT', json.dumps(dict(ranks=list(map(int, ranks)), mean_pt=mean_pt, games=n_games)), flush=True)


if __name__ == '__main__':
    main()
