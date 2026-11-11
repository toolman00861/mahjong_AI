"""PPO self-play RL in the hanchan libriichi arena with KL regularisation to the SL policy,
fixed rollout advantages and paired-seed evaluation.

return_mode:
  grp      (default, PLAN S9 "Global Reward Prediction"): Mortal-style per-kyoku reward
           r_k = phi(s_{k+1}) - phi(s_k), phi = GRP-expected placement pt + raw score/1000.
           Return-to-go from kyoku k equals r_k plus (final - phi(s_{k+1})), whose expectation given
           s_{k+1} is zero when GRP is calibrated, so r_k is a lower-variance estimate of the same target.
  terminal full final Mahjong Soul pt at every decision (unbiased, higher variance)."""
import os, sys, json, time, glob, shutil, random, argparse, math, subprocess, tempfile
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, '/root/gpufree-data/mj/code'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libriichi.arena import OneVsThree
from libriichi.dataset import GameplayLoader
from model import TileTransformer
from engine import TFEngine
from data import UMA, PLACE_BONUS
from grpnet import GRPNet, featurize
M = '/root/gpufree-data/mj'

def load_model(path, dev):
    st = torch.load(path, map_location='cpu', weights_only=False)
    m = TileTransformer(**st['cfg']['model']).to(dev); m.load_state_dict(st['model']); m.eval()
    return m, st

def generate(policy, opponent, gen_dir, seeds, seed_start, dev, temp, kind="hanchan"):
    # Each iteration owns a fresh directory; never delete an arbitrary supplied path.
    if os.path.exists(gen_dir) and os.listdir(gen_dir):
        raise ValueError('Generation directory must be empty')
    os.makedirs(gen_dir, exist_ok=True)
    if kind not in UMA or seeds < 1 or temp <= 0:
        raise ValueError('Invalid rollout kind, seed count or temperature')
    ch = TFEngine(policy, device=dev, name='challenger', sample=True, temp=temp)
    cp = TFEngine(opponent, device=dev, name='champion', sample=False)
    env = OneVsThree(disable_progress_bar=True, log_dir=gen_dir, game_length=4 if kind == "tonpuu" else 8)
    ranks = env.py_vs_py(challenger=ch, champion=cp, seed_start=(seed_start, 0x5eed), seed_count=seeds)
    if sum(ranks) != seeds * 4:
        raise RuntimeError('Incomplete duplicate rollout')
    return ranks


def generate_and_build(policy_path, opp_path, gen_dir, seeds, seed_start, cfg, kind, workers):
    """Parallel rollout: `workers` processes each play a contiguous seed slice and build their own
    batch. Seeds/ordering are identical to a single generate() call over the same range; the batch is
    the concatenation in seed-slice order. Returns (batch, ranks, mean_pt)."""
    if os.path.exists(gen_dir) and os.listdir(gen_dir):
        raise ValueError('Generation directory must be empty')
    workers = max(1, min(int(workers), seeds))
    base, extra = divmod(seeds, workers)
    procs, start = [], seed_start
    for w in range(workers):
        n = base + (w < extra)
        cmd = [sys.executable, '-u', f'{os.path.dirname(os.path.abspath(__file__))}/gen_worker.py',
               '--policy', policy_path, '--opponent', opp_path, '--gen_dir', f'{gen_dir}/w{w}',
               '--out', f'{gen_dir}/w{w}.pt', '--seed_start', str(start), '--seeds', str(n),
               '--temp', str(cfg.get('temp', 1.)), '--kind', kind, '--return_mode', cfg.get('return_mode', 'grp'),
               '--max_batch_bytes', str(cfg.get('max_batch_bytes', 24 * 1024**3) // workers)]
        if cfg.get('grp_ckpt'): cmd += ['--grp_ckpt', cfg['grp_ckpt']]
        env = dict(os.environ, OMP_NUM_THREADS='1', RAYON_NUM_THREADS='1')
        procs.append((w, n, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)))
        start += n
    results, errors = [], []
    for w, n, p in procs:
        out, err = p.communicate()
        res = next((json.loads(l[len('WORKER_RESULT '):]) for l in out.splitlines() if l.startswith('WORKER_RESULT ')), None)
        if p.returncode != 0 or res is None or res['games'] != 4 * n:
            errors.append(f'worker {w} rc={p.returncode}: {out[-1000:]} {err[-3000:]}')
        results.append(res)
    if errors:
        raise RuntimeError('Rollout worker failure:\n' + '\n'.join(errors))
    parts = [torch.load(f'{gen_dir}/w{w}.pt', weights_only=False) for w in range(workers)]
    batch = tuple(torch.cat([p[i] for p in parts]) for i in range(4))
    del parts
    ranks = [sum(r['ranks'][i] for r in results) for i in range(4)]
    games = sum(r['games'] for r in results)
    if sum(ranks) != seeds * 4 or games != seeds * 4:
        raise RuntimeError('Incomplete duplicate rollout')
    mean_pt = sum(r['mean_pt'] * r['games'] for r in results) / games
    return batch, ranks, mean_pt


def build_batch(gen_dir, grp=None, dev='cpu', kind="hanchan", max_batch_bytes=24 * 1024**3, return_mode='grp',
                with_count=False):
    """Per-decision return / 100.

    return_mode='grp': reward of the kyoku the decision belongs to, r_k = phi(s_{k+1}) - phi(s_k)
    (Mortal-style GRP shaping; the true return-to-go final - phi(s_k) = r_k + (final - phi(s_{k+1})),
    and the second term has zero mean given s_{k+1} for a calibrated GRP, so it is dropped as noise).
    Note the value head then regresses kyoku-level reward rather than the SL full-final-point target,
    so it re-adapts during the first RL iterations.
    return_mode='terminal': full final Mahjong Soul pt at every decision (unbiased, higher variance).
    """
    pt_table = np.asarray(UMA[kind] + PLACE_BONUS, dtype=np.float64)
    files = sorted(glob.glob(f'{gen_dir}/*.json.gz'))
    if not files:
        raise ValueError('No rollout logs')
    if return_mode == 'grp' and grp is None:
        raise ValueError('return_mode=grp needs a GRP model')
    pt_t = torch.tensor(pt_table, dtype=torch.float32)
    loader = GameplayLoader(6, oracle=False, player_names=['challenger'])
    obs_l, mask_l, act_l, ret_l = [], [], [], []
    final_pts, nbytes = [], 0
    for i in range(0, len(files), 8):
        for fg in loader.load_gz_log_files(files[i:i + 8]):
            for g in fg:
                obs = g.take_obs(); acts = g.take_actions(); masks = g.take_masks(); atk = g.take_at_kyoku()
                gr = g.take_grp(); pid = g.take_player_id()
                feat = gr.take_feature()
                final = np.asarray(gr.take_final_scores(), dtype=np.float64)
                rk = int(gr.take_rank_by_player()[pid])
                pt = float(pt_table[rk] + (final[pid] - 25000) / 1000.)
                if not np.isfinite(pt):
                    raise ValueError('Nonfinite terminal reward')
                final_pts.append(pt)
                if return_mode == 'grp':
                    with torch.no_grad():
                        p = grp(torch.from_numpy(featurize(feat, pid)).to(dev)).float().softmax(-1).cpu()
                    place_seq = np.append((p * pt_t).sum(-1).numpy(), pt_table[rk])
                    sc = np.concatenate([feat[:, 3 + pid] * 1e4, [final[pid]]])
                    kyoku_r = (place_seq[1:] - place_seq[:-1]) + (sc[1:] - sc[:-1]) / 1000.
                    # telescoping check: sum of kyoku rewards == final pt - phi(start)
                    if abs(kyoku_r.sum() - (pt - place_seq[0] - (sc[0] - 25000) / 1000.)) > 1e-3:
                        raise ValueError('GRP reward does not telescope')
                masks = np.stack(masks); multi = masks.sum(-1) > 1
                for j in np.flatnonzero(multi):
                    if not masks[j, acts[j]]:
                        raise ValueError('Illegal rollout action')
                    nbytes += obs[j].size * 2 + masks[j].nbytes + 12
                    if nbytes > max_batch_bytes:
                        raise MemoryError('Rollout batch budget exceeded; reduce seeds_per_iter')
                    obs_l.append(obs[j].astype(np.float16)); mask_l.append(masks[j]); act_l.append(acts[j])
                    if return_mode == 'grp':
                        k = int(atk[j])
                        if not 0 <= k < len(kyoku_r): raise ValueError('Decision kyoku index out of range')
                        ret_l.append(kyoku_r[k] / 100.)
                    else:
                        ret_l.append(pt / 100.)
    if not act_l or len(final_pts) != len(files):
        raise ValueError('Empty decisions or incomplete challenger rollout loading')
    batch = (torch.from_numpy(np.stack(obs_l)), torch.from_numpy(np.stack(mask_l)),
             torch.tensor(act_l, dtype=torch.long), torch.tensor(ret_l, dtype=torch.float32))
    if with_count:
        return batch, float(np.mean(final_pts)), len(final_pts)
    return batch, float(np.mean(final_pts))


def ppo_update(policy, sl_model, opt, batch, cfg, dev, fwd=None, sl_fwd=None):
    """One rollout update; old probabilities and advantages frozen across epochs.
    fwd / sl_fwd: optional torch.compile'd (static-shape) wrappers sharing the same weights; they are
    used only for full-size minibatches, the variable-length tail goes through the eager modules so the
    compiled graph is never rebuilt."""
    obs, mask, act, ret = batch
    n = len(act)
    mb = int(cfg['mb']); epochs = int(cfg.get('ppo_epochs', 2)); temp = cfg.get('temp', 1.)
    pick = lambda eager, comp, size: comp if (comp is not None and size == mb) else eager
    if n < 2 or mb < 1 or epochs < 1 or temp <= 0:
        raise ValueError('Invalid PPO batch/configuration')
    device = torch.device(dev)
    use_amp = device.type == 'cuda'
    # Keep the behaviour network mode fixed (also correct if dropout is nonzero).
    policy.eval(); sl_model.eval()
    old_lp, old_v = torch.empty(n), torch.empty(n)
    # The SL reference is frozen, so its log-probs are computed once here instead of every epoch.
    sl_lp_all = torch.empty(n, mask.shape[1])
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
        for i in range(0, n, mb):
            o, m = obs[i:i+mb].to(device), mask[i:i+mb].to(device)
            out = pick(policy, fwd, len(o))(o, m)
            lg = out['logits'].float() / temp
            old_lp[i:i+mb] = lg.log_softmax(-1).gather(-1, act[i:i+mb, None].to(device)).squeeze(-1).cpu()
            old_v[i:i+mb] = out['value'].float().cpu()
            # Same temperature as the policy: the KL term compares behaviour distributions, otherwise
            # temp != 1 makes KL > 0 at the SL init and the penalty pushes the policy's logit scale down.
            sl_lp_all[i:i+mb] = (pick(sl_model, sl_fwd, len(o))(o, m)['logits'].float() / temp).log_softmax(-1).cpu()
    adv = ret - old_v
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)
    if not all(torch.isfinite(x).all() for x in (ret, old_lp, adv)):
        raise FloatingPointError('Nonfinite PPO rollout targets')
    stats = dict(pg=0., v=0., kl_sl=0., ent=0., clipfrac=0.)
    seen = updates = 0
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, mb):
            idx = perm[i:i+mb]
            o = obs[idx].to(device); m = mask[idx].to(device); a = act[idx].to(device)
            R = ret[idx].to(device); A = adv[idx].to(device); olp = old_lp[idx].to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = pick(policy, fwd, len(idx))(o, m)
            logp = (out['logits'].float() / temp).log_softmax(-1)
            lp = logp.gather(-1, a[:, None]).squeeze(-1)
            ratio = (lp - olp).exp()
            pg = -torch.minimum(ratio * A, ratio.clamp(1-cfg['clip'], 1+cfg['clip']) * A).mean()
            v_loss = F.mse_loss(out['value'].float(), R)
            prob = logp.exp(); sl_lp = sl_lp_all[idx].to(device)
            kl = (prob * (logp - sl_lp)).masked_fill(~m, 0).sum(-1).mean()
            ent = -(prob * logp).masked_fill(~m, 0).sum(-1).mean()
            loss = pg + cfg['c_v'] * v_loss + cfg['beta_kl'] * kl - cfg['c_ent'] * ent
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite PPO loss')
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            values = dict(pg=pg.item(), v=v_loss.item(), kl_sl=kl.item(), ent=ent.item(),
                          clipfrac=((ratio-1).abs()>cfg['clip']).float().mean().item())
            for k, v in values.items(): stats[k] += v * len(idx)
            seen += len(idx); updates += 1
    return {**{k: round(v / seen, 6) for k, v in stats.items()}, 'updates': updates, 'samples_seen': seen}


def compare_evals(candidate, incumbent):
    """Paired seed-cluster difference; a two-SE heuristic, not a sequential test."""
    for key in ('kind', 'champion', 'seed_start', 'seed_key', 'seed_count'):
        if candidate[key] != incumbent[key]:
            raise ValueError(f'Incompatible evaluation protocol: {key}')
    if candidate.get('bf16', False) != incumbent.get('bf16', False):
        raise ValueError('Incompatible evaluation protocol: bf16')
    a = np.asarray(candidate['seed_pt'], dtype=np.float64)
    b = np.asarray(incumbent['seed_pt'], dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2 or len(a) != candidate['seed_count']:
        raise ValueError('Paired evaluation needs >=2 matching seed clusters')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite evaluation points')
    diff = a - b
    gain = float(diff.mean()); se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    return dict(gain_pt=gain, se_gain_pt=se, clear_improvement=bool(gain > 2 * se))


def run_eval(ckpt, champion, seeds, out, seed_start=900000, kind="hanchan", workers=1, bf16=False):
    r = subprocess.run([sys.executable, f'{M}/code/mjtf/eval.py', '--challenger', ckpt, '--champion', champion,
                        '--length', '4' if kind == 'tonpuu' else '8', '--seeds', str(seeds), '--seed_start', str(seed_start), '--out', out,
                        '--workers', str(workers)] + (['--bf16'] if bf16 else []), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-3000:])
    for l in r.stdout.splitlines():
        if l.startswith('EVAL_RESULT'):
            return json.loads(l[len('EVAL_RESULT '):])
    raise RuntimeError(r.stdout[-2000:] + r.stderr[-3000:])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cfg', required=True); a = ap.parse_args()
    cfg = json.load(open(a.cfg)); run = cfg['run']; rdir = f'{M}/ckpt/{run}'; os.makedirs(f'{rdir}/pool', exist_ok=True)
    kind = cfg.get('kind', 'hanchan')
    if kind not in UMA: raise ValueError(f'Unknown game kind: {kind}')
    if cfg['eval_seeds'] < 2: raise ValueError('RL evaluation requires at least two seeds')
    dev = torch.device('cuda'); torch.backends.cuda.matmul.allow_tf32 = True
    sl_model, sl_st = load_model(cfg['sl_ckpt'], dev)
    for p in sl_model.parameters(): p.requires_grad_(False)
    state_path = f'{rdir}/last.pth'
    policy, st = load_model(state_path if os.path.exists(state_path) else cfg['sl_ckpt'], dev)
    it = st.get('rl_iter', 0) if os.path.exists(state_path) else 0
    return_mode = cfg.get('return_mode', 'grp')
    if return_mode not in ('grp', 'terminal'):
        raise ValueError(f'Unknown return_mode {return_mode}')
    if st.get('rl_iter', 0) and st.get('kind') != kind:
        raise ValueError('RL checkpoint game kind mismatch')
    if st.get('rl_iter', 0) and st.get('return_mode') != return_mode:
        raise ValueError('RL checkpoint return_mode mismatch; use a new run directory')
    grp = None
    if return_mode == 'grp':
        gs = torch.load(cfg['grp_ckpt'], map_location='cpu', weights_only=False)
        if gs.get('kind') != kind or gs.get('rounds', 12) != 12:
            raise ValueError('GRP checkpoint kind/rounds mismatch')
        grp = GRPNet().to(dev); grp.load_state_dict(gs['model']); grp.eval()
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg['lr'], weight_decay=0.0, betas=(0.9, 0.99))
    # Static-shape compiled forwards (about 19% faster steady-state update on L40; ~65 s one-time compile).
    fwd = sl_fwd = None
    if cfg.get('compile', False):
        fwd = torch.compile(policy, dynamic=False); sl_fwd = torch.compile(sl_model, dynamic=False)
    if os.path.exists(state_path) and 'opt' in st:
        opt.load_state_dict(st['opt'])
        # load_state_dict restores the saved lr; the config value wins so a resumed run can change it.
        for g in opt.param_groups: g['lr'] = cfg['lr']
    hist = json.load(open(f'{rdir}/hist.json')) if os.path.exists(f'{rdir}/hist.json') else {'best_pt': None, 'no_improve': 0, 'evals': []}
    if hist['evals'] and 'best_eval' not in hist:
        raise ValueError('Legacy placement-only history; use a new run directory')
    base_cfg = sl_st['cfg']
    # Baseline and candidate evals in one run must use the same settings (paired comparison).
    eval_kw = dict(workers=int(cfg.get('eval_workers', 1)), bf16=bool(cfg.get('eval_bf16', False)))
    def save(path, extra={}):
        torch.save({'model': policy.state_dict(), 'cfg': base_cfg, 'rl_iter': it, 'return_mode': return_mode, 'kind': kind, **extra}, path + '.tmp'); os.replace(path + '.tmp', path)
    if not os.path.exists(f'{rdir}/pool/it0.pth'): save(f'{rdir}/pool/it0.pth')
    if not hist.get('best_eval'):
        if it != 0:
            raise ValueError('Missing evaluation history for resumed RL checkpoint')
        baseline = run_eval(cfg['sl_ckpt'], cfg['sl_ckpt'], cfg['eval_seeds'], f'{rdir}/eval.jsonl', kind=kind, **eval_kw)
        baseline['iter'] = 0
        hist.update(best_eval=baseline, best_pt=baseline['mean_pt'], no_improve=0)
        hist['evals'].append(baseline)
        shutil.copy(cfg['sl_ckpt'], f'{rdir}/best.pth')
        with open(f'{rdir}/hist.json', 'w') as f: json.dump(hist, f, indent=1)
    while it < cfg['max_iters']:
        t0 = time.time()
        pool = sorted(glob.glob(f'{rdir}/pool/*.pth'))
        opp_path = cfg['sl_ckpt'] if (random.random() < cfg.get('p_sl_opponent', 0.5) or not pool) else random.choice(pool[-cfg.get('pool_size', 8):])
        policy.eval()
        os.makedirs(f'{M}/data/rl_gen', exist_ok=True)
        seed0 = 10_000_000 + it * cfg['seeds_per_iter']
        workers = int(cfg.get('gen_workers', 1))
        with tempfile.TemporaryDirectory(prefix=f'{run}_{it}_', dir=f'{M}/data/rl_gen') as gen_dir:
            if workers > 1:
                # Workers load the current policy from disk (bf16 inference copies).
                save(f'{rdir}/gen_policy.pth')
                batch, ranks, mean_pt = generate_and_build(f'{rdir}/gen_policy.pth', opp_path, f'{gen_dir}/roll',
                                                           cfg['seeds_per_iter'], seed0, cfg, kind, workers)
                t_gen = time.time() - t0
            else:
                opponent, _ = load_model(opp_path, dev) if opp_path != cfg['sl_ckpt'] else (sl_model, None)
                ranks = generate(policy, opponent, gen_dir, cfg['seeds_per_iter'], seed0, dev, cfg.get('temp', 1.), kind=kind)
                t_gen = time.time() - t0
                batch, mean_pt = build_batch(gen_dir, grp=grp, dev=dev, kind=kind,
                    max_batch_bytes=cfg.get('max_batch_bytes', 24 * 1024**3), return_mode=return_mode)
                del opponent
        n = len(batch[2])
        t_upd = time.time()
        stats = ppo_update(policy, sl_model, opt, batch, cfg, dev, fwd=fwd, sl_fwd=sl_fwd)
        stats['t_update'] = round(time.time() - t_upd)
        del batch
        it += 1
        r = np.array(ranks) / sum(ranks)
        log = dict(iter=it, opp=os.path.basename(opp_path), samples=n, gen_avg_rank=round(float((r * np.arange(1, 5)).sum()), 3),
                   gen_mean_pt=round(mean_pt, 2), t_gen=round(t_gen), t_total=round(time.time() - t0), **stats)
        print(json.dumps(log), flush=True)
        with open(f'{rdir}/train.jsonl', 'a') as f: f.write(json.dumps(log) + '\n')
        save(state_path, {'opt': opt.state_dict()})
        if it % cfg.get('pool_every', 5) == 0: save(f'{rdir}/pool/it{it:04d}.pth')
        if it % cfg['eval_every'] == 0:
            save(f'{rdir}/cand.pth')
            res = run_eval(f'{rdir}/cand.pth', cfg['sl_ckpt'], cfg['eval_seeds'], f'{rdir}/eval.jsonl', kind=kind, **eval_kw)
            res['iter'] = it; hist['evals'].append(res)
            incumbent = hist.get('best_eval')
            comparison = compare_evals(res, incumbent) if incumbent else None
            # Keep the highest measured full-point checkpoint. The patience
            # counter resets only on a clear paired gain against the saved best.
            if incumbent is None or res['mean_pt'] > incumbent['mean_pt']:
                shutil.copy(f'{rdir}/cand.pth', f'{rdir}/best.pth')
                hist['best_eval'] = res; hist['best_pt'] = res['mean_pt']
            if incumbent is None or comparison['clear_improvement']:
                hist['no_improve'] = 0
            else:
                hist['no_improve'] += 1
            res['paired_comparison'] = comparison
            print('EVAL', json.dumps(res), 'no_improve', hist['no_improve'], flush=True)
            json.dump(hist, open(f'{rdir}/hist.json', 'w'), indent=1)
            if hist['no_improve'] >= cfg.get('patience', 3):
                print('RL_STOP: no significant improvement in', hist['no_improve'], 'evals', flush=True); break
    print('RL_DONE', flush=True)

if __name__ == '__main__':
    main()