import os, sys, time, glob, math, json, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/root/gpufree-data/mj/code')
from model import TileTransformer, param_count
from data import GameDataset, collate

M = '/root/gpufree-data/mj'

import zlib
def is_val(f):
    return zlib.crc32(os.path.basename(f).encode()) % 100 == 0

def load_index(sources, kinds=None, max_files=None, split=None):
    files, kmap = [], {}
    for src in sources:
        p = f'{M}/data/index/{src}.tsv'
        if not os.path.exists(p + '.done'):
            print('index not ready', p); continue
        badp = f'{M}/data/index/{src}.bad'
        bad = set(l.strip() for l in open(badp)) if os.path.exists(badp) else set()
        for line in open(p):
            f, s, kind, nk, fr, sc = line.rstrip('\n').split('\t')
            if kind not in ('tonpuu', 'hanchan') or f in bad:
                continue
            if kinds and kind not in kinds:
                continue
            if split is not None and is_val(f) != (split == 'val'):
                continue
            files.append(f); kmap[f] = kind
    if max_files:
        random.Random(1).shuffle(files); files = files[:max_files]
    return files, kmap

def cat_of(a):
    # action category for metrics
    a = a.clone()
    c = torch.zeros_like(a)
    c[a <= 36] = 0          # discard / kan-select
    c[a == 37] = 1          # riichi
    c[(a >= 38) & (a <= 41)] = 2  # chi/pon
    c[a == 42] = 3          # kan
    c[a == 43] = 4          # agari
    c[a == 44] = 5          # ryukyoku
    c[a == 45] = 6          # pass
    return c
CATS = ['discard', 'riichi', 'call', 'kan', 'agari', 'ryukyoku', 'pass']

def compute_loss(model, batch, dev, cfg):
    obs, mask, act, rank, delta, pt, tonpuu = [x.to(dev, non_blocking=True) for x in batch[:7]]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(obs, mask)
    logits = out['logits']
    ce = F.cross_entropy(logits, act)
    rank_loss = F.cross_entropy(out['rank'], rank)
    delta_loss = F.smooth_l1_loss(out['delta'], delta)
    value_loss = F.mse_loss(out['value'], pt)
    loss = ce + cfg['w_rank'] * rank_loss + cfg['w_delta'] * delta_loss + cfg['w_value'] * value_loss
    with torch.no_grad():
        pred = logits.argmax(-1)
        correct = (pred == act)
        # only count samples with >1 legal action
        multi = mask.sum(-1) > 1
    return loss, dict(ce=ce.item(), rank=rank_loss.item(), delta=delta_loss.item(), value=value_loss.item()), correct, multi, act

def evaluate(model, loader, dev, cfg, max_batches):
    model.eval()
    tot = {}; n = 0
    cat_c = torch.zeros(len(CATS)); cat_n = torch.zeros(len(CATS))
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches: break
            _, parts, correct, multi, act = compute_loss(model, batch, dev, cfg)
            for k, v in parts.items(): tot[k] = tot.get(k, 0) + v
            n += 1
            c = cat_of(act)[multi].cpu(); cr = correct[multi].float().cpu()
            cat_n += torch.bincount(c, minlength=len(CATS)).float()
            cat_c += torch.bincount(c, weights=cr, minlength=len(CATS)).float()
    model.train()
    res = {k: v / max(n, 1) for k, v in tot.items()}
    for i, name in enumerate(CATS):
        if cat_n[i] > 0: res[f'acc_{name}'] = (cat_c[i] / cat_n[i]).item()
    res['acc_all'] = (cat_c.sum() / cat_n.sum().clamp(min=1)).item()
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.cfg))
    run = cfg['run']; rdir = f'{M}/ckpt/{run}'; os.makedirs(rdir, exist_ok=True)
    json.dump(cfg, open(f'{rdir}/cfg.json', 'w'), indent=1)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    dev = torch.device('cuda')
    tr_files, kmap = load_index(cfg['train_sources'], cfg.get('kinds'), cfg.get('max_files'), split=cfg.get('split', 'train'))
    va_files, vkmap = load_index(cfg['val_sources'], cfg.get('val_kinds', cfg.get('kinds')), 3000, split=cfg.get('val_split', 'val'))
    if not tr_files or not va_files:
        raise RuntimeError('Training and validation indices must both contain usable games')
    if set(tr_files).intersection(va_files):
        raise RuntimeError('Training/validation file overlap')
    kmap.update(vkmap)
    print(f'train files {len(tr_files)}  val files {len(va_files)}', flush=True)
    ds = GameDataset(tr_files, version=6, file_batch=cfg.get('file_batch', 16), epochs=cfg.get('epochs', 1),
                     augment=cfg.get('augment', True), kinds=kmap, seed=cfg.get('seed', 0),
                     shuffle_buffer=cfg.get('shuffle_buffer', 0),
                     max_samples_per_player=cfg.get('max_samples_per_player', 0))
    dl = DataLoader(ds, batch_size=cfg['batch_size'], num_workers=cfg.get('workers', 11), collate_fn=collate,
                    pin_memory=True, prefetch_factor=cfg.get('prefetch_factor', 2), persistent_workers=False, drop_last=True)
    vds = GameDataset(va_files, version=6, file_batch=16, shuffle=True, epochs=1, augment=False, kinds=kmap, seed=123)
    model = TileTransformer(**cfg['model']).to(dev)
    print('params', param_count(model), flush=True)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and 'emb' not in n and 'cls' != n]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and 'emb' not in n and 'cls' != n)]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': cfg.get('wd', 0.05)},
                             {'params': no_decay, 'weight_decay': 0.}], lr=cfg['lr'], betas=(0.9, 0.98), fused=True)
    total = cfg['max_steps']; warm = cfg.get('warmup', 2000)
    def lr_at(s):
        if s < warm: return cfg['lr'] * s / warm
        p = min(1., (s - warm) / max(1, total - warm))
        return cfg['lr'] * (cfg.get('final_lr_frac', 0.05) + (1 - cfg.get('final_lr_frac', 0.05)) * 0.5 * (1 + math.cos(math.pi * p)))
    step = 0
    ck = f'{rdir}/last.pth'
    init = cfg.get('init_from')
    if os.path.exists(ck):
        st = torch.load(ck, map_location='cpu', weights_only=False)
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); step = st['step']
        print('resumed at step', step, flush=True)
        ds.seed = cfg.get('seed', 0) + step  # new shuffle order after resume
    elif init:
        st = torch.load(init, map_location='cpu', weights_only=False)
        model.load_state_dict(st['model']); print('init from', init, flush=True)
    cmodel = torch.compile(model) if cfg.get('compile', True) else model
    tb = SummaryWriter(f'{rdir}/tb')
    best = json.load(open(f'{rdir}/best.json')) if os.path.exists(f'{rdir}/best.json') else {'acc_all': 0}
    t0 = time.time(); acc_n = 0; acc_c = 0; agg = {}; samples = 0
    model.train()
    val_limit_hits = 0
    while step < total:
        pass_start = step
        for batch in dl:
            for g in opt.param_groups: g['lr'] = lr_at(step)
            loss, parts, correct, multi, act = compute_loss(cmodel, batch, dev, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss at step {step}: {parts}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('clip', 1.0))
            opt.step()
            step += 1; samples += act.shape[0]
            for k, v in parts.items(): agg[k] = agg.get(k, 0) + v
            acc_c += correct[multi].sum().item(); acc_n += multi.sum().item()
            if step % cfg.get('log_every', 100) == 0:
                dt = time.time() - t0
                msg = {k: round(v / cfg.get('log_every', 100), 4) for k, v in agg.items()}
                msg.update(step=step, acc=round(acc_c / max(acc_n, 1), 4), lr=lr_at(step), gn=round(gn.item(), 3),
                           sps=round(samples / dt))
                print(json.dumps(msg), flush=True)
                for k, v in msg.items(): tb.add_scalar(f'train/{k}', v, step)
                agg = {}; acc_c = acc_n = 0; samples = 0; t0 = time.time()
            if step % cfg.get('save_every', 2000) == 0 or step >= total:
                torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step, 'cfg': cfg}, ck + '.tmp')
                os.replace(ck + '.tmp', ck)
            if step % cfg.get('eval_every', 10000) == 0 or step >= total:
                vdl = DataLoader(vds, batch_size=1024, num_workers=3, collate_fn=collate)
                res = evaluate(cmodel, vdl, dev, cfg, cfg.get('eval_batches', 150))
                res['step'] = step
                print('EVAL', json.dumps(res), flush=True)
                for k, v in res.items(): tb.add_scalar(f'val/{k}', v, step)
                with open(f'{rdir}/eval.jsonl', 'a') as f: f.write(json.dumps(res) + '\n')
                if res['acc_all'] > best['acc_all']:
                    best = res
                    torch.save({'model': model.state_dict(), 'step': step, 'cfg': cfg}, f'{rdir}/best.pth')
                    json.dump(best, open(f'{rdir}/best.json', 'w'))
                limits = cfg.get('val_loss_limits', {})
                exceeded = {k: res.get(k) for k, limit in limits.items()
                            if k not in res or not math.isfinite(res[k]) or res[k] > limit}
                val_limit_hits = val_limit_hits + 1 if exceeded else 0
                if val_limit_hits >= cfg.get('val_limit_patience', 3):
                    raise RuntimeError(f'VALIDATION_GUARD_STOP step={step}: {exceeded}; checkpoint preserved')
                t0 = time.time(); samples = 0
            if step >= total: break
        else:
            if step == pass_start:
                raise RuntimeError('Training dataset yielded no full batches')
            ds.seed += 1
    print('TRAIN_DONE', flush=True)

if __name__ == '__main__':
    main()