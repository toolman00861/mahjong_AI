"""Micro-benchmark of policy inference throughput (samples/s) for self-play, by variant and batch size.
Variants: autocast (current engine), bf16 weights without autocast (enables nn.Transformer fast path),
and bf16 + SDPA math via torch.compile on a fixed padded batch."""
import sys, time, json, torch
sys.path.insert(0, '/root/gpufree-data/mj/code/mjtf')
from rl import load_model

dev = torch.device('cuda')
m, _ = load_model(sys.argv[1], dev)
mb = load_model(sys.argv[1], dev)[0].to(torch.bfloat16)


def bench(fn, B, iters=30):
    o = torch.randn(B, 1014, 34, device=dev).clamp(0, 1)
    k = torch.ones(B, 46, dtype=torch.bool, device=dev)
    with torch.inference_mode():
        for _ in range(3): fn(o, k)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(iters): fn(o, k)
        torch.cuda.synchronize()
    return B * iters / (time.time() - t)


def autocast(o, k):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return m(o, k)['logits']


def bf16(o, k):
    return mb(o.to(torch.bfloat16), k)['logits']


# Reference check: the bf16 fast path must give nearly the same policy.
with torch.inference_mode():
    o = torch.randn(256, 1014, 34, device=dev).clamp(0, 1); k = torch.ones(256, 46, dtype=torch.bool, device=dev)
    a, b = autocast(o, k).float(), bf16(o, k).float()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print('AGREE', round(agree, 4), 'max_abs_logit_diff', round((a - b).abs().max().item(), 4), flush=True)
for B in (64, 256, 1024):
    r = dict(B=B, autocast=round(bench(autocast, B)), bf16_fastpath=round(bench(bf16, B)))
    print('INFER', json.dumps(r), flush=True)
