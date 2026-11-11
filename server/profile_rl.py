"""Throughput profiler for one RL iteration, split into generate / build_batch / ppo_update.
Runs on a throwaway copy of the policy; never writes to ckpt/. Prints one PROFILE json line per phase.

Usage: python code/profile_rl.py --sl ckpt/sl_a/s8_candidate.pth --phase gen --seeds 250
       python code/profile_rl.py --sl ... --phase update --gen_dir data/rl_gen/prof --mb 1024 [--compile]
"""
import os, sys, json, time, argparse, tempfile, threading, subprocess
sys.path.insert(0, '/root/gpufree-data/mj/code/mjtf')
import torch
import rl

M = '/root/gpufree-data/mj'


class Sampler:
    """Samples GPU utilisation and process RSS every second in a background thread."""
    def __init__(self):
        self.gpu, self.rss, self.cpu, self.stop = [], [], [], False
        self.t = threading.Thread(target=self.run, daemon=True); self.t.start()

    def run(self):
        last = None
        while not self.stop:
            try:
                g = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
                u, mem = map(int, g.strip().split(', '))
                self.gpu.append((u, mem))
                with open('/proc/self/status') as f:
                    for l in f:
                        if l.startswith('VmRSS'): self.rss.append(int(l.split()[1]) // 1024)
                with open('/proc/self/stat') as f:
                    x = f.read().split(); ticks = int(x[13]) + int(x[14])
                now = time.time()
                if last: self.cpu.append((ticks - last[0]) / os.sysconf('SC_CLK_TCK') / (now - last[1]))
                last = (ticks, now)
            except Exception:
                pass
            time.sleep(1)

    def done(self):
        self.stop = True; self.t.join()
        u = [a for a, _ in self.gpu] or [0]
        return dict(gpu_util_mean=round(sum(u) / len(u), 1), gpu_mem_peak_MiB=max((b for _, b in self.gpu), default=0),
                    rss_peak_MiB=max(self.rss, default=0),
                    cpu_cores_mean=round(sum(self.cpu) / max(len(self.cpu), 1), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sl', required=True)
    ap.add_argument('--phase', choices=('gen', 'build', 'update'), required=True)
    ap.add_argument('--seeds', type=int, default=250)
    ap.add_argument('--gen_dir', default=f'{M}/data/rl_gen/prof')
    ap.add_argument('--mb', type=int, default=1024)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    cfg = json.load(open(f'{M}/code/mjtf/cfg/rl.json'))
    cfg.update(mb=a.mb, ppo_epochs=a.epochs)
    dev = torch.device('cuda'); torch.backends.cuda.matmul.allow_tf32 = True
    policy, _ = rl.load_model(a.sl, dev)
    out = dict(phase=a.phase, tag=a.tag, rayon=os.environ.get('RAYON_NUM_THREADS'))
    if a.phase == 'gen':
        import shutil
        if os.path.exists(a.gen_dir): shutil.rmtree(a.gen_dir)
        opponent, _ = rl.load_model(a.sl, dev)
        s = Sampler(); t = time.time()
        rl.generate(policy, opponent, a.gen_dir, a.seeds, 10_000_000, dev, 1.0)
        out.update(seconds=round(time.time() - t, 1), seeds=a.seeds, **s.done())
    else:
        gs = torch.load(cfg['grp_ckpt'], map_location='cpu', weights_only=False)
        grp = rl.GRPNet().to(dev); grp.load_state_dict(gs['model']); grp.eval()
        s = Sampler(); t = time.time()
        batch, _ = rl.build_batch(a.gen_dir, grp=grp, dev=dev)
        t_build = time.time() - t
        out.update(build_seconds=round(t_build, 1), samples=len(batch[2]))
        if a.phase == 'update':
            sl_model, _ = rl.load_model(a.sl, dev)
            for p in sl_model.parameters(): p.requires_grad_(False)
            if a.compile:
                policy = torch.compile(policy, dynamic=True); sl_model = torch.compile(sl_model, dynamic=True)
            opt = torch.optim.AdamW(policy.parameters(), lr=cfg['lr'], weight_decay=0.0, betas=(0.9, 0.99))
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            stats = rl.ppo_update(policy, sl_model, opt, batch, cfg, dev)
            torch.cuda.synchronize()
            out.update(update_seconds=round(time.time() - t, 1), mb=a.mb, epochs=a.epochs, compile=a.compile,
                       torch_peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2), kl=stats['kl_sl'])
        out.update(**s.done())
    print('PROFILE', json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
