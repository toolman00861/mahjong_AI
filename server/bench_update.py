"""Steady-state PPO update speed: eager vs torch.compile(dynamic=False), measured on the 2nd/3rd call
(compile cost is paid once per RL process, so the first call is reported separately). Throwaway copies only."""
import sys, time, json, torch
sys.path.insert(0, '/root/gpufree-data/mj/code/mjtf')
import rl

M = '/root/gpufree-data/mj'
sl_path, mode, mb = sys.argv[1], sys.argv[2], int(sys.argv[3])
dev = torch.device('cuda'); torch.backends.cuda.matmul.allow_tf32 = True
cfg = json.load(open(f'{M}/code/mjtf/cfg/rl.json')); cfg.update(mb=mb)
gs = torch.load(cfg['grp_ckpt'], map_location='cpu', weights_only=False)
grp = rl.GRPNet().to(dev); grp.load_state_dict(gs['model']); grp.eval()
batch, _ = rl.build_batch(f'{M}/data/rl_gen/prof', grp=grp, dev=dev)
policy, _ = rl.load_model(sl_path, dev); sl, _ = rl.load_model(sl_path, dev)
for p in sl.parameters(): p.requires_grad_(False)
if mode == 'compile':
    torch._dynamo.config.cache_size_limit = 16
    policy_c = torch.compile(policy, dynamic=False); sl_c = torch.compile(sl, dynamic=False)
else:
    policy_c, sl_c = policy, sl
opt = torch.optim.AdamW(policy.parameters(), lr=0.0, weight_decay=0.0)  # lr 0: timing only
times = []
for k in range(3):
    torch.cuda.synchronize(); t = time.time()
    rl.ppo_update(policy_c, sl_c, opt, batch, cfg, dev)
    torch.cuda.synchronize(); times.append(round(time.time() - t, 1))
print('UPDATE', json.dumps(dict(mode=mode, mb=mb, samples=len(batch[2]), seconds=times,
                                peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2))), flush=True)
