"""Isolated tiny CPU rollout->PPO->save/load->duplicate evaluation smoke.
Optional --production-ckpt performs GPU updates on a copy, never saves it.
"""
import argparse, copy, json, os, sys, tempfile, time, subprocess, resource
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import TileTransformer
from rl import generate, build_batch, ppo_update, load_model
from eval import summarize_logs

ap=argparse.ArgumentParser()
ap.add_argument('--production-ckpt')
ap.add_argument('--production-mb',type=int,default=8)
a=ap.parse_args()
torch.set_num_threads(1); torch.manual_seed(42)
cfg=dict(mb=16,ppo_epochs=2,temp=1.,clip=.2,c_v=.5,beta_kl=.05,c_ent=.003)
model_cfg=dict(d_model=32,n_layers=1,n_heads=4,ff_mult=2)
root=Path('/root/gpufree-data/mj/data/smoke'); root.mkdir(exist_ok=True)
t0=time.time()
with tempfile.TemporaryDirectory(prefix='ppo_',dir=root) as tmp:
    policy=TileTransformer(**model_cfg).eval(); sl=copy.deepcopy(policy)
    ranks=generate(policy,sl,str(Path(tmp)/'rollout'),1,12200,'cpu',1.)
    batch,mean_pt=build_batch(str(Path(tmp)/'rollout'),return_mode='terminal')
    from grpnet import GRPNet
    gb,gpt=build_batch(str(Path(tmp)/'rollout'),grp=GRPNet().eval(),dev='cpu',return_mode='grp')
    assert abs(gpt-mean_pt)<1e-9 and len(gb[3])==len(batch[3]) and torch.isfinite(gb[3]).all()
    print('GRP_MODE_OK',len(gb[3]),float(gb[3].abs().mean()),float(batch[3].abs().mean()),flush=True)
    rollout_eval=summarize_logs(str(Path(tmp)/'rollout'),'challenger',12200,0x5eed,1,8,ranks)
    assert abs(mean_pt-rollout_eval['mean_pt']) < 1e-6, 'RL/eval terminal reward mismatch'
    before=copy.deepcopy(policy.state_dict())
    opt=torch.optim.AdamW(policy.parameters(),lr=1e-4)
    stats=ppo_update(policy,sl,opt,batch,cfg,'cpu')
    assert all(np.isfinite(v) for v in stats.values())
    assert stats['samples_seen']==2*len(batch[2])
    assert any(not torch.equal(before[k],v) for k,v in policy.state_dict().items())
    path=Path(tmp)/'tiny.pth'
    torch.save(dict(model=policy.state_dict(),cfg=dict(model=model_cfg)),path)
    restored,_=load_model(path,'cpu')
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in policy.state_dict().items())
    r=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'eval.py'),
        '--challenger',str(path),'--champion','rule','--seeds','2','--device','cpu'],
        text=True,capture_output=True,check=True)
    result=json.loads(next(l.removeprefix('EVAL_RESULT ') for l in r.stdout.splitlines() if l.startswith('EVAL_RESULT ')))
    assert result['games']==8 and result['seed_count']==2 and len(result['seed_pt'])==2
    assert np.isfinite(result['mean_pt']) and np.isfinite(result['se_pt'])
    full=None
    if a.production_ckpt:
        torch.cuda.reset_peak_memory_stats()
        large,_=load_model(a.production_ckpt,'cuda'); reference=copy.deepcopy(large)
        for p in reference.parameters(): p.requires_grad_(False)
        small=tuple(t[:max(32,a.production_mb)] for t in batch)
        large_opt=torch.optim.AdamW(large.parameters(),lr=1e-6)
        full=ppo_update(large,reference,large_opt,small,{**cfg,'mb':a.production_mb,'ppo_epochs':1},'cuda')
        full['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
        del large,reference,large_opt
    print('PPO_HANCHAN_SMOKE_OK',json.dumps(dict(games=sum(ranks),samples=len(batch[2]),
        mean_pt=mean_pt,update=stats,eval=result,production_arch=full,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seconds=time.time()-t0)),flush=True)