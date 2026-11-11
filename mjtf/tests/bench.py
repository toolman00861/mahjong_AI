import sys, time, os, glob, torch
sys.path.insert(0,'/root/gpufree-data/mj/code'); sys.path.insert(0,'/root/gpufree-data/mj/code/mjtf')
from data import GameDataset
from model import TileTransformer, param_count
import torch.nn.functional as F
files=[l.split('\t')[0] for l in open('/root/gpufree-data/mj/data/index/majsoul-jade-2019.tsv')][:60]
ds=GameDataset(files, file_batch=12, augment=False)
t=time.time(); n=0
for x in ds: n+=1
dt=time.time()-t
print(f'loader: {n} samples from {len(files)} games in {dt:.1f}s -> {n/dt:.0f} samples/s/worker, {n/len(files):.0f} samples/game', flush=True)
dev='cuda'
for cfg in [dict(d_model=384,n_layers=8,n_heads=6), dict(d_model=512,n_layers=12,n_heads=8), dict(d_model=768,n_layers=12,n_heads=12)]:
    m=TileTransformer(**cfg).to(dev); opt=torch.optim.AdamW(m.parameters(),1e-4,fused=True)
    cm=torch.compile(m)
    B=1024
    obs=torch.rand(B,1014,34,device=dev).half(); mask=torch.ones(B,46,dtype=torch.bool,device=dev); act=torch.randint(0,46,(B,),device=dev)
    for i in range(25):
        if i==5: torch.cuda.synchronize(); t=time.time()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=cm(obs,mask)
        loss=F.cross_entropy(out['logits'],act)+out['value'].mean()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); dt=(time.time()-t)/20
    print(cfg, 'params %.1fM'%(param_count(m)/1e6), 'train %.0f samples/s'%(B/dt), 'mem %.1fGB'%(torch.cuda.max_memory_allocated()/1e9), flush=True)
    del m,opt,cm; torch.cuda.empty_cache()
