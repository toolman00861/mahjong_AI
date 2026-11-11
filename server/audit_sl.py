"""Read-only checkpoint/data audit before resuming SL."""
import sys, os, json, random, collections, gzip
sys.path.insert(0, '/root/gpufree-data/mj/code/mjtf')
import numpy as np, torch
from torch.utils.data import DataLoader
from train_sl import load_index, evaluate
from data import GameDataset, collate
from model import TileTransformer
from libriichi.dataset import Grp
M='/root/gpufree-data/mj'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=True
st=torch.load(f'{M}/ckpt/sl_a/last.pth', map_location='cpu', weights_only=False)
cfg=st['cfg']; model=TileTransformer(**cfg['model']).cuda(); model.load_state_dict(st['model'])
print('checkpoint', st['step'], flush=True)
for src in cfg['train_sources']:
    for split in ('train','val'):
        files, kinds=load_index([src], ['hanchan'], 128, split)
        if not files: continue
        # Verify labels against the independent preprocessing index and raw terminal events.
        chosen=set(files); mismatches=[]; nk=[]; scores=[]
        for line in open(f'{M}/data/index/{src}.tsv'):
            f,_,kind,n,fr,sc=line.rstrip().split('\t')
            if f not in chosen: continue
            g=Grp.load_gz_log_files([f])[0]
            fs=np.array(g.take_final_scores()); ix=np.array([int(x) for x in sc.split(",")])
            if not np.array_equal(fs, ix): mismatches.append((os.path.basename(f), fs.tolist(), ix.tolist()))
            nk.append(int(n)); scores.extend(fs.tolist())
        print('LABELS',src,split,'n',len(files),'mismatches',len(mismatches),'examples',mismatches[:3], 'mean_rounds',np.mean(nk),flush=True)
        ds=GameDataset(files, file_batch=4, kinds=kinds, seed=123, augment=False)
        dl=DataLoader(ds,batch_size=512,num_workers=0,collate_fn=collate)
        print('FROZEN',src,split,json.dumps(evaluate(model,dl,'cuda',cfg,24)),flush=True)
# train/eval must agree because dropout is zero; test same fixed minibatch.
files,kinds=load_index(cfg['val_sources'],['hanchan'],16,'val')
batch=next(iter(DataLoader(GameDataset(files,file_batch=1,kinds=kinds),batch_size=128,collate_fn=collate)))
obs,mask=batch[0].cuda(),batch[1].cuda()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    model.train(); a=model(obs,mask)
    model.eval(); b=model(obs,mask)
print('MODE_DIFF',{k:float((a[k]-b[k]).abs().max()) for k in a},flush=True)
print('AUDIT_DONE',flush=True)


