"""Controlled short online-loss comparison. Does not overwrite production checkpoints."""
import os,sys,json,time,gc,argparse
sys.path.insert(0,'/root/gpufree-data/mj/code/mjtf')
import torch,numpy as np
from torch.utils.data import DataLoader
from model import TileTransformer
from train_sl import load_index,compute_loss,evaluate
from data import GameDataset,collate
M='/root/gpufree-data/mj'

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--buffer',type=int,default=0); ap.add_argument('--cap',type=int,default=0); ap.add_argument('--steps',type=int,default=300); args=ap.parse_args()
 torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=True
 st=torch.load(f'{M}/ckpt/sl_a/last.pth',map_location='cpu',weights_only=False); cfg=st['cfg']
 files,kinds=load_index(cfg['train_sources'],['hanchan'],None,'train')
 va,vk=load_index(cfg['val_sources'],['hanchan'],3000,'val'); kinds.update(vk)
 ds=GameDataset(files,epochs=1,augment=True,kinds=kinds,seed=36000,file_batch=4 if args.buffer else 16,shuffle_buffer=args.buffer,max_samples_per_player=args.cap)
 dl=DataLoader(ds,batch_size=1024,num_workers=8,prefetch_factor=2,pin_memory=True,collate_fn=collate)
 vd=GameDataset(va,file_batch=16,seed=123,kinds=kinds)
 model=TileTransformer(**cfg['model']).cuda(); model.load_state_dict(st['model'])
 decay=[p for n,p in model.named_parameters() if p.dim()>=2 and 'emb' not in n and n!='cls']
 no_decay=[p for n,p in model.named_parameters() if not (p.dim()>=2 and 'emb' not in n and n!='cls')]
 opt=torch.optim.AdamW([{'params':decay,'weight_decay':cfg['wd']},{'params':no_decay,'weight_decay':0.}],lr=0.0002914,betas=(.9,.98),fused=True)
 opt.load_state_dict(st['opt']); del st
 vl=DataLoader(vd,batch_size=1024,num_workers=3,collate_fn=collate)
 print('PROBE_BASE',json.dumps(evaluate(model,vl,'cuda',cfg,40)),flush=True)
 cm=torch.compile(model); model.train(); agg={}; start=time.time()
 for i,batch in enumerate(dl,1):
  opt.zero_grad(set_to_none=True); loss,parts,*_=compute_loss(cm,batch,'cuda',cfg)
  loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
  for k,v in parts.items():agg[k]=agg.get(k,0)+v
  if i%50==0:
   print('PROBE',args.buffer,i,{k:round(v/50,4) for k,v in agg.items()},'seconds',round(time.time()-start),flush=True);agg={}
  if i>=args.steps:break
 # Workers exit before evaluation; no parent-side Rust pool was initialized.
 vl=DataLoader(vd,batch_size=1024,num_workers=3,collate_fn=collate)
 print('PROBE_EVAL',args.buffer,args.cap,json.dumps(evaluate(cm,vl,'cuda',cfg,40)),flush=True)
 print('PROBE_DONE',flush=True)
if __name__=='__main__':main()
