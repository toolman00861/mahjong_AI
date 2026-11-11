"""Evaluate a frozen checkpoint using the exact SL validation sampling settings."""
import os,sys,json,argparse
sys.path.insert(0,'/root/gpufree-data/mj/code/mjtf')
import torch
from torch.utils.data import DataLoader
from train_sl import load_index,evaluate
from data import GameDataset,collate
from model import TileTransformer

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--ckpt',required=True);ap.add_argument('--batches',type=int,default=100);a=ap.parse_args()
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=True
 st=torch.load(a.ckpt,map_location='cpu',weights_only=False);cfg=st['cfg']
 files,kinds=load_index(cfg['val_sources'],cfg.get('val_kinds',cfg.get('kinds')),3000,'val')
 ds=GameDataset(files,file_batch=16,shuffle=True,epochs=1,augment=False,kinds=kinds,seed=123)
 dl=DataLoader(ds,batch_size=1024,num_workers=3,collate_fn=collate)
 model=TileTransformer(**cfg['model']).cuda();model.load_state_dict(st['model'])
 r=evaluate(model,dl,'cuda',cfg,a.batches);r.update(step=st['step'],checkpoint=a.ckpt)
 print('FROZEN_EVAL',json.dumps(r),flush=True)
if __name__=='__main__':main()
