import sys,json,numpy as np
sys.path.insert(0,'/root/gpufree-data/mj/code/mjtf')
from train_sl import load_index
from libriichi.dataset import Grp
M='/root/gpufree-data/mj'
for src in ('majsoul-throne-2023','majsoul-throne-2024','majsoul-jade-2019'):
 for split in ('train','val'):
  files,_=load_index([src],['hanchan'],128,split); files=set(files)
  checked=adjusted=bad=0
  for line in open(f'{M}/data/index/{src}.tsv'):
   f,_,_,_,_,sc=line.rstrip().split('\t')
   if f not in files:continue
   ix=np.array([int(x) for x in sc.split(',')]); actual=np.array(Grp.load_gz_log_files([f])[0].take_final_scores())
   if ix.sum()<100000:
    ix[ix.argmax()]+=100000-ix.sum(); adjusted+=1
   checked+=1;bad+=int(not np.array_equal(ix,actual))
  print(src,split,'checked',checked,'deposit_adjustments',adjusted,'unexplained',bad,flush=True)
