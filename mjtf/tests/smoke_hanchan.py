"""CPU integration smoke test for migrated hanchan arena and GRP targets."""
import sys,os,tempfile,gzip,json
from pathlib import Path
sys.path.insert(0,'/root/gpufree-data/mj/code/mjtf')
import torch,numpy as np
from libriichi.arena import OneVsThree
from libriichi.state import PlayerState
from engine import RuleEngine
from grpnet import GRPNet
from rl import build_batch
M=Path('/root/gpufree-data/mj')
(M/'data/smoke').mkdir(exist_ok=True)
d=tempfile.mkdtemp(prefix='hanchan_',dir=M/'data/smoke')
torch.set_num_threads(1)
r=OneVsThree(disable_progress_bar=True,log_dir=d,game_length=8).py_vs_py(challenger=RuleEngine('challenger',1),champion=RuleEngine('champion',2),seed_start=(10300,0x1234),seed_count=1)
assert sum(r)==4
last=[]
for p in Path(d).glob('*.json.gz'):
 ev=[json.loads(l) for l in gzip.open(p,'rt')]; starts=[e for e in ev if e['type']=='start_kyoku']
 assert all(not e.get('tonpuu',False) for e in starts)
 last.append(starts[-1]['bakaze']+str(starts[-1]['kyoku']))
 ps=PlayerState(0); ps.update(json.dumps(starts[0])); obs,mask=ps.encode_obs(6,False)
 assert obs.shape==(1014,34) and np.all(obs[1012]==0)
grp=GRPNet().eval()
(obs,mask,act,ret),pt=build_batch(d,grp,'cpu',kind='hanchan')
assert len(act)>0 and obs.shape[1:]==(1014,34)
assert torch.isfinite(ret).all()
assert mask[torch.arange(len(act)),act].all()
print('HANCHAN_SMOKE_OK',{'games':sum(r),'last_rounds':last,'samples':len(act),'mean_pt':pt,'logs':d},flush=True)
