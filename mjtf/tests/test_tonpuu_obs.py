import sys, json, numpy as np
sys.path.insert(0,'/root/gpufree-data/mj/code')
from libriichi.state import PlayerState
from libriichi.arena import OneVsThree
tehais=[["1m","2m","3m","7m","9m","1p","7p","9p","2s","3s","5s","5sr","7s"],["1m","7m","8m","8m","3p","6p","7p","8p","9p","1s","2s","4s","8s"],["1m","2m","6m","7m","2p","5p","5p","6p","7p","3s","9s","P","C"],["2m","4m","5m","5m","6m","5pr","9p","6s","S","W","N","C","C"]]
def obs_flags(bakaze,kyoku,tonpuu):
    ps=PlayerState(0)
    ev={"type":"start_kyoku","bakaze":bakaze,"dora_marker":"1p","kyoku":kyoku,"honba":0,"kyotaku":0,"oya":kyoku-1,"scores":[25000]*4,"tehais":tehais}
    if tonpuu is not None: ev["tonpuu"]=tonpuu
    ps.update(json.dumps(ev))
    ps.update(json.dumps({"type":"tsumo","actor":0,"pai":"1s"}))
    obs,mask=ps.encode_obs(5,False)
    return float(obs[1012,0]), float(obs[1013,0]), obs.shape
cases=[("E",1,None,(0,0)),("E",4,None,(0,0)),("S",4,None,(0,1)),("E",4,True,(1,1)),("E",3,True,(1,0)),("S",1,True,(1,1)),("W",1,None,(0,1))]
ok=True
for b,k,t,exp in cases:
    a,c,sh=obs_flags(b,k,t)
    r=(a,c)==exp; ok&=r
    print(b,k,t,'->',(a,c),sh,'OK' if r else 'FAIL')
print('OBS_TEST', 'PASS' if ok else 'FAIL')
