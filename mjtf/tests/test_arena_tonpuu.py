import sys, glob, gzip, json, collections
sys.path.insert(0,'/root/gpufree-data/mj/code'); sys.path.insert(0,'/root/gpufree-data/mj/code/mjtf')
from libriichi.arena import OneVsThree
from engine import RuleEngine
import shutil, os
for L in (4, 8):
    d=f'/tmp/arena_test_{L}'; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    env=OneVsThree(disable_progress_bar=True, log_dir=d, game_length=L)
    r=env.py_vs_py(challenger=RuleEngine('a',1), champion=RuleEngine('b',2), seed_start=(10000, 0x1234), seed_count=50)
    c=collections.Counter(); flags=collections.Counter()
    for f in glob.glob(d+'/*.json.gz'):
        evs=[json.loads(l) for l in gzip.open(f,'rt')]
        ks=[e for e in evs if e['type']=='start_kyoku']
        c[ks[-1]['bakaze']+str(ks[-1]['kyoku'])]+=1
        flags[ks[0].get('tonpuu')]+=1
    print('length',L,'rankings',r,'last kyoku dist',sorted(c.items()),'tonpuu flag',dict(flags))
