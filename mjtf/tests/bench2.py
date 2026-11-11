import sys, time, numpy as np
sys.path.insert(0,'/root/gpufree-data/mj/code')
from libriichi.dataset import GameplayLoader
bad=set(l.strip() for l in open('/root/gpufree-data/mj/data/index/majsoul-jade-2019.bad'))
files=[l.split('\t')[0] for l in open('/root/gpufree-data/mj/data/index/majsoul-jade-2019.tsv') if l.split('\t')[0] not in bad and l.split('\t')[2] in ('tonpuu','hanchan')][100:130]
for v in (6,3):
    L=GameplayLoader(v, oracle=False)
    t=time.time(); g=L.load_gz_log_files(files); t1=time.time()-t
    n=0; t=time.time()
    for fg in g:
        for x in fg:
            o=x.take_obs(); n+=len(o)
            arr=np.stack(o).astype(np.float16)
    t2=time.time()-t
    print(f'v{v}: load {t1:.2f}s python {t2:.2f}s samples {n} -> {n/(t1+t2):.0f}/s')