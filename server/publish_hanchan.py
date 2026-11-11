"""Quality-filter a staged hanchan index and publish it to data/index.
Drops games whose (estimated) final scores contradict the hanchan ending rules:
  * ended in S1-S3 (or W1-W3 not by score) without anyone below 0  -> estimate missed a tobi/ura etc.
  * ended at S4/W* while nobody reached 30000 (except W4, the hard end)
Usage: python publish_hanchan.py 2022 [2023 ...]"""
import os, sys, json, shutil
M = '/root/gpufree-data/mj'
for y in sys.argv[1:]:
    src = f'majsoul-jade-{y}'
    st = f'{M}/data/index_hanchan/{src}.tsv'
    assert os.path.exists(st + '.done'), f'{src} not finished'
    keep, drop = [], {}
    for l in open(st):
        p = l.rstrip('\n').split('\t')
        if p[2] != 'hanchan': drop['kind'] = drop.get('kind', 0) + 1; continue
        fr = p[4]; sc = list(map(int, p[5].split(',')))
        tobi = min(sc) < 0
        if fr in ('E1','E2','E3','E4','S1','S2','S3') and not tobi:
            drop['early_end_no_tobi'] = drop.get('early_end_no_tobi', 0) + 1; continue
        if fr in ('S4','W1','W2','W3') and max(sc) < 30000 and not tobi:
            drop['end_below_30000'] = drop.get('end_below_30000', 0) + 1; continue
        keep.append(l)
    dst = f'{M}/data/index/{src}.tsv'
    if os.path.exists(dst) and not os.path.exists(dst + '.tonpuu_only'):
        shutil.move(dst, dst + '.tonpuu_only')
        if os.path.exists(dst + '.done'): shutil.move(dst + '.done', dst + '.done.tonpuu_only')
    with open(dst + '.tmp', 'w') as f: f.writelines(keep)
    os.replace(dst + '.tmp', dst)
    for ext in ('.bad', '.valerr'):
        if os.path.exists(st + ext): shutil.copy(st + ext, dst + ext)
    d = json.load(open(st + '.done')); d['published'] = len(keep); d['quality_drop'] = drop
    json.dump(d, open(dst + '.done', 'w'))
    print(src, 'published', len(keep), 'dropped', drop, flush=True)
