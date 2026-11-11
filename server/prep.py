#!/usr/bin/env python3
"""Convert dataset tar shards (per-game .mjai.json.zst) into per-game .json.gz files
usable by libriichi GameplayLoader. Classifies tonpuu (East-only) games and injects
"tonpuu":true into their start_kyoku events. Writes an index TSV per shard."""
import sys, os, io, json, gzip, tarfile, multiprocessing as mp, time, shutil
import zstandard as zstd

import random
KEEP_HANCHAN = float(os.environ.get('KEEP_HANCHAN', '1'))
RAW = '/root/gpufree-data/mj/data/raw'
OUT = os.environ.get('PREP_OUT', '/root/gpufree-data/mj/data/games')
IDX = os.environ.get('PREP_IDX', '/root/gpufree-data/mj/data/index')
ONLY_KIND = os.environ.get('PREP_KIND')
MIN_FREE = float(os.environ.get('MIN_FREE_GB', '10')) * 1024**3

def classify(lines):
    """returns (kind, n_kyoku, final_round, final_scores, ok)"""
    ev0 = json.loads(lines[0])
    if ev0.get('type') != 'start_game':
        return None
    cur = None; rounds = []; final = None; ended = False; s_over30 = False
    for l in lines[1:]:
        if l.startswith('{"type":"start_kyoku"'):
            e = json.loads(l); cur = list(e['scores']); rounds.append((e['bakaze'], e['kyoku']))
            if e['bakaze'] != 'E' and max(cur) >= 30000:
                s_over30 = True
        elif l.startswith('{"type":"reach_accepted"'):
            e = json.loads(l); cur[e['actor']] -= 1000
        elif l.startswith('{"type":"hora"') or l.startswith('{"type":"ryukyoku"'):
            e = json.loads(l)
            d = e.get('deltas')
            if d: cur = [a + b for a, b in zip(cur, d)]
        elif l.startswith('{"type":"end_game"'):
            ended = True
    if not rounds or not ended or cur is None:
        return None
    fb, fk = rounds[-1]
    # final scores of repaired jade logs are estimated (no ura/kan dora) -> use a safety margin
    near_tobi = min(cur) < 3000
    if fb in ('W', 'N') or s_over30:
        # a tonpuu game would have ended before reaching a South kyoku with someone >= 30000
        kind = 'hanchan'
    elif fb == 'E':
        # hanchan cannot end in East without tobi; tonpuu ends normally at E4
        kind = 'tonpuu' if (fk == 4 and not near_tobi) else 'unknown_tobi'
    elif fk < 4 and not near_tobi and max(cur) >= 30000:
        # South sudden death (tonpuu extension); a hanchan cannot end here without tobi
        kind = 'tonpuu'
    else:
        kind = 'unknown_tobi'
    return kind, len(rounds), f'{fb}{fk}', cur

def fix_haipai(lines):
    """Mahjong Soul logs give the dealer 14 haipai tiles; convert to 13 + tsumo event."""
    out = []
    for i, l in enumerate(lines):
        if l.startswith('{"type":"start_kyoku"'):
            e = json.loads(l)
            th = e['tehais']
            if any(len(t) == 14 for t in th):
                a = next(j for j, t in enumerate(th) if len(t) == 14)
                t14 = th[a]
                nxt = json.loads(lines[i + 1]) if i + 1 < len(lines) else {}
                pick = len(t14) - 1
                if nxt.get('type') == 'dahai' and nxt.get('actor') == a:
                    if nxt.get('tsumogiri') and nxt['pai'] in t14:
                        pick = t14.index(nxt['pai'])
                    else:
                        for j in range(len(t14) - 1, -1, -1):
                            if t14[j] != nxt['pai']:
                                pick = j; break
                tile = t14.pop(pick)
                out.append(json.dumps(e, separators=(',', ':'), ensure_ascii=False))
                out.append(json.dumps({'type': 'tsumo', 'actor': a, 'pai': tile}, separators=(',', ':')))
                continue
        out.append(l)
    return out

def deaka(t):
    return t[:-1] if t.endswith('r') else t

def _state_upto(pid, evs):
    from libriichi.state import PlayerState
    ps = PlayerState(pid)
    for e in evs:
        ps.update(json.dumps(e))
    return ps

def _final_deltas(seg, start):
    """estimate deltas of the last kyoku (no following start_kyoku) with libriichi scoring (no ura / kan dora)."""
    sticks = start['kyotaku'] + sum(1 for e in seg if e['type'] == 'reach_accepted')
    h = start['honba']; oya = start['oya']
    terms = [i for i, e in enumerate(seg) if e['type'] in ('hora', 'ryukyoku')]
    first = terms[0]
    d = [0, 0, 0, 0]
    if seg[first]['type'] == 'hora':
        for n, i in enumerate(t for t in terms if seg[t]['type'] == 'hora'):
            e = seg[i]; a, t = e['actor'], e['target']
            ps = _state_upto(a, seg[:first])
            ron, ko, oy = ps.agari_points(a != t)
            if a != t:
                pay = ron + 300 * h
                d[a] += pay; d[t] -= pay
            else:
                for o in range(4):
                    if o == a: continue
                    pay = (ko if (a == oya or o != oya) else oy) + 100 * h
                    d[o] -= pay; d[a] += pay
            if n == 0:
                d[a] += 1000 * sticks
    else:
        if seg[first].get('reason'):
            return d
        ten = [_state_upto(p, seg[:first]).shanten == 0 for p in range(4)]
        nt = sum(ten)
        if 0 < nt < 4:
            for p in range(4):
                d[p] = 3000 // nt if ten[p] else -3000 // (4 - nt)
    return d

def repair(lines):
    """Repair the lossy Mahjong Soul jade conversion: wrong call targets, missing kakan consumed,
    missing hora/ryukyoku deltas and ura_markers. Returns (lines, n_estimated) or raises."""
    evs = [json.loads(l) for l in lines]
    need = any(e['type'] == 'hora' and 'deltas' not in e for e in evs) or \
           any(e['type'] in ('chi', 'pon', 'daiminkan') and e.get('target') == e['actor'] for e in evs)
    if not need:
        return lines, 0
    last_dahai = None; pons = [{} for _ in range(4)]; hands = [[] for _ in range(4)]
    def take(a, tiles):
        for x in tiles:
            hands[a].remove(x)
    for e in evs:
        t = e['type']
        if t == 'start_kyoku':
            last_dahai = None; pons = [{} for _ in range(4)]; hands = [list(h) for h in e['tehais']]
        elif t == 'tsumo':
            hands[e['actor']].append(e['pai'])
        elif t == 'dahai':
            last_dahai = e['actor']; take(e['actor'], [e['pai']])
        elif t in ('chi', 'pon', 'daiminkan'):
            assert last_dahai is not None and last_dahai != e['actor']
            e['target'] = last_dahai
            if t == 'chi': assert (last_dahai + 1) % 4 == e['actor']
            take(e['actor'], e['consumed'])
            if t == 'pon': pons[e['actor']][deaka(e['pai'])] = [e['pai']] + e['consumed']
        elif t in ('kakan', 'ankan'):
            a = e['actor']
            p = deaka(e['pai'] if 'pai' in e else e['consumed'][0])
            if p in pons[a]:
                # kakan (sometimes mislabelled as ankan)
                pt = next(x for x in hands[a] if deaka(x) == p)
                cons = pons[a].pop(p)
                take(a, [pt])
                e.clear(); e.update({'type': 'kakan', 'actor': a, 'pai': pt, 'consumed': cons})
            else:
                # ankan (sometimes mislabelled as kakan): consume all 4 copies actually in hand
                cons = [x for x in hands[a] if deaka(x) == p]
                assert len(cons) == 4, f'ankan {p} ev {e} pons {pons[a]} hand {hands[a]}'
                take(a, cons)
                e.clear(); e.update({'type': 'ankan', 'actor': a, 'consumed': cons})
        elif t == 'hora':
            e.setdefault('ura_markers', [])
    starts = [i for i, e in enumerate(evs) if e['type'] == 'start_kyoku']
    est = 0
    for k, s in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(evs)
        seg = evs[s:end]; start = seg[0]
        terms = [i for i, e in enumerate(seg) if e['type'] in ('hora', 'ryukyoku')]
        if not terms:
            continue
        if all('deltas' in seg[i] for i in terms):
            continue
        if k + 1 < len(starts):
            rv = [0, 0, 0, 0]
            for e in seg:
                if e['type'] == 'reach_accepted': rv[e['actor']] += 1000
            nxt = evs[starts[k + 1]]['scores']
            D = [nxt[p] - (start['scores'][p] - rv[p]) for p in range(4)]
        else:
            D = _final_deltas(seg, start); est = 1
        horas = [i for i in terms if seg[i]['type'] == 'hora']
        if len(horas) > 1 and k + 1 < len(starts):
            acc = [0, 0, 0, 0]
            for i in horas[:-1]:
                a, tg = seg[i]['actor'], seg[i]['target']
                dd = [0, 0, 0, 0]; dd[a] = D[a]; dd[tg] = -D[a]
                seg[i]['deltas'] = dd; acc = [x + y for x, y in zip(acc, dd)]
            seg[horas[-1]]['deltas'] = [x - y for x, y in zip(D, acc)]
        else:
            for j, i in enumerate(terms):
                seg[i]['deltas'] = D if j == 0 else [0, 0, 0, 0]
    return [json.dumps(e, separators=(',', ':'), ensure_ascii=False) for e in evs], est

def process(tarname):
    src = os.path.basename(tarname).replace('.tar', '')
    idx_path = f'{IDX}/{src}.tsv'
    if os.path.exists(idx_path + '.done'):
        return src, 'skip'
    os.makedirs(IDX, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    dctx = zstd.ZstdDecompressor()
    n = 0; stats = {}
    t0 = time.time()
    with tarfile.open(tarname, 'r|') as tf, open(idx_path, 'w') as idx:
        for m in tf:
            if not m.isfile() or not m.name.endswith('.zst'):
                continue
            raw = tf.extractfile(m).read()
            try:
                text = dctx.decompress(raw, max_output_size=50_000_000).decode('utf-8')
                lines = [l for l in text.split('\n') if l]
                if KEEP_HANCHAN < 1:
                    sk = [json.loads(l) for l in lines if l.startswith('{"type":"start_kyoku"')]
                    surely_h = any(e['bakaze'] in ('W', 'N') or (e['bakaze'] == 'S' and max(e['scores']) >= 30000) for e in sk)
                    if surely_h and random.random() >= KEEP_HANCHAN:
                        stats['skipped_hanchan'] = stats.get('skipped_hanchan', 0) + 1
                        continue
                lines = fix_haipai(lines)
                lines, est = repair(lines)
                stats['estimated_final'] = stats.get('estimated_final', 0) + est
                c = classify(lines)
            except Exception as ex:
                c = None
                k = 'err_' + type(ex).__name__ + ':' + str(ex)[:40]
                stats[k] = stats.get(k, 0) + 1
            if c is None:
                stats['bad'] = stats.get('bad', 0) + 1
                continue
            kind, nk, fr, sc = c
            if ONLY_KIND and kind != ONLY_KIND:
                stats['filtered_' + kind] = stats.get('filtered_' + kind, 0) + 1
                continue
            if n % 1000 == 0 and shutil.disk_usage(OUT).free < MIN_FREE:
                raise RuntimeError('Preprocessing stopped: disk reserve reached; original data retained')
            stats[kind] = stats.get(kind, 0) + 1
            if kind == 'tonpuu':
                lines = [l.replace('{"type":"start_kyoku",', '{"type":"start_kyoku","tonpuu":true,', 1)
                         if l.startswith('{"type":"start_kyoku"') else l for l in lines]
            shard = n // 5000
            d = f'{OUT}/{src}/{shard:04d}'
            if n % 5000 == 0:
                os.makedirs(d, exist_ok=True)
            name = os.path.basename(m.name).split('.')[0]
            p = f'{d}/{name}.json.gz'
            with gzip.open(p, 'wt', compresslevel=5) as f:
                f.write('\n'.join(lines) + '\n')
            idx.write(f'{p}\t{src}\t{kind}\t{nk}\t{fr}\t{",".join(map(str, sc))}\n')
            n += 1
            if n % 5000 == 0:
                print(time.strftime('%H:%M:%S'), src, 'written', n, 'elapsed', round(time.time()-t0), flush=True)
    import subprocess
    r = subprocess.run(['/root/gpufree-data/mj/tools/validate_logs', f'{OUT}/{src}'], capture_output=True, text=True,
                       env={**os.environ, 'LD_LIBRARY_PATH': '/root/gpufree-data/mj/env/py/lib', 'RAYON_NUM_THREADS': '2'})
    if r.returncode != 0:
        raise RuntimeError(f'Validator failed ({r.returncode}): {r.stderr[-2000:]}')
    bad = [l.split()[3] for l in r.stdout.splitlines() if l.startswith('error in log')]
    open(f'{IDX}/{src}.bad', 'w').write('\n'.join(bad) + '\n')
    open(f'{IDX}/{src}.valerr', 'w').write(r.stdout[-2000000:])
    stats['invalid'] = len(bad)
    stats['written'] = n
    stats['valid'] = n - len(bad)
    open(idx_path + '.done', 'w').write(json.dumps(stats))
    return src, stats, round(time.time() - t0)

if __name__ == '__main__':
    tars = sys.argv[1:]
    with mp.Pool(min(len(tars), int(os.environ.get('NPROC', 10)))) as pool:
        for r in pool.imap_unordered(process, tars):
            print(time.strftime('%H:%M:%S'), r, flush=True)