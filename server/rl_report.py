"""Summarise an RL run: eval history with paired comparisons, and training curve every 5 iters."""
import json, sys
d = sys.argv[1] if len(sys.argv) > 1 else '/root/gpufree-data/mj/ckpt/rl_hanchan'
h = json.load(open(f'{d}/hist.json'))
print('EVALS iter mean_pt se avg_rank rank_dist agari houjuu riichi fuuro | gain se_gain clear')
for e in h['evals']:
    c = e.get('paired_comparison') or {}
    s = e.get('style', {})
    print(e['iter'], round(e['mean_pt'], 2), round(e['se_pt'], 2), e['avg_rank'], e['rank_dist'],
          s.get('agari_rate'), s.get('houjuu_rate'), s.get('riichi_rate'), s.get('fuuro_rate'), '|',
          c.get('gain_pt') is not None and round(c['gain_pt'], 2), c.get('se_gain_pt') is not None and round(c['se_gain_pt'], 2),
          c.get('clear_improvement'))
print('best_iter', h['best_eval']['iter'], 'best_pt', round(h['best_pt'], 2), 'no_improve', h['no_improve'])
print('TRAIN iter opp gen_mean_pt kl_sl ent clipfrac v t_total')
for l in open(f'{d}/train.jsonl'):
    t = json.loads(l)
    if t['iter'] % 5 == 0 or t['iter'] == 1:
        print(t['iter'], t['opp'], t['gen_mean_pt'], t['kl_sl'], t['ent'], t['clipfrac'], t['v'], t['t_total'])
