import gzip
import json
import sys
import tempfile
from pathlib import Path

import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import libriichi  # require the real project dependency; no sys.modules stubs
    import eval as eval_module
except ImportError as exc:
    raise unittest.SkipTest(f"libriichi dependencies unavailable: {exc}")


def write_log(path, seed, key, seat, scores=(25000, 25000, 25000, 25000), length=4, deltas=(0, 0, 0, 0), kyotaku=0, complete=True):
    names = ["champion"] * 4
    names[seat] = "challenger"
    events = [{"type": "start_game", "names": names, "seed": [seed, key]}]
    for _ in range(length):
        events.append({"type": "start_kyoku", "scores": list(scores), "kyotaku": kyotaku})
        events.append({"type": "ryukyoku", "deltas": list(deltas)})
        events.append({"type": "end_kyoku"})
    if complete:
        events.append({"type": "end_game"})
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(e) for e in events) + "\n")


class EvalMetricsTests(unittest.TestCase):
    def test_four_rotations_metrics_and_single_seed_se(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seat in range(4):
                write_log(root / f"100_9_{seat}.json.gz", 100, 9, seat)
            out = eval_module.summarize_logs(root, "challenger", 100, 9, 1, 4, [1, 1, 1, 1])
            self.assertEqual(out["games"], 4)
            self.assertEqual(out["ranks"], [1, 1, 1, 1])
            self.assertEqual((out["seed_start"], out["seed_key"], out["seed_count"]), (100, 9, 1))
            self.assertEqual(out["seed_pt"], [1.25])  # tonpuu room table sums to +5
            self.assertEqual(out["mean_pt"], 1.25)
            self.assertIsNone(out["se_pt"])
            self.assertIsNone(out["se_rank"])

    def test_score_bonus_and_seed_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Every seed is a cluster of four seats, not four IID observations.
            for seed in (21, 20):
                for seat in range(4):
                    scores = [20000 if seed == 20 else 30000] * 4
                    scores[seat] = 40000 if seed == 20 else 10000
                    write_log(root / f"{seed}_7_{seat}.json.gz", seed, 7, seat, scores=scores)
            out = eval_module.summarize_logs(root, "challenger", 20, 7, 2, 8, [4, 0, 0, 4])
            self.assertEqual(out["seed_pt"], [140., -195.])
            self.assertEqual(out["mean_pt"], -27.5)
            self.assertAlmostEqual(out["se_pt"], 167.5)
            self.assertAlmostEqual(out["se_rank"], 1.5)
            with self.assertRaisesRegex(ValueError, 'rank mismatch'):
                eval_module.summarize_logs(root, "challenger", 20, 7, 2, 8, [2, 2, 2, 2])
            with self.assertRaisesRegex(ValueError, 'seed/key'):
                eval_module.summarize_logs(root, "challenger", 20, 8, 2, 8, [4, 0, 0, 4])

    def test_missing_logs_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seat in range(3):
                write_log(root / f"100_9_{seat}.json.gz", 100, 9, seat)
            with self.assertRaisesRegex(ValueError, "expected 4 game logs"):
                eval_module.summarize_logs(root, "challenger", 100, 9, 1, 4, [1, 1, 1, 0])

    def test_unclaimed_kyotaku_to_highest_tied_score_first_seat(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "100_9_1.json.gz"
            write_log(path, 100, 9, 1, scores=(40000, 40000, 10000, 9000), length=1, kyotaku=1)
            game = eval_module._read_game(path, "challenger")
            self.assertEqual(game[3], [41000, 40000, 10000, 9000])
            self.assertEqual(game[4], 1)

    def test_early_bust_and_incomplete_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seat in range(4):
                write_log(root / f"100_9_{seat}.json.gz", 100, 9, seat, length=1)
            out = eval_module.summarize_logs(root, "challenger", 100, 9, 1, 8, [1, 1, 1, 1])
            self.assertEqual(out["games"], 4)
            bad = root / "bad"
            bad.mkdir()
            for seat in range(4):
                write_log(bad / f"100_9_{seat}.json.gz", 100, 9, seat, length=1, complete=False)
            with self.assertRaisesRegex(ValueError, "end_game"):
                eval_module.summarize_logs(bad, "challenger", 100, 9, 1, 8, [1, 1, 1, 1])

    def test_play_style_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'style.json.gz'
            S = [25000] * 4
            ev = [{"type": "start_game", "names": ["challenger", "champion", "champion", "champion"], "seed": [1, 2]},
                  # kyoku 1: challenger calls pon twice, riichi-less tsumo
                  {"type": "start_kyoku", "scores": S, "kyotaku": 0},
                  {"type": "pon", "actor": 0, "target": 1}, {"type": "pon", "actor": 0, "target": 2},
                  {"type": "hora", "actor": 0, "target": 0, "deltas": [3000, -1000, -1000, -1000]},
                  {"type": "end_kyoku"},
                  # kyoku 2: challenger riichi, deals into a double ron
                  {"type": "start_kyoku", "scores": [28000, 24000, 24000, 24000], "kyotaku": 0},
                  {"type": "reach_accepted", "actor": 0},
                  {"type": "hora", "actor": 1, "target": 0, "deltas": [-2000, 2000, 0, 0]},
                  {"type": "hora", "actor": 2, "target": 0, "deltas": [-2000, 0, 2000, 0]},
                  {"type": "end_kyoku"},
                  # kyoku 3: others' call and ron, challenger uninvolved
                  {"type": "start_kyoku", "scores": [23000, 26000, 26000, 24000], "kyotaku": 0},
                  {"type": "chi", "actor": 2, "target": 1},
                  {"type": "hora", "actor": 2, "target": 1, "deltas": [0, -1000, 1000, 0]},
                  {"type": "end_kyoku"}, {"type": "end_game"}]
            with gzip.open(path, 'wt') as f: f.write('\n'.join(map(json.dumps, ev)))
            style = eval_module._read_game(path, 'challenger')[6]
            self.assertEqual(style, dict(kyoku=3, agari=1, tsumo=1, houjuu=1, riichi=1, fuuro=1))

    def test_terminal_kyoku_requires_its_own_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'missing_result.json.gz'
            write_log(path, 1, 2, 0, length=2)
            with gzip.open(path, 'rt') as f: events = [json.loads(l) for l in f]
            del events[-3]  # remove only the final ryukyoku, retain end_kyoku/end_game
            with gzip.open(path, 'wt') as f: f.write('\n'.join(map(json.dumps, events)))
            with self.assertRaisesRegex(ValueError, 'without result'):
                eval_module._read_game(path, 'challenger')


if __name__ == "__main__":
    unittest.main()
