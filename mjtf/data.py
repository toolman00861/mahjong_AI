import random, os, warnings
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from libriichi.dataset import GameplayLoader

# Mahjong Soul rank-point settings (Jade room, Master-1 4th penalty). See rules.md.
# rank pt = room uma + placement bonus (+15/+5/-5/-15) + (final_score - 25000)/1000
UMA = {
    'tonpuu': np.array([55., 30., 0., -80.]),
    'hanchan': np.array([110., 55., 0., -165.]),
}
PLACE_BONUS = np.array([15., 5., -5., -15.])

def mjs_pt(rank, final_score, kind):
    return UMA[kind][rank] + PLACE_BONUS[rank] + (final_score - 25000) / 1000.

class GameDataset(IterableDataset):
    def __init__(self, files, version=6, file_batch=24, shuffle=True, epochs=1,
                 augment=False, oracle=False, kinds=None, seed=0, buffer_mult=1, shuffle_buffer=0, max_samples_per_player=0):
        super().__init__()
        self.files = list(files)
        self.kinds = kinds or {}
        self.version = version
        self.file_batch = file_batch
        self.shuffle = shuffle
        self.epochs = epochs
        self.augment = augment
        self.oracle = oracle
        self.seed = seed
        self.max_samples_per_player = int(max_samples_per_player)
        if self.max_samples_per_player < 0:
            raise ValueError("max_samples_per_player must be nonnegative")
        self.shuffle_buffer = int(shuffle_buffer)
        if self.shuffle_buffer < 0:
            raise ValueError("shuffle_buffer must be nonnegative")

    def _iter_files(self):
        wi = get_worker_info()
        files = self.files
        if wi is not None:
            files = files[wi.id::wi.num_workers]
        rng = random.Random(self.seed + (wi.id if wi else 0) * 7919)
        for ep in range(self.epochs):
            fl = list(files)
            if self.shuffle:
                rng.shuffle(fl)
            yield ep, fl, rng

    def __iter__(self):
        for ep, fl, rng in self._iter_files():
            loaders = {False: GameplayLoader(self.version, oracle=self.oracle)}
            if self.augment:
                loaders[True] = GameplayLoader(self.version, oracle=self.oracle, augmented=True)
            pending = []
            for i in range(0, len(fl), self.file_batch):
                chunk = fl[i:i + self.file_batch]
                aug = self.augment and rng.random() < 0.5
                try:
                    games = loaders[aug].load_gz_log_files(chunk)
                    if len(games) != len(chunk):
                        raise ValueError("loader returned an unexpected number of files")
                    pairs = list(zip(chunk, games))
                except Exception:
                    # Preserve the filename association when an earlier file fails.
                    pairs = []
                    for f in chunk:
                        try:
                            one = loaders[aug].load_gz_log_files([f])
                            if len(one) != 1:
                                raise ValueError("loader returned an unexpected number of files")
                            pairs.append((f, one[0]))
                        except Exception as ex:
                            warnings.warn(f"Skipping invalid game {f}: {ex}")
                buf = []
                for f, file_games in pairs:
                    kind = self.kinds.get(f, 'hanchan')
                    tonpuu = 1 if kind == 'tonpuu' else 0
                    for g in file_games:
                        buf.extend(self._samples(g, kind, tonpuu, rng))
                # Rust observations are float32 and can be released before yielding.
                pairs.clear()
                games = None
                file_games = None
                g = None
                if self.shuffle:
                    rng.shuffle(buf)
                if not self.shuffle or not self.shuffle_buffer:
                    yield from buf
                    continue
                # Bounded streaming shuffle across file batches. Every sample is
                # emitted exactly once, including the final buffer drain.
                for item in buf:
                    if len(pending) < self.shuffle_buffer:
                        pending.append(item)
                    else:
                        j = rng.randrange(len(pending))
                        old = pending[j]
                        pending[j] = item
                        yield old
                buf.clear()
            if self.shuffle:
                rng.shuffle(pending)
            yield from pending

    def _samples(self, g, kind, tonpuu, rng=None):
        obs = g.take_obs()
        if not obs:
            return []
        inv = g.take_invisible_obs() if self.oracle else None
        actions = g.take_actions()
        masks = g.take_masks()
        at_kyoku = g.take_at_kyoku()
        grp = g.take_grp()
        pid = g.take_player_id()
        feat = grp.take_feature()           # (K, 7): grand_kyoku, honba, kyotaku, scores/1e4 x4
        final_scores = np.array(grp.take_final_scores(), dtype=np.float64)
        rank_by_player = grp.take_rank_by_player()
        my_rank = int(rank_by_player[pid])
        scores_seq = np.concatenate([feat[:, 3:] * 1e4, final_scores[None]], axis=0)
        deltas = (scores_seq[1:, pid] - scores_seq[:-1, pid]) / 1e4   # per-kyoku delta (in 10k)
        pt = mjs_pt(my_rank, final_scores[pid], kind) / 100.
        out = []
        indices = range(len(obs))
        if self.max_samples_per_player and len(obs) > self.max_samples_per_player:
            # The SL corpus is much larger than the step budget. Sample uniformly
            # within each player's game before allocating float16 observations,
            # instead of repeatedly learning one final outcome from hundreds of
            # strongly correlated decisions. Validation keeps every decision.
            rng = rng or random.Random(self.seed)
            indices = rng.sample(indices, self.max_samples_per_player)
        for i in indices:
            k = int(at_kyoku[i])
            item = (
                obs[i].astype(np.float16),
                np.asarray(masks[i], dtype=np.bool_),
                np.int64(actions[i]),
                np.int64(my_rank),
                np.float32(deltas[k] if k < len(deltas) else 0.),
                np.float32(pt),
                np.int64(tonpuu),
            )
            if self.oracle:
                item = item + (inv[i].astype(np.float16),)
            out.append(item)
        return out

def collate(batch):
    cols = list(zip(*batch))
    out = [torch.from_numpy(np.stack(c)) for c in cols]
    return out