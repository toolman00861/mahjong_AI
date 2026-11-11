"""Small deterministic regressions; no GPU or dataset required."""
import sys, unittest, warnings
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import GameDataset, mjs_pt
from grpnet import featurize, GRPNet

class Loader:
    def __init__(self,*args,**kwargs): pass
    def load_gz_log_files(self, files):
        if 'bad' in files: raise ValueError('broken fixture')
        return [[f] for f in files]

class Samples(GameDataset):
    def _samples(self,g,kind,tonpuu,rng=None):
        return [(g,kind,i) for i in range(4)]

class RegressionTests(unittest.TestCase):
    def test_shuffle_preserves_all_samples(self):
        files=[str(i) for i in range(30)]
        with patch('data.GameplayLoader',Loader):
            plain=list(Samples(files,file_batch=3,shuffle=False))
            mixed=list(Samples(files,file_batch=3,shuffle_buffer=17,seed=42))
            again=list(Samples(files,file_batch=3,shuffle_buffer=17,seed=42))
        self.assertCountEqual(plain,mixed)
        self.assertEqual(mixed,again)
        self.assertNotEqual(plain,mixed)
    def test_fallback_preserves_filename_kind(self):
        with patch('data.GameplayLoader',Loader),warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rows=list(Samples(['bad','east','south'],file_batch=3,shuffle=False,kinds={'east':'tonpuu','south':'hanchan'}))
        self.assertEqual({(x[0],x[1]) for x in rows},{('east','tonpuu'),('south','hanchan')})
    def test_worker_partition(self):
        files=list(range(13)); seen=[]
        for worker in range(3):
            wi=type('Worker',(),{'id':worker,'num_workers':3})()
            with patch('data.get_worker_info',return_value=wi):
                seen.extend(next(Samples(files)._iter_files())[1])
        self.assertCountEqual(seen,files)
    def test_per_game_cap(self):
        class Grp:
            def take_feature(self): return np.array([[0,0,0,2.5,2.5,2.5,2.5]])
            def take_final_scores(self): return [30000,27000,23000,20000]
            def take_rank_by_player(self): return [0,1,2,3]
        class Game:
            def take_obs(self): return [np.full((2,3),i) for i in range(20)]
            def take_actions(self): return list(range(20))
            def take_masks(self): return [np.ones(46,dtype=bool)]*20
            def take_at_kyoku(self): return [0]*20
            def take_grp(self): return Grp()
            def take_player_id(self): return 0
        ds=GameDataset([],max_samples_per_player=5,seed=42)
        rows=ds._samples(Game(),'hanchan',0)
        self.assertEqual(len(rows),5)
        self.assertEqual(len(set(int(x[2]) for x in rows)),5)
        self.assertEqual([int(x[2]) for x in rows],[int(x[2]) for x in ds._samples(Game(),'hanchan',0)])
        self.assertEqual(len(GameDataset([])._samples(Game(),'hanchan',0)),20)

    def test_hanchan_reward(self):
        self.assertEqual(mjs_pt(3,25000,'hanchan'),-180.)
        self.assertEqual(sum(mjs_pt(i,25000,'hanchan') for i in range(4)),0.)
    def test_west_round_encoding(self):
        feat=np.array([[r,0,0,2.5,2.5,2.5,2.5] for r in range(12)],dtype=float)
        x=featurize(feat,2)
        self.assertEqual(x.shape,(12,24))
        np.testing.assert_array_equal(x[:,:12],np.eye(12))
        self.assertEqual(GRPNet().net[0].in_features,24)
        with self.assertRaises(ValueError):
            featurize(np.array([[12,0,0,2.5,2.5,2.5,2.5]]),0)

if __name__=='__main__': unittest.main()
