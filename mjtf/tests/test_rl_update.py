"""PPO target/update and paired-selection regressions (CPU, no real checkpoint)."""
import sys, unittest, copy, tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl import build_batch, ppo_update, compare_evals, generate

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.head = torch.nn.Linear(2, 4); self.value = torch.nn.Linear(2, 1)
    def forward(self, obs, mask):
        x = obs.float()
        return dict(logits=self.head(x).masked_fill(~mask, -1e9), value=self.value(x).squeeze(-1))

class Tests(unittest.TestCase):
    def test_full_terminal_reward(self):
        class GR:
            def take_final_scores(self): return [40000, 30000, 20000, 10000]
            def take_rank_by_player(self): return [0, 1, 2, 3]
            def take_feature(self): return np.array([[0, 0, 0, 2.5, 2.5, 2.5, 2.5]])
        class Game:
            def take_obs(self): return [np.ones((3, 2))] * 3
            def take_actions(self): return [0, 1, 0]
            def take_masks(self): return [np.array([1,1], dtype=bool)] * 3
            def take_grp(self): return GR()
            def take_player_id(self): return 3
            def take_at_kyoku(self): return [0, 0, 0]
        class Loader:
            def __init__(self,*a,**kw): pass
            def load_gz_log_files(self, files): return [[Game()]]
        with patch('rl.glob.glob', return_value=['fixture']), patch('rl.GameplayLoader', Loader):
            batch, pt = build_batch('ignored', return_mode='terminal')
            self.assertEqual(pt, -195.) # -165 room, -15 place, -15 raw points
            torch.testing.assert_close(batch[3], torch.full((3,), -1.95))
            with self.assertRaises(MemoryError): build_batch('ignored',max_batch_bytes=1,return_mode='terminal')
    def test_tail_batch_and_finite_update(self):
        torch.manual_seed(10); policy=Tiny(); sl=copy.deepcopy(policy)
        before=copy.deepcopy(policy.state_dict())
        opt=torch.optim.AdamW(policy.parameters(),lr=1e-3)
        batch=(torch.randn(9,2),torch.tensor([[1,1,0,0]]*9,dtype=torch.bool),
               torch.tensor([0,1,0,1,0,1,0,1,0]),torch.linspace(-2,2,9))
        cfg=dict(mb=4,ppo_epochs=2,temp=1.,clip=.2,c_v=.5,beta_kl=.05,c_ent=.003)
        stats=ppo_update(policy,sl,opt,batch,cfg,'cpu')
        self.assertEqual(stats['updates'],6)
        self.assertEqual(stats['samples_seen'],18)
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in policy.state_dict().items()))
    def test_pairing_and_protocol(self):
        base=dict(kind='hanchan',champion='fixed',seed_start=100,seed_key=7,seed_count=3,seed_pt=[1.,2.,3.])
        newer={**base,'seed_pt':[2.,3.,4.]}
        r=compare_evals(newer,base)
        self.assertEqual(r['gain_pt'],1.)
        self.assertTrue(r['clear_improvement'])
        self.assertFalse(compare_evals(base,base)['clear_improvement'])
        with self.assertRaises(ValueError): compare_evals({**newer,'seed_key':8},base)
        with self.assertRaises(ValueError): compare_evals({**newer,'bf16':True},base)
    def test_kl_zero_when_policy_equals_sl(self):
        # SL log-probs are precomputed once; with identical nets the first-minibatch KL must be ~0.
        torch.manual_seed(3); policy=Tiny(); sl=copy.deepcopy(policy)
        opt=torch.optim.SGD(policy.parameters(),lr=0.)
        batch=(torch.randn(8,2),torch.tensor([[1,1,1,0]]*8,dtype=torch.bool),
               torch.tensor([0,1,2,0,1,2,0,1]),torch.linspace(-1,1,8))
        cfg=dict(mb=8,ppo_epochs=1,temp=1.,clip=.2,c_v=.5,beta_kl=.05,c_ent=.003)
        stats=ppo_update(policy,sl,opt,batch,cfg,'cpu')
        self.assertLess(abs(stats['kl_sl']),1e-6)
        self.assertEqual(stats['clipfrac'],0.)
        # Also at a non-unit temperature (regression: SL log-probs must use the same temp).
        stats=ppo_update(policy,sl,opt,batch,{**cfg,'temp':.5},'cpu')
        self.assertLess(abs(stats['kl_sl']),1e-6)
    def test_generation_preserves_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'keep'; f.write_text('safe')
            with self.assertRaises(ValueError): generate(None,None,d,1,1,'cpu',1.)
            self.assertEqual(f.read_text(),'safe')

if __name__=='__main__': unittest.main()
