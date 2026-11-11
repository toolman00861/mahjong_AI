import numpy as np
import torch

class TFEngine:
    """libriichi arena engine adapter for TileTransformer (version 6 obs)."""
    def __init__(self, model, device='cuda', name='tf', sample=False, temp=1.0, top_p=1.0,
                 enable_amp=True, record=None):
        self.engine_type = 'mortal'
        self.model = model.eval() if model is not None else None
        self.device = torch.device(device)
        self.name = name
        self.is_oracle = False
        self.version = 6
        self.enable_quick_eval = False
        self.enable_rule_based_agari_guard = True
        self.sample = sample
        self.temp = temp
        self.top_p = top_p
        self.enable_amp = enable_amp

    def react_batch(self, obs, masks, invisible_obs):
        try:
            with torch.inference_mode():
                return self._react(obs, masks)
        except Exception as ex:
            import traceback
            raise Exception(f'{ex}\n{traceback.format_exc()}')

    def _react(self, obs, masks):
        o = torch.as_tensor(np.stack(obs), device=self.device)
        m = torch.as_tensor(np.stack(masks), device=self.device)
        if next(self.model.parameters()).dtype == torch.bfloat16:
            # bf16 weights without autocast: enables the fused nn.Transformer inference fast path
            # (about 1.4-1.8x faster on L40; identical argmax on checks).
            out = self.model(o.to(torch.bfloat16), m)
        else:
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.enable_amp):
                out = self.model(o, m)
        logits = out['logits'].float()
        if self.sample:
            lg = (logits / self.temp).masked_fill(~m, -torch.inf)
            probs = lg.softmax(-1)
            if self.top_p < 1:
                ps, pi = probs.sort(-1, descending=True)
                cs = ps.cumsum(-1)
                ps[(cs - ps) > self.top_p] = 0
                act = pi.gather(-1, ps.multinomial(1)).squeeze(-1)
            else:
                act = probs.multinomial(1).squeeze(-1)
            greedy = act == logits.argmax(-1)
        else:
            act = logits.argmax(-1)
            greedy = torch.ones_like(act, dtype=torch.bool)
        q = logits.masked_fill(~m, -1e9)
        return act.tolist(), q.tolist(), m.tolist(), greedy.tolist()

class RuleEngine:
    """Random-legal-discard baseline: agari > riichi > pass > random discard; never calls."""
    PREF = None
    def __init__(self, name='rule', seed=0):
        self.engine_type = 'mortal'
        self.name = name
        self.is_oracle = False
        self.version = 6
        self.enable_quick_eval = False
        self.enable_rule_based_agari_guard = True
        self.rng = np.random.default_rng(seed)

    def react_batch(self, obs, masks, invisible_obs):
        acts = []
        for o, m in zip(obs, masks):
            m = np.asarray(m)
            if m[43]: a = 43
            elif m[37]: a = 37
            elif m[45]: a = 45
            else:
                cand = np.flatnonzero(m[:37])
                if len(cand):
                    a = int(self.rng.choice(cand))
                else:
                    a = int(np.flatnonzero(m)[0])
            acts.append(a)
        q = [[0.0] * 46 for _ in acts]
        return acts, q, [list(map(bool, mm)) for mm in masks], [True] * len(acts)