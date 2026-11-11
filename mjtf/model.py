import math
import torch
from torch import nn
import torch.nn.functional as F

AKA_BASE = (4, 13, 22)  # 5m 5p 5s in 34-tile index
ACTION_SPACE = 46

class TileTransformer(nn.Module):
    """Tile-as-token Transformer.
    Input: libriichi obs (C, 34) -> 34 tile tokens (+1 CLS, +1 oracle token optional).
    Output: 46 action logits, aux heads.
    """
    def __init__(self, in_ch=1014, d_model=512, n_layers=12, n_heads=8, ff_mult=4,
                 dropout=0.0, oracle_ch=0):
        super().__init__()
        self.in_ch = in_ch
        self.d_model = d_model
        self.in_proj = nn.Sequential(nn.Linear(in_ch, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.tile_emb = nn.Parameter(torch.randn(34, d_model) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.oracle_ch = oracle_ch
        if oracle_ch:
            self.oracle_proj = nn.Linear(oracle_ch, d_model)
            self.oracle_gate = 1.0
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * ff_mult, dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.tile_head = nn.Linear(d_model, 2)     # [discard/kan logit, aka-discard logit]
        self.cls_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 9))
        self.rank_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 4))
        self.delta_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, obs, mask=None, oracle=None):
        # obs: (B, C, 34)
        x = obs.transpose(1, 2).to(self.in_proj[0].weight.dtype)  # (B, 34, C)
        x = self.in_proj(x) + self.tile_emb
        if self.oracle_ch and oracle is not None:
            x = x + self.oracle_gate * self.oracle_proj(oracle.transpose(1, 2).to(x.dtype))
        cls = self.cls.expand(x.shape[0], -1, -1)
        h = self.encoder(torch.cat([cls, x], dim=1))
        h = self.norm(h)
        hc, ht = h[:, 0], h[:, 1:]
        tl = self.tile_head(ht)                    # (B, 34, 2)
        logits = torch.cat([tl[..., 0], tl[:, AKA_BASE, 1], self.cls_head(hc)], dim=-1)  # (B, 46)
        logits = logits.float()
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return {
            'logits': logits,
            'rank': self.rank_head(hc).float(),
            'delta': self.delta_head(hc).squeeze(-1).float(),
            'value': self.value_head(hc).squeeze(-1).float(),
        }

def param_count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
