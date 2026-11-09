# Copiado de github.com/SALabUFMG/fc-jepa (backbone/backbone_joint.py, commit 334b3a3); só os imports mudaram.
# backbone/backbone_joint.py

import torch
import torch.nn as nn
from torch.profiler import record_function
from torch.utils.checkpoint import checkpoint  # Import checkpoint utility

from mtah27.jepa.common import MLP, AgentSlotEmbedding, SinusoidalPE, apply_sdpa
from mtah27.jepa.pooling import PossessionPool, TemporalPool


class JointBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, A: int, same_agent_bias=False, bias_init=0.2, p=0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=True)
        self.Wk = nn.Linear(d_model, d_model, bias=True)
        self.Wv = nn.Linear(d_model, d_model, bias=True)
        self.Wo = nn.Linear(d_model, d_model, bias=True)
        self.mlp = MLP(d_model, p=p)
        self.n_heads = n_heads
        self.A = A
        self.same_agent_bias = same_agent_bias
        self.bias_strength = nn.Parameter(torch.tensor(float(bias_init)))

    def forward(self, x, attn_mask_bool, key_pad_bool, T, A):
        """
        x: [B, S, D], attn_mask_bool/key_pad_bool: [B,S,S]/[B,S]
        """
        B, S, D = x.shape
        with record_function("jointblock_norm_qkv"):
            h = self.norm1(x)
            q = self.Wq(h)
            k = self.Wk(h)
            v = self.Wv(h)  # [B,S,D]
            H = self.n_heads
            d_h = D // H
            q = q.view(B, S, H, d_h).transpose(1, 2)  # [B,H,S,d_h]
            k = k.view(B, S, H, d_h).transpose(1, 2)
            v = v.view(B, S, H, d_h).transpose(1, 2)

        # Masks (avoid materializing [B,H,S,S])
        with record_function("jointblock_mask_bias"):
            merged_mask = None
            if key_pad_bool is not None:
                key_mask = key_pad_bool[:, None, None, :]  # [B,1,1,S]
                merged_mask = key_mask if merged_mask is None else (merged_mask | key_mask)
            if attn_mask_bool is not None:
                if attn_mask_bool.dim() == 3:  # [B,S,S] -> [B,1,S,S]
                    attn_mask_bool = attn_mask_bool.unsqueeze(1)
                merged_mask = (
                    attn_mask_bool if merged_mask is None else (merged_mask | attn_mask_bool)
                )

            add_bias = None
            if self.same_agent_bias:
                add_bias = ("same_agent", int(A), self.bias_strength)

        with record_function("jointblock_sdpa"):
            out = apply_sdpa(q, k, v, merged_mask, add_bias)

        with record_function("jointblock_out_mlp"):
            out = out.transpose(1, 2).reshape(B, S, D)
            x = x + self.Wo(out)
            x = x + self.mlp(self.norm2(x))
        return x


class JointBackbone(nn.Module):
    """
    Joint space-time attention over S=T*A tokens.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        depth: int,
        n_heads: int,
        A: int = 23,
        time_pe_dim: int = 64,
        agent_emb_dim: int = 64,
        same_agent_bias: bool = False,
        p: float = 0.1,
        pool_temporal: str = "attn",
    ):
        super().__init__()
        self.A = A
        self.proj_in = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.time_pe = SinusoidalPE(time_pe_dim)
        self.agent_emb = AgentSlotEmbedding(A, agent_emb_dim)
        self.pe_proj = nn.Linear(time_pe_dim + agent_emb_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                JointBlock(d_model, n_heads, A, same_agent_bias=same_agent_bias, p=p)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.temp_pool = TemporalPool(d_model, mode=pool_temporal)
        self.poss_pool = PossessionPool(d_model)

    def forward(self, tokens, attn_mask_bool, key_pad_bool, T, A, use_checkpoint=False):
        """
        tokens: [B, S, F], S=T*A
        attn_mask_bool: optional; may be [B,S,S] or already broadcastable for SDPA.
        key_pad_bool: [B,S] (True=pad) — converted to [B,1,1,S] inside the block.
        Returns:
            H_token [B,T,A,D], H_agent [B,A,D], h_poss [B,D]
        """
        B, S, F = tokens.shape
        assert S == T * A == T * self.A
        D = self.proj_in[2].out_features
        with record_function("proj_in"):
            x = self.proj_in(tokens)  # [B,S,D]

        # Add time & agent encodings
        with record_function("positional_encoding"):
            pe_t, pe_a = self.time_pe(T, A, device=tokens.device)  # [T,dt], [A,da]
            pe = torch.cat(
                [
                    pe_t.repeat_interleave(A, dim=0),  # [T,dt] -> [S,dt]
                    pe_a.repeat(T, 1),  # [A,da] -> [S,da]
                ],
                dim=1,
            )  # [S, dt+da]
            pe = self.pe_proj(pe).unsqueeze(0).expand(B, -1, -1)  # [B,S,D]
            x = x + pe

        for blk in self.blocks:
            if use_checkpoint and x.requires_grad:
                x = checkpoint(blk, x, attn_mask_bool, key_pad_bool, T, A, use_reentrant=False)
            else:
                x = blk(x, attn_mask_bool, key_pad_bool, T=T, A=A)

        with record_function("norm_pool"):
            x = self.norm(x)  # [B,S,D]
            H_token = x.view(B, T, A, D)
            presence = (~key_pad_bool).view(B, T, A)
            H_agent = self.temp_pool(H_token, presence)  # [B,A,D]
            h_poss = self.poss_pool(H_token, presence)  # [B,D]
        return H_token, H_agent, h_poss
