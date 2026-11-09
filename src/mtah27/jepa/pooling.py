# Copiado de github.com/SALabUFMG/fc-jepa (backbone/pooling.py, commit 334b3a3); só os imports mudaram.
# backbone/pooling.py

import torch
import torch.nn as nn


class TemporalPool(nn.Module):
    """
    Per-agent temporal pooling over T: mean|max|attn.
    Input:  H_token [B, T, A, D]
    Output: H_agent [B, A, D]
    """

    def __init__(self, d_model: int, mode: str = "attn"):
        super().__init__()
        self.mode = mode
        if mode == "attn":
            self.query = nn.Parameter(torch.randn(1, 1, d_model))  # [1,1,D]
            self.proj = nn.Linear(d_model, d_model)

    def forward(self, H_token: torch.Tensor, presence: torch.Tensor):
        # presence: [B,T,A] (True where token exists); use as mask in pooling
        B, T, A, D = H_token.shape
        x = H_token  # [B,T,A,D]
        mask = ~presence  # True where pad
        if self.mode == "mean":
            x = x.masked_fill(mask[..., None], 0.0)
            denom = (~mask).sum(dim=1).clamp(min=1)[..., None]  # [B,A,1]
            return x.sum(dim=1) / denom  # [B,A,D]
        elif self.mode == "max":
            x = x.masked_fill(mask[..., None], float("-inf"))
            return x.max(dim=1).values  # [B,A,D]
        else:  # attn
            # Learned query per agent
            q = self.query.expand(B, A, -1)  # [B,A,D]
            k = self.proj(x)  # [B,T,A,D]
            attn = torch.einsum("bad,btad->bta", q, k) / (D**0.5)  # [B,T,A]
            # mask True -> invalid positions
            attn = attn.masked_fill(mask, float("-inf"))

            # Safe masked softmax over time dim. If an agent has no valid tokens
            # we force its attention weights to zero to avoid NaNs.
            # Compute per-agent max (for numerical stability)
            attn_max = attn.max(dim=1, keepdim=True).values  # [B,1,A]
            # Replace -inf (when all positions masked) with 0 to avoid NaNs
            attn_max = torch.where(torch.isinf(attn_max), torch.zeros_like(attn_max), attn_max)
            exp_attn = torch.exp(attn - attn_max)
            exp_attn = exp_attn.masked_fill(mask, 0.0)
            denom = exp_attn.sum(dim=1, keepdim=True).clamp_min(1.0)  # avoid div0
            w = (exp_attn / denom)[..., None]  # [B,T,A,1]

            H_agent = (w * x).sum(dim=1)  # [B,A,D]

            # Zero-out agents with no presence
            agent_is_present = presence.any(dim=1)  # [B,A]
            H_agent = H_agent.masked_fill(~agent_is_present[..., None], 0.0)
            return H_agent


class PossessionPool(nn.Module):
    """
    Attentive pooling over all tokens to get a possession embedding.
    Input:  H_token [B, T, A, D], presence [B,T,A]
    Output: h_poss [B, D]
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model))
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, H_token: torch.Tensor, presence: torch.Tensor):
        B, T, A, D = H_token.shape
        q = self.query.expand(B, -1)  # [B,D]
        k = self.proj(H_token)  # [B,T,A,D]
        logits = torch.einsum("bd,btad->bta", q, k) / (D**0.5)  # [B,T,A]
        mask = ~presence
        logits = logits.masked_fill(mask, float("-inf"))

        # Safe masked softmax over flattened (T*A). If all positions are masked
        # for a sample, we return a zero vector for the possession embedding.
        flat = logits.view(B, -1)  # [B, T*A]
        flat_max = flat.max(dim=1, keepdim=True).values  # [B,1]
        flat_max = torch.where(torch.isinf(flat_max), torch.zeros_like(flat_max), flat_max)
        exp_flat = torch.exp(flat - flat_max)
        # Zero-out masked positions
        exp_flat = exp_flat * (~mask).view(B, -1).float()
        denom = exp_flat.sum(dim=1, keepdim=True)
        # If denom==0 -> all masked -> return zeros
        zero_den = denom.squeeze(1) == 0
        denom = denom.clamp_min(1.0)
        w_flat = exp_flat / denom  # [B, T*A]
        w = w_flat.view(B, T, A, 1)
        h = (w * H_token).sum(dim=(1, 2))  # [B,D]
        # Zero-out any samples that had no valid tokens
        if zero_den.any():
            h = h.masked_fill(zero_den.unsqueeze(1), 0.0)
        return h
