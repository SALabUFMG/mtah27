# Copiado de github.com/SALabUFMG/fc-jepa (backbone/common.py, commit 334b3a3); só os imports mudaram.
# backbone/common.py

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPE(nn.Module):
    """
    Classic transformer PE. We build separate PEs for time (T) and agent slots (A),
    then sum them after broadcasting to tokens.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0, "PE dim must be even"

    def _pe_1d(self, n: int, device):
        d = self.dim
        pos = torch.arange(n, device=device).float().unsqueeze(1)  # [n,1]
        i = torch.arange(0, d, 2, device=device).float()  # [d/2]
        div = torch.exp(-i * math.log(10000.0) / d)
        pe = torch.zeros(n, d, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe  # [n, d]

    def forward(self, T: int, A: int, device=None):
        pe_t = self._pe_1d(T, device)  # [T, d]
        pe_a = self._pe_1d(A, device)  # [A, d]
        return pe_t, pe_a


# ---- Agent-slot embeddings ----


class AgentSlotEmbedding(nn.Module):
    """
    Learnable embeddings per agent slot id (0..A-1). No team IDs.
    """

    def __init__(self, A: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(A, dim)

    def forward(self, A: int, device=None):
        idx = torch.arange(A, device=device)
        return self.emb(idx)  # [A, dim]


def build_same_agent_bias(S: int, A: int, device, bias_value: float) -> torch.Tensor:
    """
    Legacy helper (O(S^2)): Return [S,S] float bias matrix: +bias where tokens share the same agent slot.
    Kept for backward compatibility, but DO NOT USE in training paths if you care about OOM.
    """
    idx = torch.arange(S, device=device)
    a_i = (idx % A).unsqueeze(1)  # [S,1]
    a_j = (idx % A).unsqueeze(0)  # [1,S]
    same = (a_i == a_j).float()
    return same * bias_value  # [S,S]


# -----------------------------------------------------------------------------
# Efficient same-agent bias implementation
# -----------------------------------------------------------------------------
_ONE_HOT_CACHE: Dict[Tuple[str, str, int, int, int], torch.Tensor] = {}
# key: (device_str, dtype_str, length, A, A_pad) -> [1,1,length,A_pad]


def _round_up_to_multiple(x: int, m: int) -> int:
    if m <= 1:
        return x
    return ((x + m - 1) // m) * m


def _get_cached_one_hot(length: int, A: int, A_pad: int, device, dtype) -> torch.Tensor:
    """
    Returns [1,1,length,A_pad] float tensor where row i is one-hot(agent_id=i%A) in dim A_pad.
    Caches per (device,dtype,length,A,A_pad). This avoids repeatedly rebuilding one-hots each layer/step.
    """
    device_str = str(device)
    dtype_str = str(dtype)
    key = (device_str, dtype_str, int(length), int(A), int(A_pad))

    t = _ONE_HOT_CACHE.get(key, None)
    if t is not None:
        return t

    # Simple cache control to avoid unbounded growth
    if len(_ONE_HOT_CACHE) > 64:
        _ONE_HOT_CACHE.clear()

    # Build one-hot via eye lookup (fast, deterministic)
    # agent_ids: [length]
    agent_ids = torch.arange(length, device=device) % int(A)
    eye = torch.eye(int(A_pad), device=device, dtype=dtype)
    one_hot = eye.index_select(0, agent_ids)  # [length, A_pad]
    one_hot = one_hot.unsqueeze(0).unsqueeze(0).contiguous()  # [1,1,length,A_pad]

    _ONE_HOT_CACHE[key] = one_hot
    return one_hot


def _bias_to_alpha_and_sign(
    bias_value: Union[float, torch.Tensor], d0: int, device, dtype
) -> Tuple[Union[float, torch.Tensor], float]:
    """
    We want to add a constant bias 'b' to attention logits for same-agent pairs.

    In the fast SDPA path we do it by augmenting q/k:
      logits = (q_aug @ k_aug^T) * (1/sqrt(d0))
    and choose alpha such that:
      (alpha^2 / sqrt(d0)) == |b|   -> alpha = sqrt(|b| * sqrt(d0))

    Returns:
      alpha: float or scalar tensor (keeps grad if bias_value is a tensor param)
      sign:  +1.0 or -1.0 (detached; if bias_value is tensor and flips sign, sign gradient is undefined anyway)
    """
    if torch.is_tensor(bias_value):
        # Keep gradient through magnitude; sign is treated as constant (detached)
        sign = 1.0 if float(bias_value.detach().item()) >= 0.0 else -1.0
        bias_abs = torch.abs(bias_value).to(device=device, dtype=dtype)
        alpha = torch.sqrt(bias_abs * math.sqrt(float(d0)))
        return alpha, sign

    # float path
    b = float(bias_value)
    if b == 0.0:
        return 0.0, 1.0
    sign = 1.0 if b >= 0.0 else -1.0
    alpha = math.sqrt(abs(b) * math.sqrt(float(d0)))
    return alpha, sign


def apply_sdpa(q, k, v, attn_mask_bool: Optional[torch.Tensor], add_bias):
    """
    q: [B, H, S_q, d_h]
    k,v: [B, H, S_k, d_h]
    attn_mask_bool: [B, S_q, S_k] or [B,1,S_q,S_k] or [B,H,S_q,S_k] or
                    key-padding broadcast [B,1,1,S_k] (True=BLOCK) or None
    add_bias:
      - None
      - ("same_agent", A, bias_value)   # efficient, no [S,S]
      - (legacy) Tensor [S,S] or [B,S,S] or [B,1,S,S]/[B,H,S,S]
    """
    B, H, S_q, d0 = q.shape
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"k/v must be 4D [B,H,S_k,d], got k={k.dim()} v={v.dim()}")
    if k.shape[:2] != (B, H) or v.shape[:2] != (B, H):
        raise ValueError(
            f"k/v batch/head mismatch: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    S_k = k.shape[2]
    if v.shape[2] != S_k:
        raise ValueError(f"v must have same S_k as k. got k={S_k}, v={v.shape[2]}")

    # Normalize bool mask shape to [B,1,*,S_k] (broadcast over heads) if present
    if attn_mask_bool is not None:
        if attn_mask_bool.dim() == 3:
            attn_mask_bool = attn_mask_bool.unsqueeze(1)  # [B,1,L,S_k]
        elif attn_mask_bool.dim() == 4:
            pass
        else:
            raise ValueError(f"attn_mask_bool must have dim 3 or 4, got {attn_mask_bool.dim()}")

        if attn_mask_bool.size(-1) != S_k:
            raise ValueError(
                f"attn_mask_bool last dim must be S_k={S_k}, got {attn_mask_bool.size(-1)}"
            )

        L = attn_mask_bool.size(-2)  # query length

        # L==1 is broadcastable key padding mask: DO NOT expand it.
        if L == 1:
            pass
        elif L == S_q:
            pass
        else:
            raise ValueError(
                f"attn_mask_bool has query length L={L}, but q length is S_q={S_q}. "
                "Fix mask construction upstream (make L==1 or L==S_q)."
            )

    def _needs_chunked(mask: Optional[torch.Tensor], S_q: int, S_k: int) -> bool:
        if mask is None:
            return False
        if mask.dim() != 4:
            return True
        L = mask.size(-2)
        if L == 1:
            return False
        # Only chunk for genuinely large per-query masks
        return S_q >= 2048 or S_k >= 2048

    # -------------------------------------------------------------------------
    # Fast SDPA path
    # -------------------------------------------------------------------------
    if not _needs_chunked(attn_mask_bool, S_q, S_k):
        if add_bias is None:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask_bool, is_causal=False
            )

        if isinstance(add_bias, tuple) and len(add_bias) == 3 and add_bias[0] == "same_agent":
            _, A, bias_value = add_bias
            if A is None:
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask_bool, is_causal=False
                )

            # If bias is effectively zero
            if (torch.is_tensor(bias_value) and float(bias_value.detach().item()) == 0.0) or (
                (not torch.is_tensor(bias_value)) and float(bias_value) == 0.0
            ):
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask_bool, is_causal=False
                )

            # -----------------------------------------------------------------
            # SAME-AGENT BIAS via q/k augmentation, but with A padded to multiple of 8
            # to keep head dim aligned (helps flash/mem-efficient kernels).
            #
            # We preserve original scaling based on d0 by setting SDPA scale=1/sqrt(d0).
            # -----------------------------------------------------------------
            scale = 1.0 / math.sqrt(float(d0))

            A_int = int(A)
            # pad to 8 to avoid awkward head dims (e.g., 64+23=87); 64+24=88 is 8-aligned
            A_pad = _round_up_to_multiple(A_int, 8)

            alpha, sign = _bias_to_alpha_and_sign(bias_value, d0=d0, device=q.device, dtype=q.dtype)

            one_hot_q = _get_cached_one_hot(S_q, A_int, A_pad, device=q.device, dtype=q.dtype)
            one_hot_k = _get_cached_one_hot(S_k, A_int, A_pad, device=q.device, dtype=q.dtype)
            # [1,1,S_q,A_pad], [1,1,S_k,A_pad]

            # Expand without allocating per-(B,H) copies; concatenation allocates final q_aug/k_aug
            q_extra = (alpha * one_hot_q).expand(B, H, S_q, A_pad)
            k_extra = (sign * alpha * one_hot_k).expand(B, H, S_k, A_pad)

            q_aug = torch.cat([q, q_extra], dim=-1)
            k_aug = torch.cat([k, k_extra], dim=-1)

            return F.scaled_dot_product_attention(
                q_aug, k_aug, v, attn_mask=attn_mask_bool, is_causal=False, scale=scale
            )

        # Legacy tensor bias handling (discouraged with per-example masks)
        if torch.is_tensor(add_bias):
            if attn_mask_bool is None and add_bias.dim() == 2:
                return F.scaled_dot_product_attention(q, k, v, attn_mask=add_bias, is_causal=False)
            raise RuntimeError(
                "apply_sdpa: tensor add_bias with per-example bool mask can allocate huge buffers. "
                "Use add_bias=('same_agent', A, bias_value) or rely on chunked fallback."
            )

        raise TypeError(f"apply_sdpa: unsupported add_bias type: {type(add_bias)}")

    # -------------------------------------------------------------------------
    # Chunked attention fallback (mask-friendly, avoids [B,H,S,S])
    # -------------------------------------------------------------------------
    q_dtype = q.dtype
    device = q.device

    # Stable softmax: do matmul in fp32
    qf = q.float()
    kf = k.float()
    vf = v.float()

    scale = 1.0 / math.sqrt(float(d0))

    # Precompute agent ids for same-agent bias if requested
    same_agent = False
    bias_value_f = 0.0
    A_bias = None
    if isinstance(add_bias, tuple) and len(add_bias) == 3 and add_bias[0] == "same_agent":
        _, A_bias, bias_value = add_bias
        if A_bias is not None:
            A_bias = int(A_bias)
            if torch.is_tensor(bias_value):
                bias_value_f = float(bias_value.detach().item())
            else:
                bias_value_f = float(bias_value)
            same_agent = bias_value_f != 0.0

            if same_agent:
                agent_k = torch.arange(S_k, device=device) % A_bias  # [S_k]

    blk = 256
    out = torch.empty((B, H, S_q, d0), device=device, dtype=q_dtype)

    for qs in range(0, S_q, blk):
        qe = min(S_q, qs + blk)
        q_blk = qf[:, :, qs:qe, :]  # [B,H,blk,d]
        scores = torch.matmul(q_blk, kf.transpose(-2, -1)) * scale  # [B,H,blk,S_k] in fp32

        if same_agent:
            agent_q = torch.arange(qs, qe, device=device) % A_bias
            same = (agent_q[:, None] == agent_k[None, :]).to(dtype=scores.dtype)
            scores = scores + (bias_value_f * same)[None, None, :, :]

        if torch.is_tensor(add_bias):
            if add_bias.dim() == 2:
                scores = scores + add_bias[qs:qe, :].to(scores.dtype)[None, None, :, :]
            elif add_bias.dim() == 3:
                scores = scores + add_bias[:, qs:qe, :].to(scores.dtype)[:, None, :, :]
            elif add_bias.dim() == 4:
                scores = scores + add_bias[:, :, qs:qe, :].to(scores.dtype)
            else:
                raise ValueError(f"add_bias tensor dim must be 2/3/4, got {add_bias.dim()}")

        if attn_mask_bool is not None:
            m_slice = attn_mask_bool[:, :, qs:qe, :]  # [B,1,blk,S_k] or [B,H,blk,S_k]
            all_masked = m_slice.all(dim=-1, keepdim=True)
            scores = scores.masked_fill(m_slice, float("-inf"))
            scores = torch.where(all_masked, torch.zeros_like(scores), scores)

        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q_dtype)  # [B,H,blk,S_k]
        out[:, :, qs:qe, :] = torch.matmul(attn, vf)  # matmul in fp32->bf16 path is ok

    return out


class MLP(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mult * d_model),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(mult * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)
