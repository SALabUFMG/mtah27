# Copiado de github.com/SALabUFMG/fc-jepa (jepa/model.py, commit 334b3a3); só os imports mudaram.
# jepa/model.py

from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
import torch.nn.init as init
from torch.profiler import record_function

from mtah27.jepa.backbone_joint import JointBackbone
from mtah27.jepa.common import MLP
MAX_N_AGENTS = 23
from mtah27.jepa.masking import generate_mask_vjepa


class TransformerPredictor(nn.Module):
    """
    Transformer encoder predictor over [poss_token + agent_tokens].
    """

    def __init__(
        self,
        d_model: int,
        A: int = MAX_N_AGENTS,
        n_layers: int = 4,
        n_heads: int = 4,
        p: float = 0.1,
    ):
        super().__init__()
        self.A = A
        self.d_model = d_model

        self.poss_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=p, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        init.trunc_normal_(self.poss_token, std=0.02)

    def forward(
        self, H_agent_ctx: torch.Tensor, h_poss_ctx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        H_agent_ctx: [B, A, D]
        h_poss_ctx : [B, D]
        """
        B, A, D = H_agent_ctx.shape

        poss_token_bias = self.poss_token.expand(B, 1, D)
        poss_token_input = poss_token_bias + h_poss_ctx.unsqueeze(1)  # [B,1,D]
        sequence = torch.cat([poss_token_input, H_agent_ctx], dim=1)  # [B,A+1,D]

        encoded_sequence = self.transformer_encoder(sequence)  # [B,A+1,D]

        h_poss_pred = self.norm(encoded_sequence[:, 0, :])  # [B,D]
        H_agent_pred = self.norm(encoded_sequence[:, 1:, :])  # [B,A,D]
        return H_agent_pred, h_poss_pred


class FCJEPAModel(nn.Module):
    """
    V-JEPA-style pretraining over soccer possession tokens S=T*A.

    Inputs:
      x_full       : [B, S, F], where S = T * A
      key_pad_full : [B, S], True means PAD (ignored / hidden)
    """

    def __init__(
        self,
        in_dim_enc: int,
        d_model: int,
        depth: int,
        n_heads: int,
        same_agent_bias: bool,
        A: int = MAX_N_AGENTS,
    ):
        super().__init__()
        self.A = A
        self.d_model = d_model

        self.context_encoder = JointBackbone(
            in_dim=in_dim_enc,
            d_model=d_model,
            depth=depth,
            n_heads=n_heads,
            A=A,
            same_agent_bias=same_agent_bias,
        )

        self.target_encoder = JointBackbone(
            in_dim=in_dim_enc,
            d_model=d_model,
            depth=depth,
            n_heads=n_heads,
            A=A,
            same_agent_bias=same_agent_bias,
        )

        self.predictor = TransformerPredictor(d_model=d_model, A=A, n_layers=4, n_heads=n_heads)
        self.token_predictor = MLP(d_model, p=0.1)

        self._init_weights()

        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1.0)
                init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                init.trunc_normal_(m.weight, mean=0.0, std=0.02)

    def get_mask_modes(self):
        return Literal["time_span", "agent_slots", "mixed_block"]

    @staticmethod
    def _ensure_nonpad_mask_sane(mask_bta: torch.Tensor, pad_bta: torch.Tensor) -> torch.Tensor:
        """
        Restrict target mask to non-pad tokens and ensure:
          - at least one non-pad target token
          - at least one non-pad context token
        """
        device = mask_bta.device
        B, T, A = mask_bta.shape

        nonpad = ~pad_bta  # True where token is real
        mask_bta = mask_bta & nonpad

        # Per-sample corrections
        for i in range(B):
            nonpad_i = nonpad[i]  # [T,A]
            if not nonpad_i.any():
                # All padding: leave mask empty; caller should skip sample.
                continue

            # If no targets in nonpad region, pick one nonpad token to mask
            if not mask_bta[i].any():
                idx = torch.nonzero(nonpad_i, as_tuple=False)
                pick = idx[torch.randint(0, idx.size(0), (1,), device=device).item()]
                mask_bta[i, pick[0].item(), pick[1].item()] = True

            # If everything nonpad is masked, unmask one nonpad token
            if (mask_bta[i] == nonpad_i).all():
                idx = torch.nonzero(nonpad_i, as_tuple=False)
                pick = idx[torch.randint(0, idx.size(0), (1,), device=device).item()]
                mask_bta[i, pick[0].item(), pick[1].item()] = False

        return mask_bta

    def forward(
        self,
        x_full: torch.Tensor,
        key_pad_full: torch.Tensor,
        T: int,
        A: int,
        mask_mode: str,
        mask_kwargs: dict,
        use_checkpoint: bool = False,
    ):
        """
        x_full       : [B, S, F] with S = T * A
        key_pad_full : [B, S] (True = pad)

        Returns:
          total_loss, token_loss, agent_loss, poss_loss, H_token_pred, H_agent_pred, h_poss_pred
        """
        B, S, _F = x_full.shape
        device = x_full.device

        if S != T * A:
            raise ValueError(f"[JEPA] S != T*A (S={S}, T={T}, A={A})")

        real_pad_s = key_pad_full.bool()  # [B,S]
        pad_bta = real_pad_s.view(B, T, A)  # [B,T,A]

        # 1) Sample mask (True = target) then restrict to non-pad region
        with record_function("mask_generate+sanitize"):
            mask_bta = generate_mask_vjepa(
                T=T, A=A, batch_size=B, device=device, mask_mode=mask_mode, **mask_kwargs
            )
            mask_bta = self._ensure_nonpad_mask_sane(mask_bta, pad_bta)
            mask_sa = mask_bta.view(B, S)  # [B,S]

        # If any sample is fully padded, caller should skip; but keep safe behavior
        if real_pad_s.all(dim=1).any():
            # loss will be zero; training loop should skip these samples
            pass

        # Context sees ONLY context tokens: non-pad AND not masked
        key_pad_ctx = real_pad_s | mask_sa  # True = hidden from context

        # Target sees ONLY target tokens: non-pad AND masked
        key_pad_tgt = real_pad_s | (~mask_sa)

        # 2) Context encoder
        with record_function("context_encoder"):
            H_token_ctx, H_agent_ctx, h_poss_ctx = self.context_encoder(
                tokens=x_full,
                attn_mask_bool=None,
                key_pad_bool=key_pad_ctx,
                T=T,
                A=A,
                use_checkpoint=use_checkpoint,
            )

        # 3) Predictor
        with record_function("predictor_transformer"):
            H_agent_pred, h_poss_pred = self.predictor(H_agent_ctx, h_poss_ctx)
        with record_function("predictor_token_mlp"):
            H_token_pred = self.token_predictor(H_token_ctx)

        # 4) Target encoder (EMA)
        with record_function("target_encoder_ema"):

            with torch.no_grad():
                H_token_tgt, H_agent_tgt, h_poss_tgt = self.target_encoder(
                    tokens=x_full,
                    attn_mask_bool=None,
                    key_pad_bool=key_pad_tgt,
                    T=T,
                    A=A,
                    use_checkpoint=False,
                )

        # 5) Loss on TARGET region only (and non-pad)
        with record_function("loss_token_agent_poss"):
            token_mask = mask_bta & (~pad_bta)  # [B,T,A]

            if token_mask.any():
                H_token_pred_masked = H_token_pred[token_mask]  # [N,D]
                H_token_tgt_masked = H_token_tgt[token_mask]
                token_loss = torch_functional.l1_loss(
                    H_token_pred_masked, H_token_tgt_masked, reduction="mean"
                )
            else:
                token_loss = torch.zeros((), device=device)

            agent_has_target = token_mask.any(dim=1)  # [B,A]
            if agent_has_target.any():
                H_agent_pred_masked = H_agent_pred[agent_has_target]  # [N,D]
                H_agent_tgt_masked = H_agent_tgt[agent_has_target]
                agent_loss = torch_functional.l1_loss(
                    H_agent_pred_masked, H_agent_tgt_masked, reduction="mean"
                )
            else:
                agent_loss = torch.zeros((), device=device)

            poss_has_target = token_mask.any(dim=(1, 2))  # [B]
            if poss_has_target.any():
                h_poss_pred_masked = h_poss_pred[poss_has_target]
                h_poss_tgt_masked = h_poss_tgt[poss_has_target]
                poss_loss = torch_functional.l1_loss(
                    h_poss_pred_masked, h_poss_tgt_masked, reduction="mean"
                )
            else:
                poss_loss = torch.zeros((), device=device)

            total_loss = token_loss + agent_loss + poss_loss

        return (
            total_loss,
            token_loss,
            agent_loss,
            poss_loss,
            H_token_pred,
            H_agent_pred,
            h_poss_pred,
        )

    def update_target_encoder(self, momentum: float):
        with torch.no_grad():
            for param_q, param_k in zip(
                self.context_encoder.parameters(), self.target_encoder.parameters()
            ):
                param_k.data.mul_(momentum).add_(param_q.data, alpha=1.0 - momentum)
