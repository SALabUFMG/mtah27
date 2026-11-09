# Copiado de github.com/SALabUFMG/fc-jepa (jepa/masking.py, commit 334b3a3); só os imports mudaram.
# jepa/masking.py

from typing import Literal

import torch


def generate_mask_vjepa(
    T: int,
    A: int,
    batch_size: int,
    device: torch.device,
    mask_mode: Literal["time_span", "agent_slots", "mixed_block"],
    time_mask_ratio: float = 0.3,
    agent_mask_ratio: float = 0.3,
    num_blocks: int = 4,
) -> torch.Tensor:
    """
    Generates a boolean mask [B, T, A] (True = MASKED/TARGET, False = VISIBLE/CONTEXT).
    """
    mask = torch.zeros((batch_size, T, A), dtype=torch.bool, device=device)

    if mask_mode == "time_span":
        mask_len = int(T * time_mask_ratio)
        if mask_len == 0:
            mask_len = 1

        max_start = T - mask_len
        start_t = torch.randint(0, max(1, max_start + 1), (batch_size,), device=device)

        for i in range(batch_size):
            end_t = min(T, start_t[i].item() + mask_len)
            mask[i, start_t[i].item() : end_t, :] = True

    elif mask_mode == "agent_slots":
        mask_count = int(A * agent_mask_ratio)
        if mask_count == 0:
            mask_count = 1

        for i in range(batch_size):
            masked_agents = torch.randperm(A, device=device)[:mask_count]
            mask[i, :, masked_agents] = True

    elif mask_mode == "mixed_block":
        mask_A_count = max(1, int(A * agent_mask_ratio))
        mask_T_len = max(1, int(T * (time_mask_ratio * 0.5)))

        for i in range(batch_size):
            for _ in range(max(1, num_blocks)):
                if mask_A_count >= A:
                    start_a = 0
                    end_a = A
                else:
                    max_start_a = A - mask_A_count
                    start_a = torch.randint(0, max_start_a + 1, (1,), device=device).item()
                    end_a = start_a + mask_A_count

                if mask_T_len >= T:
                    start_t = 0
                    end_t = T
                else:
                    max_start_t = T - mask_T_len
                    start_t = torch.randint(0, max_start_t + 1, (1,), device=device).item()
                    end_t = start_t + mask_T_len

                mask[i, start_t:end_t, start_a:end_a] = True

    # Safety: avoid all/none masking per sample (generic, not pad-aware)
    Btot = mask.shape[0]
    for i in range(Btot):
        if not mask[i].any():
            t = torch.randint(0, T, (1,), device=device).item()
            a = torch.randint(0, A, (1,), device=device).item()
            mask[i, t, a] = True
        elif mask[i].all():
            t = torch.randint(0, T, (1,), device=device).item()
            a = torch.randint(0, A, (1,), device=device).item()
            mask[i, t, a] = False

    return mask
