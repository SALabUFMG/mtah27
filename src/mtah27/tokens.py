"""Janelas de tracking na grade de 5 Hz como tokens do FC-JEPA: [T * A, F] + máscara de padding.

A = 23 agentes por frame: bola (agente 0), até 11 slots do Barcelona/mandante (1-11) e até 11 do
visitante (12-22). Diferenças em relação ao FC-JEPA do laboratório:
- nada depende de quem ataca: sem espelhamento pelo lado de ataque e sem flag de time atacante.
  A orientação é a do campo; o treino sorteia a rotação de 180° (flip_tokens) para o modelo não
  aprender "ataca para a direita = H";
- a janela é contínua no jogo inteiro (bola morta e trocas de posse incluídas), não uma posse;
- bola ausente e slot vazio viram token mascarado (pad = True), como o FC-JEPA já faz com padding.

Os slots do ATD não seguem uma pessoa. Por janela, cada time fica com os 11 slots mais presentes,
ordenados pela posição média em x.
"""

from __future__ import annotations

import numpy as np

from mtah27.tracking import AWAY, HOME, Tracking

A = 23
N_PER_TEAM = 11
FEATURES = ("x", "y", "vx", "vy", "is_ball", "is_home", "is_away")
F = len(FEATURES)
V_MAX = 12.0  # m/s; acima disso é troca de tracklet, velocidade vira 0
V_SCALE = 10.0


def velocity(pos: np.ndarray, hz: float) -> np.ndarray:
    """Diferença para trás (m/s) por frame, mesmo shape de pos [N, ..., 2]; NaN onde falta um dos dois."""
    v = np.full_like(pos, np.nan)
    v[1:] = (pos[1:] - pos[:-1]) * hz
    v[0] = v[1] if len(pos) > 1 else 0.0
    fast = np.linalg.norm(v, axis=-1) > V_MAX
    v[fast] = 0.0
    return v


def _pick_slots(players: np.ndarray, team: np.ndarray, side: int) -> np.ndarray:
    """Até 11 slots do time `side` na janela [T, S, 2]: os mais presentes, ordenados por x médio."""
    slots = np.flatnonzero(team == side)
    count = (~np.isnan(players[:, slots, 0])).sum(axis=0)
    slots, count = slots[count > 0], count[count > 0]
    top = slots[np.lexsort((slots, -count))[:N_PER_TEAM]]  # desempate pelo índice: determinístico
    mean_x = np.nanmean(players[:, top, 0], axis=0)
    return top[np.lexsort((top, mean_x))]


def window_tokens(
    trk: Tracking, vel_ball: np.ndarray, vel_players: np.ndarray, start: int, T: int, pitch
) -> tuple[np.ndarray, np.ndarray]:
    """Tokens [T * A, F] float32 e pad [T * A] bool (True = escondido) da janela [start, start + T).

    trk já na grade (observe.subsample); vel_* de velocity() sobre o mesmo trk.
    """
    length, width = pitch
    sl = slice(start, start + T)
    players = trk.players[sl]
    pos = np.full((T, A, 2), np.nan, np.float32)
    vel = np.zeros((T, A, 2), np.float32)
    flags = np.zeros((A, 3), np.float32)
    pos[:, 0], vel[:, 0], flags[0, 0] = trk.ball[sl], vel_ball[sl], 1.0
    for side, first, col in ((HOME, 1, 1), (AWAY, 1 + N_PER_TEAM, 2)):
        chosen = _pick_slots(players, trk.team, side)
        k = len(chosen)
        pos[:, first : first + k] = players[:, chosen]
        vel[:, first : first + k] = vel_players[sl][:, chosen]
        flags[first : first + N_PER_TEAM, col] = 1.0
    pad = np.isnan(pos[..., 0])
    xy = (pos - np.array([length / 2, width / 2], np.float32)) / np.array([length / 2, width / 2], np.float32)
    vel = np.nan_to_num(vel) / V_SCALE
    tok = np.concatenate([np.nan_to_num(xy), vel, np.broadcast_to(flags, (T, A, 3))], axis=-1)
    tok[pad] = 0.0
    return tok.reshape(T * A, F).astype(np.float32), pad.reshape(T * A)


def flip_tokens(tok: np.ndarray) -> np.ndarray:
    """Rotação de 180° do campo (x, y, vx, vy trocam de sinal); funciona em numpy e torch."""
    out = tok.clone() if hasattr(tok, "clone") else tok.copy()
    out[..., :4] = -out[..., :4]
    return out
