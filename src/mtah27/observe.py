"""Observação por frame para o HMM de posse: quem está perto da bola, ou fora/sem dado."""

from __future__ import annotations

import numpy as np

from mtah27.tracking import AWAY, Tracking

OBS = ("home_close", "away_close", "home_far", "away_far", "out", "missing")
_IDX = {name: i for i, name in enumerate(OBS)}


def observe(
    trk: Tracking, r_close_m: float, pitch: tuple[float, float], out_margin_m: float = 1.0
) -> np.ndarray:
    """Código int8 (índice em OBS) por frame.

    missing: bola NaN ou nenhum jogador presente; out: bola além de out_margin_m do campo;
    senão time do jogador mais próximo, close se a distância < r_close_m.
    """
    length, width = pitch
    ball = trk.ball
    d = np.hypot(trk.players[..., 0] - ball[:, None, 0], trk.players[..., 1] - ball[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)  # slot vazio nunca é o mais próximo
    nearest = d.argmin(axis=1)
    dmin = np.take_along_axis(d, nearest[:, None], axis=1)[:, 0]

    missing = np.isnan(ball).any(axis=1) | np.isinf(dmin)
    bx, by = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        out = (
            (bx < -out_margin_m)
            | (bx > length + out_margin_m)
            | (by < -out_margin_m)
            | (by > width + out_margin_m)
        )

    away = trk.team[nearest] == AWAY
    close = dmin < r_close_m
    code = np.where(close, 0, 2) + away  # home_close=0, away_close=1, home_far=2, away_far=3
    code = np.where(out, _IDX["out"], code)
    code = np.where(missing, _IDX["missing"], code)
    return code.astype(np.int8)


def degrade(trk: Tracking, missing_rate: float, mean_gap_s: float, fps: int, seed: int) -> Tracking:
    """Cópia de trk com a bola apagada em buracos contíguos (comprimento geométrico) até a fração
    de frames sem bola (contando os já ausentes) chegar a missing_rate. Determinístico por seed.

    A bola é copiada; players e team são compartilhados com a entrada (não são modificados).
    """
    ball = trk.ball.copy()
    n = len(ball)
    gone = np.isnan(ball).any(axis=1)
    target = int(round(missing_rate * n))
    rng = np.random.default_rng(seed)
    p = 1.0 / max(mean_gap_s * fps, 1.0)
    while gone.sum() < target:
        start = int(rng.integers(n))
        size = min(int(rng.geometric(p)), target - int(gone.sum()))  # último buraco não ultrapassa a meta
        gone[start : start + size] = True
    ball[gone] = np.nan
    return Tracking(ball=ball, players=trk.players, team=trk.team)
