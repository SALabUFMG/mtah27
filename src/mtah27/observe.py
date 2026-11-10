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


def grid(n: int, fps: float, hz: float) -> np.ndarray:
    """Frame nativo de cada amostra da grade de hz: amostra k -> round(k / hz * fps), dentro de [0, n).

    Grade comum a todos os providers (25 fps no alvo e na Metrica, 29,97 na PFF): o HMM vê o mesmo
    passo de tempo em todos, e o tracking da PFF (~15 Hz repetido a 29,97) deixa de ter frames duplicados.
    """
    k = np.arange(int(np.floor((n - 1) * hz / fps)) + 1)
    return np.minimum(np.round(k * fps / hz).astype(np.int64), n - 1)


def subsample(trk: Tracking, idx: np.ndarray) -> Tracking:
    return Tracking(ball=trk.ball[idx], players=trk.players[idx], team=trk.team)


def grid_blocks(idx: np.ndarray, blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Blocos nativos [first, last] (inclusivos) em índices da grade; blocos sem amostra somem."""
    out = []
    for a, b in blocks:
        ks = np.flatnonzero((idx >= a) & (idx <= b))
        if len(ks):
            out.append((int(ks[0]), int(ks[-1])))
    return out


def player_speed(trk: Tracking, fps: int, window_s: float = 1.0, v_max: float = 12.0) -> np.ndarray:
    """Mediana da velocidade dos jogadores (m/s) por frame, média móvel centrada de window_s.

    Não depende da bola: em bola morta os jogadores param, mesmo quando o tracking perde a bola.
    Saltos acima de v_max são troca de identidade do tracklet e ficam de fora. NaN sem jogador.
    """
    v = np.linalg.norm(np.diff(trk.players, axis=0), axis=2) * fps
    v = np.where(v > v_max, np.nan, v)
    ok_slot = ~np.isnan(v)
    med = np.full(len(v), np.nan)
    rows = ok_slot.any(axis=1)
    med[rows] = np.nanmedian(v[rows], axis=1)
    med = np.r_[med[:1], med]
    ok = ~np.isnan(med)
    k = np.ones(max(1, round(window_s * fps)))
    den = np.convolve(ok, k, "same")
    num = np.convolve(np.where(ok, med, 0.0), k, "same")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan).astype(np.float32)


def normalize_speed(speed: np.ndarray, blocks: list[tuple[int, int]]) -> np.ndarray:
    """Velocidade dividida pela mediana do próprio bloco [first, last] (inclusivo), um por tempo de jogo.

    A escala absoluta muda com o provider (D ~0,9 m/s na Metrica, ~1,2 na PFF); relativa à mediana do
    tempo, fica parecida (D ~0,7, vivo ~1,2 nas duas). Fora dos blocos: NaN.
    """
    out = np.full(len(speed), np.nan, dtype=np.float32)
    for a, b in blocks:
        seg = speed[a : b + 1]
        if not np.isnan(seg).all():
            out[a : b + 1] = seg / np.nanmedian(seg)
    return out


def speed_bin(speed: np.ndarray, bins: list[float]) -> np.ndarray:
    """Faixa de velocidade (0 = mais lenta) por frame; NaN cai na faixa mais rápida (sem sinal de parada)."""
    return np.digitize(np.nan_to_num(speed, nan=np.inf), bins).astype(np.int8)


def degrade(
    trk: Tracking,
    missing_rate: float,
    mean_gap_s: float,
    fps: float,
    seed: int,
    blocks: list[tuple[int, int]],
) -> Tracking:
    """Cópia de trk com a bola apagada em buracos contíguos (comprimento geométrico) até a fração
    de frames sem bola dentro dos blocos [first, last] (inclusivos; contando os já ausentes) chegar a
    missing_rate. Fora dos blocos nada muda: a meta do alvo (DATA.md) é sobre os frames avaliados.
    Determinístico por seed.

    A bola é copiada; players e team são compartilhados com a entrada (não são modificados).
    """
    ball = trk.ball.copy()
    inside = np.zeros(len(ball), dtype=bool)
    for a, b in blocks:
        inside[a : b + 1] = True
    pos = np.flatnonzero(inside)
    gone = np.isnan(ball).any(axis=1) & inside
    target = int(round(missing_rate * len(pos)))
    rng = np.random.default_rng(seed)
    p = 1.0 / max(mean_gap_s * fps, 1.0)
    while gone.sum() < target:
        start = int(pos[rng.integers(len(pos))])
        size = min(int(rng.geometric(p)), target - int(gone.sum()))  # último buraco não ultrapassa a meta
        gone[start : start + size] |= inside[start : start + size]
    ball[gone] = np.nan
    return Tracking(ball=ball, players=trk.players, team=trk.team)
