"""Lacunas de bola medidas no alvo (sem rótulo) e reproduzidas nos jogos externos.

Nos jogos externos a bola quase só some em bola morta; no alvo ela some também com o jogo vivo. A
degradação de observe.degrade sorteia lacunas uniformes no tempo, então a ausência continua concentrada
em D no treino e o classificador aprende "bola ausente = D".

Aqui a ausência é condicionada a um sinal que existe nos dois lados sem depender da bola nem de rótulo:
a faixa de velocidade mediana dos jogadores, em posto dentro do próprio tempo de jogo (jogadores parados
indicam bola morta). fit() mede, por faixa, a fração de amostras sem bola e os quantis do comprimento das
lacunas que começam nela; degrade_like() apaga a bola de outro jogo até chegar a essas frações.

Só agregados saem do alvo (n_bins frações e n_bins x n_q quantis), nada por frame.
"""

from __future__ import annotations

import numpy as np

from mtah27.observe import player_speed
from mtah27.tracking import Tracking


def speed_bins(trk: Tracking, hz: float, blocks: list[tuple[int, int]], n_bins: int) -> np.ndarray:
    """Faixa 0..n_bins-1 (0 = mais lenta) por amostra: posto da velocidade mediana dos jogadores dentro
    de cada bloco [first, last]. -1 fora dos blocos ou sem velocidade.

    Posto em vez de corte fixo: a escala da velocidade muda com o provider e com o ruído do tracker.
    """
    speed = player_speed(trk, hz, 1.0)
    out = np.full(len(speed), -1, dtype=np.int8)
    for a, b in blocks:
        seg = speed[a : b + 1]
        ok = np.flatnonzero(~np.isnan(seg))
        if not len(ok):
            continue
        rank = np.empty(len(ok), dtype=np.int64)
        rank[np.argsort(seg[ok], kind="stable")] = np.arange(len(ok))
        out[a + ok] = np.minimum(rank * n_bins // len(ok), n_bins - 1)
    return out


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Início e comprimento de cada trecho contíguo True."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return edges[::2], edges[1::2] - edges[::2]


def fit(trk: Tracking, hz: float, blocks: list[tuple[int, int]], n_bins: int = 10, n_q: int = 21) -> dict:
    """Por faixa de velocidade: fração de amostras sem bola e quantis do comprimento (em amostras) das
    lacunas que começam nela. trk já na grade de hz; blocks em índices da grade."""
    bins = speed_bins(trk, hz, blocks, n_bins)
    gone = np.isnan(trk.ball).any(axis=1) & (bins >= 0)
    starts, lengths = _runs(gone)
    qs = np.linspace(0.0, 1.0, n_q)
    rate, len_q = [], []
    for b in range(n_bins):
        sel = bins == b
        rate.append(float(gone[sel].mean()) if sel.any() else 0.0)
        ls = lengths[bins[starts] == b]
        len_q.append([float(v) for v in np.quantile(ls, qs)] if len(ls) else [1.0] * n_q)
    return {"hz": hz, "n_bins": n_bins, "missing_rate": rate, "gap_len_q": len_q}


def degrade_like(
    trk: Tracking, fitted: dict, hz: float, seed: int, blocks: list[tuple[int, int]]
) -> Tracking:
    """Cópia de trk com a bola apagada até cada faixa de velocidade chegar à fração sem bola de fitted
    (contando as já ausentes; faixas que já passam dela ficam como estão). O comprimento de cada lacuna
    é sorteado dos quantis da faixa em que ela começa. Determinístico por seed; players e team são
    compartilhados com a entrada.
    """
    if hz != fitted["hz"]:
        raise ValueError(f"lacunas medidas a {fitted['hz']} Hz, jogo a {hz} Hz")
    n_bins = fitted["n_bins"]
    ball = trk.ball.copy()
    bins = speed_bins(trk, hz, blocks, n_bins)
    inside = bins >= 0
    gone = np.isnan(ball).any(axis=1) & inside
    rng = np.random.default_rng(seed)
    total = np.bincount(bins[inside], minlength=n_bins)
    target = np.round(np.asarray(fitted["missing_rate"]) * total).astype(np.int64)
    count = np.bincount(bins[gone], minlength=n_bins)
    for b in range(n_bins):
        pos = np.flatnonzero(bins == b)
        q = np.asarray(fitted["gap_len_q"][b])
        while count[b] < target[b]:
            start = int(pos[rng.integers(len(pos))])
            size = max(1, int(round(np.interp(rng.random(), np.linspace(0.0, 1.0, len(q)), q))))
            seg = slice(start, start + size)
            # a lacuna só apaga amostras de faixas ainda abaixo da meta: sem isso ela vaza para a
            # faixa vizinha e passa da fração medida
            add = inside[seg] & ~gone[seg] & (count < target)[bins[seg]]
            gone[seg] |= add
            count += np.bincount(bins[seg][add], minlength=n_bins)
    ball[gone] = np.nan
    return Tracking(ball=ball, players=trk.players, team=trk.team)
