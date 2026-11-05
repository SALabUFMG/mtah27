"""Caminho C1: emissão discriminativa (GBM) no mesmo Viterbi do C0.

HMM híbrido: um classificador treinado em features por amostra estima P(estado | x). O HMM precisa de
P(x | estado), que pela regra de Bayes é proporcional a P(estado | x) / P(estado) (a P(x) é comum aos
estados e some no argmax do Viterbi). Essa "verossimilhança escalada" entra como log-emissão.

O decodificador é o mesmo do C0 (viterbi com duração mínima e transição estimada em c0.fit_transition),
então qualquer diferença de placar vem só da emissão. O classificador é o HistGradientBoosting do
scikit-learn (o LightGBM empatou na validação e saiu para ter uma dependência a menos).
"""

# ruff: noqa: I001  (a ordem importa: mtah27.paths precisa vir antes de possession_metrics)
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from mtah27 import c0, external, features, observe
from mtah27.viterbi import viterbi
import possession_metrics as pm  # type: ignore  # noqa: E402  (no sys.path via mtah27.c0 -> paths)

DEFAULTS = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "early_stopping": False}


def fit(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    params: dict | None = None,
    sample_weight: np.ndarray | None = None,
) -> dict:
    """Treina nas linhas H/A/D (y < 3); devolve modelo, prior [3] e transição [3,3]."""
    X, y = np.asarray(X), np.asarray(y)
    keep = y < 3
    present = np.unique(y[keep])
    if not np.array_equal(present, [0, 1, 2]):
        raise ValueError(f"o treino precisa das 3 classes H/A/D (0,1,2); presentes: {present.tolist()}")
    model = HistGradientBoostingClassifier(random_state=seed, **{**DEFAULTS, **(params or {})})
    w = None if sample_weight is None else np.asarray(sample_weight)[keep]
    model.fit(X[keep], y[keep], sample_weight=w)
    if not np.array_equal(model.classes_, [0, 1, 2]):  # garante colunas de predict_proba = H,A,D
        raise RuntimeError(f"ordem de classes inesperada: {model.classes_.tolist()}")
    prior = np.bincount(y[keep], minlength=3) / keep.sum()
    return {"model": model, "prior": prior, "transition": c0.fit_transition(y)}


def log_emission(fitted: dict, X: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    """log P(estado|x) - log prior, [N,3]; probabilidades clipadas em floor antes do log."""
    proba = np.clip(fitted["model"].predict_proba(np.asarray(X)), floor, 1.0)
    return np.log(proba) - np.log(fitted["prior"])


def decode_game(
    X: np.ndarray,
    fitted: dict,
    min_dur_s: dict[str, float],
    hz: float,
    idx: np.ndarray,
    blocks: list[tuple[int, int]],
    n: int,
    emission_weight: float = 1.0,
) -> np.ndarray:
    """Como c0.decode_game, mas com a emissão do GBM; int8 [n], X (3) fora dos blocos."""
    out = np.full(n, 3, dtype=np.int8)
    min_dur = [max(1, round(min_dur_s[s] * hz)) for s in "HAD"]
    log_switch = np.log(fitted["transition"])
    for a, b in blocks:
        ks = np.flatnonzero((idx >= a) & (idx <= b))
        if not len(ks):
            continue
        le = log_emission(fitted, np.asarray(X)[ks]) * emission_weight
        dec = viterbi(le, min_dur, log_switch).astype(np.int8)
        out[a : b + 1] = c0.to_native(dec, idx[ks], a, b)
    return out


def game_features(trk, labels, fps, pitch, hz, degrade: dict, window_s: float, n_aug: int) -> dict:
    """Features de n_aug cópias degradadas de um jogo externo (seeds degrade.seed + 0..n_aug-1).

    A bola é apagada só dentro dos blocos rotulados, como no C0. A cópia 0 é a da avaliação.
    """
    idx = observe.grid(len(labels), fps, hz)
    blocks = [(a, b - 1) for a, b in pm.valid_runs(labels)]
    gblocks = observe.grid_blocks(idx, blocks)
    sub = observe.subsample(trk, idx)
    feats = []
    for s in range(n_aug):
        deg = observe.degrade(sub, **{**degrade, "seed": degrade["seed"] + s}, fps=hz, blocks=gblocks)
        feats.append(features.window_features(deg, pitch, hz, gblocks, window_s))
    return {"feats": feats, "labels": labels, "labels_grid": labels[idx], "idx": idx, "blocks": blocks}


def train_rows(games: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """X, y de todas as cópias dos jogos, com 1 linha X (NaN) entre elas para não criar transição falsa."""
    xs, ys = [], []
    nan_row = np.full((1, len(features.FEATURES)), np.nan, np.float32)
    for g in games:
        for x in g["feats"]:
            xs += [x, nan_row]
            ys += [g["labels_grid"], np.full(1, 3, g["labels_grid"].dtype)]
    return np.concatenate(xs), np.concatenate(ys)


def fit_metrica(cfg_c0: dict, fitted_cfg: dict) -> dict:
    """Treina o modelo da submissão nos 3 jogos públicos da Metrica (scripts/fetch_metrica_sample.py)."""
    games = []
    for i in (1, 2, 3):
        trk, labels = external.load_sample_game(i)
        games.append(
            game_features(
                trk, labels, 25, external.PITCH, fitted_cfg["hz"], cfg_c0["degrade"],
                fitted_cfg["window_s"], fitted_cfg["n_aug"],
            )
        )  # fmt: skip
    return fit(*train_rows(games), seed=0)
