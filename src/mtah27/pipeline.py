"""Ponto de entrada do modelo: dados -> um label por frame.

Este é o único contrato que `scripts/make_submission.py` usa. A arquitetura do time entra aqui
(e em módulos irmãos dentro de src/mtah27/).

`model` no config escolhe o caminho:
- c0: observação da bola e velocidade dos jogadores (observe.py) num HMM H/A/D com duração mínima,
  decodificado por Viterbi em cada tempo; parâmetros em configs/c0_fitted.yaml (scripts/calibrate_c0.py).
- c1: o mesmo Viterbi com a emissão de um GBM sobre features de janela (features.py, c1.py), treinado
  na hora nos 3 jogos públicos da Metrica; hiperparâmetros em configs/c1_fitted.yaml
  (scripts/calibrate_c1.py).
Fora dos tempos, 'X'.
"""

# ruff: noqa: I001  (a ordem importa: mtah27.paths precisa vir antes de `games`)
from __future__ import annotations

from typing import Any

import numpy as np

from mtah27 import c0, c1, features, observe
from mtah27.config import load_config
from mtah27.submission import empty_labels
from mtah27.tracking import load_tracking
from games import GAMES, half_video_blocks  # type: ignore  # noqa: E402

CODES = np.array(list("HADX"))


def predict(config: dict[str, Any]) -> np.ndarray:
    """Retorna um array de 199.319 labels em {H, A, D, X}, indexado pelo frame do vídeo tático."""
    if config.get("model", "c0") == "c1":
        return _predict_c1(config)
    game = config.get("game", "BAR-ATH")
    fitted = load_config(config["c0"]["fitted"])
    trk = load_tracking(game)
    n, hz = len(trk.ball), fitted["hz"]
    idx = observe.grid(n, GAMES[game]["fps"], hz)
    trk = observe.subsample(trk, idx)
    obs = observe.observe(trk, fitted["r_close_m"], GAMES[game]["pitch"])
    blocks = [(first, last) for _, first, last in half_video_blocks(game)]
    speed = observe.player_speed(trk, hz, fitted["speed"]["window_s"])
    speed = observe.normalize_speed(speed, observe.grid_blocks(idx, blocks))
    mot = observe.speed_bin(speed, fitted["speed"]["bins_rel"])
    codes = c0.decode_game(obs, mot, fitted, fitted["min_dur_s"], hz, idx, blocks, n)
    out = empty_labels(game)
    out[:] = CODES[codes]
    return out


def _predict_c1(config: dict[str, Any]) -> np.ndarray:
    game = config.get("game", "BAR-ATH")
    fitted_cfg = load_config(config["c1"]["fitted"])
    model = c1.fit_metrica(config["c0"], fitted_cfg)
    trk = load_tracking(game)
    n, hz = len(trk.ball), fitted_cfg["hz"]
    idx = observe.grid(n, GAMES[game]["fps"], hz)
    blocks = [(first, last) for _, first, last in half_video_blocks(game)]
    X = features.window_features(
        observe.subsample(trk, idx), GAMES[game]["pitch"], hz, observe.grid_blocks(idx, blocks),
        fitted_cfg["window_s"],
    )  # fmt: skip
    codes = c1.decode_game(
        X, model, fitted_cfg["min_dur_s"], hz, idx, blocks, n, fitted_cfg["emission_weight"]
    )
    out = empty_labels(game)
    out[:] = CODES[codes]
    return out
