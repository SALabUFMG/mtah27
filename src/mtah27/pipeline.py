"""Ponto de entrada do modelo: dados -> um label por frame.

Este é o único contrato que `scripts/make_submission.py` usa. A arquitetura do time entra aqui
(e em módulos irmãos dentro de src/mtah27/).

Caminho C0: observação da bola e velocidade dos jogadores (observe.py) num HMM H/A/D com duração
mínima, decodificado por Viterbi em cada tempo; parâmetros em configs/c0_fitted.yaml
(scripts/calibrate_c0.py). Fora dos tempos, 'X'.
"""

# ruff: noqa: I001  (a ordem importa: mtah27.paths precisa vir antes de `games`)
from __future__ import annotations

from typing import Any

import numpy as np

from mtah27 import c0, observe
from mtah27.config import load_config
from mtah27.submission import empty_labels
from mtah27.tracking import load_tracking
from games import GAMES, half_video_blocks  # type: ignore  # noqa: E402

CODES = np.array(list("HADX"))


def predict(config: dict[str, Any]) -> np.ndarray:
    """Retorna um array de 199.319 labels em {H, A, D, X}, indexado pelo frame do vídeo tático."""
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
