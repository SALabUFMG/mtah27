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
    fps = GAMES[game]["fps"]
    obs = observe.observe(trk, fitted["r_close_m"], GAMES[game]["pitch"])
    speed = observe.player_speed(trk, fps, fitted["speed"]["window_s"])
    mot = observe.speed_bin(speed, fitted["speed"]["bins_ms"])
    blocks = [(first, last) for _, first, last in half_video_blocks(game)]
    codes = c0.decode_game(obs, mot, fitted, fitted["min_dur_s"], fps, blocks)
    out = empty_labels(game)
    out[:] = CODES[codes]
    return out
