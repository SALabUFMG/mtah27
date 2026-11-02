"""Ponto de entrada do modelo: dados -> um label por frame.

Este é o único contrato que `scripts/make_submission.py` usa. A arquitetura do time entra aqui
(e em módulos irmãos dentro de src/mtah27/).

Caminho C0: observação por frame (time do jogador mais próximo da bola, ou bola ausente) num HMM
H/A/D com duração mínima, decodificado por Viterbi em cada tempo. Fora dos tempos, 'X'.
"""

# ruff: noqa: I001  (a ordem importa: mtah27.paths precisa vir antes de `games`)
from __future__ import annotations

from typing import Any

import numpy as np

from mtah27.submission import empty_labels
from mtah27.tracking import AWAY, HOME, load_tracking, nearest_team
from mtah27.viterbi import STATES, viterbi
from games import GAMES, half_video_blocks  # type: ignore  # noqa: E402

OBS = ("near_home", "near_away", "missing")


def observations(nearest: np.ndarray) -> np.ndarray:
    """Índice em OBS por frame."""
    return np.where(nearest == HOME, 0, np.where(nearest == AWAY, 1, 2))


def decode(obs: np.ndarray, c0: dict[str, Any], fps: int) -> np.ndarray:
    """Labels H/A/D para uma sequência contínua de observações."""
    emit = np.log(np.array([[c0["emission"][s][o] for o in OBS] for s in STATES]))
    min_dur = [max(1, round(c0["min_duration_s"][s] * fps)) for s in STATES]
    p = c0["switch_prob"]
    switch = np.log(np.array([[1 - p if i == j else p / 2 for j in range(3)] for i in range(3)]))
    return STATES[viterbi(emit[:, obs].T, min_dur, switch)]


def predict(config: dict[str, Any]) -> np.ndarray:
    """Retorna um array de 199.319 labels em {H, A, D, X}, indexado pelo frame do vídeo tático."""
    game = config.get("game", "BAR-ATH")
    obs = observations(nearest_team(load_tracking(game)))
    labels = empty_labels(game)
    for _, first, last in half_video_blocks(game):
        labels[first : last + 1] = decode(obs[first : last + 1], config["c0"], GAMES[game]["fps"])
    return labels
