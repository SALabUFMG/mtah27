"""Leitura do ATD (tracking automático) em arrays por frame do vídeo tático, em metros.

Mesmo formato que a baseline oficial lê à mão (ver external/mtah-official/starter/baseline.py):
"<frame_id>:<slot>;<slot>;...:<bola>", cada grupo "x,y,..." normalizado 0-1, NaN se ausente.
"""

# ruff: noqa: I001  (a ordem importa: mtah27.paths precisa vir antes de `games`)
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mtah27 import paths  # noqa: F401  (configura sys.path para o repo oficial)
from baseline import team_of_each_track  # type: ignore  # noqa: E402
from games import GAME, GAMES, require, source_frames  # type: ignore  # noqa: E402

HOME, AWAY = 0, 1


@dataclass
class Tracking:
    ball: np.ndarray  # [N, 2] metros, NaN onde a bola não foi vista
    players: np.ndarray  # [N, S, 2] metros, NaN onde o slot está vazio
    team: np.ndarray  # [S] HOME/AWAY de cada slot


def _xy(group: str) -> tuple[float, float]:
    if group[0] == "N":
        return np.nan, np.nan
    x, y = group.split(",", 2)[:2]
    return float(x), float(y)


def load_tracking(game: str = GAME) -> Tracking:
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    team = team_of_each_track(meta_path)
    n = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]

    ball = np.full((n, 2), np.nan, dtype=np.float32)
    players = np.full((n, len(team), 2), np.nan, dtype=np.float32)
    with open(raw_path) as f:
        for line in f:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id é 1-based
            players[frame] = [_xy(g) for g in parts[1].split(";")]
            ball[frame] = _xy(parts[2])

    scale = np.array([length, width], dtype=np.float32)
    return Tracking(ball=ball * scale, players=players * scale, team=team)


def nearest_team(trk: Tracking) -> np.ndarray:
    """Time do jogador mais próximo da bola por frame: HOME, AWAY ou -1 (sem bola ou sem jogador)."""
    d = np.linalg.norm(trk.players - trk.ball[:, None, :], axis=2)  # [N, S], NaN se faltar algo
    valid = ~np.isnan(d).all(axis=1)
    out = np.full(len(d), -1, dtype=np.int8)
    out[valid] = trk.team[np.nanargmin(d[valid], axis=1)]
    return out
