"""Jogos públicos da DFL (Bundesliga 1 e 2, 2022/23) com posse e bola viva/morta por frame.

Dados em DATA_DIR/external/dfl (scripts/fetch_dfl.py). Fonte: Bassek et al. (2025), Scientific Data,
https://doi.org/10.1038/s41597-025-04505-y, CC-BY 4.0, dados da Deutsche Fußball Liga.

Diferente da Metrica e da PFF, aqui o rótulo não sai de eventos: o próprio tracking traz, em cada frame
da bola, BallStatus (1 = em jogo) e BallPossession (1 = mandante, 2 = visitante). É a mesma construção
do gabarito da competição (CHALLENGE.md): D onde o tracking marca bola morta, senão o time que o
tracking aponta como dono. Os valores foram anotados pela DFL, em outros jogos.

A bola está presente em todos os frames, inclusive com a bola morta.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from array import array

import numpy as np

from mtah27.paths import DATA_DIR
from mtah27.tracking import AWAY, HOME, Tracking

ROOT = DATA_DIR / "external" / "dfl"
FPS = 25.0
SECTIONS = ("firstHalf", "secondHalf")
GAP = 25  # frames X (sem tracking) entre os tempos, para os tempos virarem blocos separados
H, A, D, X = 0, 1, 2, 3


def game_ids() -> list[str]:
    return sorted(p.stem.rsplit("DFL-MAT-", 1)[1] for p in ROOT.glob("*positions_raw_observed*.xml"))


def _path(kind: str, match_id: str):
    found = sorted(ROOT.glob(f"*{kind}*DFL-MAT-{match_id}.xml"))
    if len(found) != 1:
        raise FileNotFoundError(f"esperado 1 arquivo {kind} do jogo {match_id} em {ROOT}; rode fetch_dfl.py")
    return found[0]


def parse_meta(source) -> tuple[tuple[float, float], dict[str, int]]:
    """(comprimento, largura) do campo em metros e {TeamId: HOME/AWAY}."""
    root = ET.parse(source).getroot()
    env = root.find(".//Environment")
    sides = {"home": HOME, "guest": AWAY}
    teams = {t.get("TeamId"): sides[t.get("Role")] for t in root.iter("Team") if t.get("Role") in sides}
    if sorted(teams.values()) != [HOME, AWAY]:
        raise ValueError(f"metadados sem mandante e visitante: {teams}")
    return (float(env.get("PitchX")), float(env.get("PitchY"))), teams


def parse_positions(source, pitch: tuple[float, float], teams: dict[str, int]) -> tuple[Tracking, np.ndarray]:
    """Tracking em metros (origem no canto, como o resto do projeto) e rótulos H/A/D/X por frame.

    Os dois tempos ficam em sequência, com GAP frames X entre eles. Árbitros ficam de fora.
    """
    length, width = pitch
    sets: dict[tuple[str, str], tuple] = {}  # (seção, objeto) -> (N, X, Y[, posse, status])
    side: dict[str, int] = {}
    cur = None
    for event, el in ET.iterparse(source, events=("start", "end")):
        if event == "start" and el.tag == "FrameSet":
            team, sec = el.get("TeamId"), el.get("GameSection")
            obj = "ball" if team == "BALL" else el.get("PersonId")
            cur = None
            if sec in SECTIONS and (obj == "ball" or team in teams):
                n_cols = 5 if obj == "ball" else 3
                cur = sets.setdefault((sec, obj), tuple(array("d") for _ in range(n_cols)))
                if obj != "ball":
                    side[obj] = teams[team]
        elif event == "end" and el.tag == "Frame":
            if cur is not None:
                cur[0].append(float(el.get("N")))
                cur[1].append(float(el.get("X")))
                cur[2].append(float(el.get("Y")))
                if len(cur) == 5:
                    cur[3].append(float(el.get("BallPossession", 0)))
                    cur[4].append(float(el.get("BallStatus", 0)))
            el.clear()
        elif event == "end" and el.tag == "FrameSet":
            el.clear()

    slots = {obj: i for i, obj in enumerate(sorted(side))}
    spans, n = [], 0
    for sec in SECTIONS:
        first = int(min(v[0][0] for (s, _), v in sets.items() if s == sec))
        last = int(max(v[0][-1] for (s, _), v in sets.items() if s == sec))
        spans.append((sec, first, n))
        n += last - first + 1 + GAP
    n -= GAP
    ball = np.full((n, 2), np.nan, np.float32)
    players = np.full((n, len(slots), 2), np.nan, np.float32)
    labels = np.full(n, X, np.int8)
    shift = np.array([length / 2, width / 2])  # a DFL usa metros com origem no centro do campo
    for sec, first, off in spans:
        for (s, obj), v in sets.items():
            if s != sec:
                continue
            idx = np.asarray(v[0], np.int64) - first + off
            xy = np.stack([np.asarray(v[1]), np.asarray(v[2])], axis=1) + shift
            if obj == "ball":
                ball[idx] = xy
                poss, alive = np.asarray(v[3]), np.asarray(v[4]) == 1
                labels[idx] = np.where(alive, np.where(poss == 1, H, A), D)
            else:
                players[idx, slots[obj]] = xy
    team = np.array([side[obj] for obj in sorted(side)], np.int8)
    return Tracking(ball=ball, players=players, team=team), labels


def load_game(match_id: str) -> tuple[Tracking, np.ndarray, float, tuple[float, float]]:
    """(tracking, rótulos, fps, campo) de um jogo; guarda um .npz ao lado dos XML na primeira leitura."""
    cache = ROOT / f"{match_id}.npz"
    if cache.exists():
        z = np.load(cache)
        trk = Tracking(ball=z["ball"], players=z["players"], team=z["team"])
        return trk, z["labels"], FPS, (float(z["pitch"][0]), float(z["pitch"][1]))
    pitch, teams = parse_meta(_path("matchinformation", match_id))
    trk, labels = parse_positions(_path("positions_raw_observed", match_id), pitch, teams)
    np.savez_compressed(
        cache, ball=trk.ball, players=trk.players, team=trk.team, labels=labels, pitch=np.array(pitch)
    )
    return trk, labels, FPS, pitch
