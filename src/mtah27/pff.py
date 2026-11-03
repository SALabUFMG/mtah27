"""Dados públicos da PFF (Copa do Mundo 2022, 64 jogos) como fonte de pseudo-rótulos de posse.

Dados em DATA_DIR/external/pff (extraia com scripts/extract_pff.py). Formato: PFF FC Tracking Data
Specification v2.2 e Event Data Specification v2.5, que vêm nos zips.

A observação usa `balls` e `homePlayers`/`awayPlayers` brutos (tracking da transmissão). `ballsSmoothed`
fica de fora: é a bola recolocada a partir dos eventos, então vazaria o rótulo.
"""

from __future__ import annotations

import bz2
import json

import numpy as np
import pandas as pd

from mtah27.external import labels_from_events
from mtah27.paths import DATA_DIR, INTERIM_DIR
from mtah27.tracking import AWAY, HOME, Tracking

ROOT = DATA_DIR / "external" / "pff"
CACHE = INTERIM_DIR / "pff"
PERIOD_MARKS = {"FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF", "END"}


def game_ids() -> list[int]:
    return sorted(int(p.stem) for p in (ROOT / "metadata").glob("*.json"))


def events_frame(events: list[dict], fps: float) -> pd.DataFrame:
    """Eventos da PFF no formato de external.labels_from_events (frames = round(segundos de vídeo * fps)).

    - OTB (jogador com a bola) vira controle do time (PASS); OUT vira BALL OUT.
    - Uma cobrança (setpieceType diferente de "O") logo após um OTB marca bola morta desde o fim desse
      OTB: falta, impedimento, gol. Vai como FAULT RECEIVED, que lá é "D a partir de start_frame".
      Depois de um OUT não precisa: o BALL OUT já cobre.
    - Saídas de bola e END delimitam os tempos (o X fora deles vem de labels_from_events).
    """
    rows = []
    prev_otb_end = None  # fim do último OTB, se ele foi o último evento de bola
    for e in sorted(events, key=lambda e: e["startTime"]):
        g = e["gameEvents"]
        kind, period = g["gameEventType"], g["period"]
        start = round(e["startTime"] * fps)
        end = start if e["endTime"] is None else round(e["endTime"] * fps)
        team = "H" if g.get("homeTeam") else "A"
        if kind == "OTB":
            if g.get("setpieceType") not in (None, "O") and prev_otb_end is not None:
                rows.append((team, "FAULT RECEIVED", "", period, prev_otb_end, prev_otb_end))
            rows.append((team, "PASS", "", period, start, end))
            prev_otb_end = end
        elif kind == "OUT":
            rows.append((team, "BALL OUT", "", period, start, start))
            prev_otb_end = None
        elif kind in PERIOD_MARKS:
            rows.append((team, kind, "", period, start, end))
            prev_otb_end = None
    return pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start_frame", "end_frame"])


def _metadata(gid: int) -> tuple[float, tuple[float, float]]:
    m = json.loads((ROOT / "metadata" / f"{gid}.json").read_text(encoding="utf-8"))
    m = m[0] if isinstance(m, list) else m
    pitch = m["stadium"]["pitches"][0]
    return float(m["fps"]), (float(pitch["length"]), float(pitch["width"]))


def parse_tracking(lines, pitch: tuple[float, float]) -> Tracking:
    """Linhas jsonl da PFF em arrays indexados por frameNum, em metros com origem no canto (0..L, 0..W).

    Slots por (time, número da camisa). Frames duplicados (vários eventos de posse) ficam com a última linha.
    """
    off = np.array(pitch, dtype=np.float32) / 2
    slot: dict[tuple[int, int], int] = {}
    frames, balls, pf, ps, pxy = [], [], [], [], []
    for line in lines:
        r = json.loads(line)
        fr = r["frameNum"]
        frames.append(fr)
        b = r["balls"] or []
        balls.append((b[0]["x"], b[0]["y"]) if b and b[0]["x"] is not None else (np.nan, np.nan))
        for team, key in ((HOME, "homePlayers"), (AWAY, "awayPlayers")):
            for p in r[key] or []:
                pf.append(fr)
                ps.append(slot.setdefault((team, p["jerseyNum"]), len(slot)))
                pxy.append((p["x"], p["y"]) if p["x"] is not None else (np.nan, np.nan))
    n = max(frames) + 1
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    ball[frames] = np.asarray(balls, dtype=np.float32) + off
    players = np.full((n, len(slot), 2), np.nan, dtype=np.float32)
    players[pf, ps] = np.asarray(pxy, dtype=np.float32).reshape(-1, 2) + off
    team = np.array([t for t, _ in slot], dtype=np.int8)
    return Tracking(ball=ball, players=players, team=team)


def load_game(gid: int) -> tuple[Tracking, np.ndarray, float, tuple[float, float]]:
    """(tracking em metros, rótulos H/A/D/X por frame, fps, (comprimento, largura)) do jogo `gid`.

    O tracking é cacheado em INTERIM_DIR/pff/<gid>.npz (ler o .bz2 leva dezenas de segundos);
    apague o cache se mudar parse_tracking. Os rótulos são recalculados a cada chamada.
    """
    fps, pitch = _metadata(gid)
    cache = CACHE / f"{gid}.npz"
    if cache.exists():
        z = np.load(cache)
        trk = Tracking(ball=z["ball"], players=z["players"], team=z["team"])
    else:
        with bz2.open(ROOT / "tracking" / f"{gid}.jsonl.bz2", "rt", encoding="utf-8") as f:
            trk = parse_tracking(f, pitch)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ball=trk.ball, players=trk.players, team=trk.team)
    events = json.loads((ROOT / "events" / f"{gid}.json").read_text(encoding="utf-8"))
    labels = labels_from_events(events_frame(events, fps), len(trk.ball))
    return trk, labels, fps, pitch
