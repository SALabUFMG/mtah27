"""Jogos públicos da Metrica (sample-data) como fonte de pseudo-rótulos de posse, em outro jogo.

Dados em DATA_DIR/external/metrica_sample, com os mesmos caminhos relativos do repositório
metrica-sports/sample-data (commit e706dd506b360d69d9d123d5b8026e7294b13996).
Baixe com scripts/fetch_metrica_sample.py.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from mtah27.paths import DATA_DIR
from mtah27.tracking import AWAY, HOME, Tracking, _xy

H, A, D, X = 0, 1, 2, 3
PITCH = (105.0, 68.0)
ROOT = DATA_DIR / "external" / "metrica_sample" / "data"
CONTROL = {"PASS", "SET PIECE", "RECOVERY", "SHOT", "CARRY"}
_TEAM_CODE = {"H": H, "A": A}


def labels_from_events(events: pd.DataFrame, n_frames: int) -> np.ndarray:
    """Rótulos H/A/D/X (int8) por frame a partir de eventos normalizados.

    Colunas: team ("H"/"A"), type, subtype (str, pode ser vazia), period, start_frame, end_frame
    (índices 0-based no array de tracking). Regras (ver docs/official/CHALLENGE.md):

    - X fora dos períodos: antes do primeiro evento e depois do último de cada período (inclui o
      intervalo entre períodos e antes/depois da partida).
    - D a partir do end_frame de `BALL OUT`, do start_frame de `FAULT RECEIVED` e do end_frame de
      `SHOT` cujo subtipo contém `GOAL`, até (exclusivo) o start_frame do próximo evento de controle
      (a retomada; tiro de meta vem como PASS/GOAL KICK, não como SET PIECE).
    - Nos frames vivos, vale o time do último evento de controle (PASS, SET PIECE, RECOVERY, SHOT,
      CARRY), a partir do seu start_frame. `BALL LOST` e `CHALLENGE` não mudam a posse.
    - Frames vivos antes de qualquer evento de controle ficam X.
    """
    ev = events.sort_values("start_frame", kind="stable")
    out = np.full(n_frames, X, dtype=np.int8)

    for r in ev[ev["type"].isin(CONTROL)].itertuples():
        out[max(r.start_frame, 0) :] = _TEAM_CODE[r.team]

    restarts = ev.loc[ev["type"].isin(CONTROL), "start_frame"].to_numpy()
    for r in ev.itertuples():
        if r.type == "BALL OUT" or (r.type == "SHOT" and "GOAL" in str(r.subtype or "")):
            t = max(r.end_frame, r.start_frame)
        elif r.type == "FAULT RECEIVED":
            t = r.start_frame
        else:
            continue
        later = restarts[restarts > t]
        out[max(t, 0) : later[0] if len(later) else n_frames] = D

    live = np.zeros(n_frames, dtype=bool)
    for _, g in ev.groupby("period"):
        lo = max(int(g["start_frame"].min()), 0)
        hi = min(int(max(g["start_frame"].max(), g["end_frame"].max())), n_frames - 1)
        live[lo : hi + 1] = True
    out[~live] = X
    return out


def _require(*paths):
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Arquivo ausente: {missing[0]}. Rode scripts/fetch_metrica_sample.py")


def _csv_tracking(path):
    df = pd.read_csv(path, skiprows=3, header=None)
    frames = df[1].to_numpy(int) - 1  # frames começam em 1
    xy = df.iloc[:, 3:].to_numpy(np.float32)
    return frames, xy[:, :-2].reshape(len(df), -1, 2), xy[:, -2:]


def _load_csv_game(i: int) -> tuple[Tracking, np.ndarray]:
    d = ROOT / f"Sample_Game_{i}"
    home = d / f"Sample_Game_{i}_RawTrackingData_Home_Team.csv"
    away = d / f"Sample_Game_{i}_RawTrackingData_Away_Team.csv"
    evp = d / f"Sample_Game_{i}_RawEventsData.csv"
    _require(home, away, evp)
    frames, hp, ball = _csv_tracking(home)
    _, ap, _ = _csv_tracking(away)
    n = int(frames.max()) + 1
    players = np.full((n, hp.shape[1] + ap.shape[1], 2), np.nan, dtype=np.float32)
    ball_all = np.full((n, 2), np.nan, dtype=np.float32)
    players[frames] = np.concatenate([hp, ap], axis=1)
    ball_all[frames] = ball
    scale = np.array(PITCH, dtype=np.float32)
    team = np.array([HOME] * hp.shape[1] + [AWAY] * ap.shape[1], dtype=np.int8)
    trk = Tracking(ball=ball_all * scale, players=players * scale, team=team)

    e = pd.read_csv(evp)
    ev = pd.DataFrame(
        {
            "team": e["Team"].map({"Home": "H", "Away": "A"}),
            "type": e["Type"],
            "subtype": e["Subtype"].fillna(""),
            "period": e["Period"],
            "start_frame": e["Start Frame"] - 1,
            "end_frame": e["End Frame"] - 1,
        }
    )
    return trk, labels_from_events(ev, n)


def _load_epts_game(i: int) -> tuple[Tracking, np.ndarray]:
    d = ROOT / f"Sample_Game_{i}"
    meta = d / f"Sample_Game_{i}_metadata.xml"
    raw = d / f"Sample_Game_{i}_tracking.txt"
    evp = d / f"Sample_Game_{i}_events.json"
    _require(meta, raw, evp)
    root = ET.parse(meta).getroot()
    chan_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    slot = {}  # playerId -> índice do slot (ordem dos canais)
    for pid in chan_player.values():
        slot.setdefault(pid, len(slot))
    team_of = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    home_id = next(root.iter("Team")).get("id")
    team = np.array([HOME if team_of[p] == home_id else AWAY for p in slot], dtype=np.int8)
    # Cada DataFormatSpecification (trecho de frames) lista quais jogadores aparecem, em ordem.
    segments = []
    for spec in root.iter("DataFormatSpecification"):
        ids = [
            slot[chan_player[r.get("playerChannelId")]]
            for r in spec.iter("PlayerChannelRef")
            if r.get("playerChannelId").endswith("_x")
        ]
        segments.append((int(spec.get("startFrame")), int(spec.get("endFrame")), ids))
    n = max(sg[1] for sg in segments)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    players = np.full((n, len(slot), 2), np.nan, dtype=np.float32)
    with open(raw) as f:
        for line in f:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            fid = int(parts[0])
            ids = next(sg[2] for sg in segments if sg[0] <= fid <= sg[1])
            players[fid - 1, ids] = [_xy(g) for g in parts[1].split(";")]
            ball[fid - 1] = _xy(parts[2])
    scale = np.array(PITCH, dtype=np.float32)
    trk = Tracking(ball=ball * scale, players=players * scale, team=team)

    rows = []
    for e in json.loads(evp.read_text(encoding="utf-8"))["data"]:
        sub = e.get("subtypes")
        if isinstance(sub, dict):
            sub = [sub]
        rows.append(
            {
                "team": "H" if e["team"]["id"] == home_id else "A",
                "type": e["type"]["name"],
                "subtype": ",".join(s["name"] for s in sub or []),
                "period": e["period"],
                "start_frame": e["start"]["frame"] - 1,
                "end_frame": e["end"]["frame"] - 1,
            }
        )
    return trk, labels_from_events(pd.DataFrame(rows), n)


def load_sample_game(i: int) -> tuple[Tracking, np.ndarray]:
    """Tracking (metros, 25 fps) e rótulos H/A/D/X do jogo de exemplo `i` (1 a 3)."""
    if i not in (1, 2, 3):
        raise ValueError("i deve ser 1, 2 ou 3")
    return _load_epts_game(i) if i == 3 else _load_csv_game(i)
