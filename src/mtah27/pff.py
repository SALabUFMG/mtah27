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
CONTROL_TYPES = {"IT", "BC", "CR", "SH"}  # recepção, condução, cruzamento, chute: o time tem a bola


def game_ids() -> list[int]:
    return sorted(int(p.stem) for p in (ROOT / "metadata").glob("*.json"))


def _merge_rows(events: list[dict]) -> list[tuple[dict, set[str], bool]]:
    """Uma entrada por gameEventId: o JSON repete o evento para cada possessionEvent dele.

    Retorna (primeira linha, tipos de posse, algum passe completo), na ordem do arquivo.
    """
    merged: dict = {}
    for e in events:
        m = merged.setdefault(e["gameEventId"], [e, set(), False])
        pe = e.get("possessionEvents") or {}
        t = pe.get("possessionEventType")
        if t:
            m[1].add(t)
        if t == "PA" and pe.get("passOutcomeType") == "C":
            m[2] = True
    return [tuple(m) for m in merged.values()]


def controls(types: set[str], pass_complete: bool) -> bool:
    """O OTB mostra posse controlada (CHALLENGE.md: toque, desvio ou duelo sem controle não trocam).

    Recepção/condução/cruzamento/chute, ou passe depois de outra ação, sim. Passe de primeira
    isolado só se completo. Só duelo, corte, rebote, toque ou falta, não.
    """
    if types & CONTROL_TYPES:
        return True
    if "PA" in types:
        return pass_complete or types != {"PA"}
    return False


def events_frame(events: list[dict], fps: float) -> pd.DataFrame:
    """Eventos da PFF no formato de external.labels_from_events (frames = round(segundos de vídeo * fps)).

    - Linhas com o mesmo gameEventId viram um evento só.
    - OTB (jogador com a bola) vira controle do time (PASS) se for cobrança ou se `controls`; OTB sem
      time (duelo sem vencedor) nunca. OUT vira BALL OUT.
    - Uma cobrança (setpieceType diferente de "O") logo após um OTB, controlado ou não, marca bola morta
      desde o fim desse OTB: falta, impedimento, gol. Vai como FAULT RECEIVED, que lá é "D a partir de
      start_frame". Depois de um OUT não precisa: o BALL OUT já cobre.
    - Saídas de bola e END delimitam os tempos (o X fora deles vem de labels_from_events). Eventos
      depois do END do próprio período são descartados: a disputa de pênaltis vem marcada como
      período 4 depois do fim da prorrogação e não é jogo.
    """
    merged = sorted(_merge_rows(events), key=lambda m: m[0]["startTime"])
    period_end: dict[int, float] = {}
    for e, _, _ in merged:
        if e["gameEvents"]["gameEventType"] == "END":
            period_end.setdefault(e["gameEvents"]["period"], e["startTime"])  # o primeiro END
    rows = []
    prev_otb_end = None  # fim do último OTB, se ele foi o último evento de bola
    for e, types, pass_complete in merged:
        g = e["gameEvents"]
        kind, period = g["gameEventType"], g["period"]
        if e["startTime"] > period_end.get(period, float("inf")):
            continue
        start = round(e["startTime"] * fps)
        end = start if e["endTime"] is None else round(e["endTime"] * fps)
        team = "H" if g.get("homeTeam") else "A"
        if kind == "OTB":
            setpiece = g.get("setpieceType") not in (None, "O")
            if setpiece and prev_otb_end is not None:
                rows.append((team, "FAULT RECEIVED", "", period, prev_otb_end, prev_otb_end))
            if g.get("homeTeam") is not None and (setpiece or controls(types, pass_complete)):
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


def parse_tracking(lines, pitch: tuple[float, float]) -> tuple[Tracking, np.ndarray, np.ndarray]:
    """Linhas jsonl da PFF em arrays indexados por frameNum, em metros com origem no canto (0..L, 0..W).

    Slots por (time, número da camisa). Frames duplicados (vários eventos de posse) ficam com a última linha.
    Retorna também as máscaras de posição estimada pela PFF (visibility == "ESTIMATED"): bola [n] e
    jogadores [n, S]. A bola estimada é ruim (o mais próximo bate com o rótulo bem menos que na visível).
    """
    off = np.array(pitch, dtype=np.float32) / 2
    slot: dict[tuple[int, int], int] = {}
    frames, balls, best, pf, ps, pxy, pest = [], [], [], [], [], [], []
    for line in lines:
        r = json.loads(line)
        fr = r["frameNum"]
        frames.append(fr)
        b = r["balls"] or []
        seen = bool(b) and b[0]["x"] is not None
        balls.append((b[0]["x"], b[0]["y"]) if seen else (np.nan, np.nan))
        best.append(seen and b[0].get("visibility") == "ESTIMATED")
        for team, key in ((HOME, "homePlayers"), (AWAY, "awayPlayers")):
            for p in r[key] or []:
                pf.append(fr)
                ps.append(slot.setdefault((team, p["jerseyNum"]), len(slot)))
                pxy.append((p["x"], p["y"]) if p["x"] is not None else (np.nan, np.nan))
                pest.append(p.get("visibility") == "ESTIMATED")
    n = max(frames) + 1
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    ball[frames] = np.asarray(balls, dtype=np.float32) + off
    ball_est = np.zeros(n, dtype=bool)
    ball_est[frames] = best
    players = np.full((n, len(slot), 2), np.nan, dtype=np.float32)
    players[pf, ps] = np.asarray(pxy, dtype=np.float32).reshape(-1, 2) + off
    players_est = np.zeros((n, len(slot)), dtype=bool)
    players_est[pf, ps] = pest
    team = np.array([t for t, _ in slot], dtype=np.int8)
    return Tracking(ball=ball, players=players, team=team), ball_est, players_est


def load_game(
    gid: int, drop_estimated: tuple[str, ...] = ()
) -> tuple[Tracking, np.ndarray, float, tuple[float, float]]:
    """(tracking em metros, rótulos H/A/D/X por frame, fps, (comprimento, largura)) do jogo `gid`.

    drop_estimated: "ball" e/ou "players" viram NaN onde a PFF estimou a posição (o ATD do alvo
    não estima: a bola falta, os jogadores somem).

    O tracking é cacheado em INTERIM_DIR/pff/<gid>.npz (ler o .bz2 leva dezenas de segundos); um cache
    sem as máscaras de estimado é refeito. Apague o cache se mudar parse_tracking. Os rótulos são
    recalculados a cada chamada.
    """
    fps, pitch = _metadata(gid)
    cache = CACHE / f"{gid}.npz"
    z = np.load(cache) if cache.exists() else None
    if z is not None and "ball_est" in z:
        trk = Tracking(ball=z["ball"], players=z["players"], team=z["team"])
        ball_est, players_est = z["ball_est"], z["players_est"]
    else:
        with bz2.open(ROOT / "tracking" / f"{gid}.jsonl.bz2", "rt", encoding="utf-8") as f:
            trk, ball_est, players_est = parse_tracking(f, pitch)
        CACHE.mkdir(parents=True, exist_ok=True)
        arrays = {"ball": trk.ball, "players": trk.players, "team": trk.team}
        np.savez_compressed(cache, **arrays, ball_est=ball_est, players_est=players_est)
    if "ball" in drop_estimated:
        trk.ball[ball_est] = np.nan
    if "players" in drop_estimated:
        trk.players[players_est] = np.nan
    events = json.loads((ROOT / "events" / f"{gid}.json").read_text(encoding="utf-8"))
    labels = labels_from_events(events_frame(events, fps), len(trk.ball))
    return trk, labels, fps, pitch
