import json

import numpy as np

from mtah27.external import A, D, H, X, labels_from_events
from mtah27.pff import events_frame, parse_tracking
from mtah27.tracking import AWAY, HOME

_ids = iter(range(10**6))


def _ge(kind, t0, t1, home=None, setpiece=None, period=1, types=("IT", "PA"), outcome="C"):
    """Um evento PFF como lista de linhas, uma por possessionEvent (como no JSON)."""
    g = {"gameEventType": kind, "period": period, "homeTeam": home, "setpieceType": setpiece}
    gid = next(_ids)
    pes = [{"possessionEventType": t, "passOutcomeType": outcome if t == "PA" else None} for t in types]
    base = {"gameEventId": gid, "startTime": t0, "endTime": t1, "gameEvents": g}
    return [{**base, "possessionEvents": p} for p in pes or [{}]]


def _flat(*evs):
    return [row for e in evs for row in e]


def test_events_labels():
    ev = _flat(
        _ge("FIRSTKICKOFF", 1, 1, True, "K", types=("PA",)),
        _ge("OTB", 1, 2, True, "K", types=("PA",)),
        _ge("OTB", 4, 5, False, "O"),  # A recupera
        _ge("OUT", 6, 6, types=()),
        _ge("OTB", 8, 9, True, "T", types=("PA",)),  # lateral: D desde o OUT, sem FAULT RECEIVED extra
        _ge("OTB", 10, 11, True, "O"),
        _ge("OTB", 13, 14, False, "F", types=("PA", "SH")),  # falta (2 linhas): D desde o fim do OTB (11)
        _ge("END", 16, 16, types=()),
    )
    df = events_frame(ev, fps=1.0)
    assert (df["type"] == "FAULT RECEIVED").sum() == 1
    lab = labels_from_events(df, 20)
    assert lab[0] == X and lab[2] == H and lab[4] == A
    assert (lab[6:8] == D).all() and lab[8] == H
    assert (lab[11:13] == D).all() and lab[13] == A and lab[17] == X


def test_disputa_de_penaltis_fica_fora():
    ev = _flat(
        _ge("FIRSTKICKOFF", 0, 0, True, "K", types=("PA",)),
        _ge("OTB", 0, 1, True, "K", types=("PA",)),
        _ge("END", 5, 5, types=()),
        _ge("OTB", 10, 10, True, "P", types=("SH",)),  # cobrança da disputa, mesmo período
        _ge("OTB", 12, 12, False, "P", types=("SH",)),
    )
    lab = labels_from_events(events_frame(ev, fps=1.0), 20)
    assert (lab[0:6] == H).all() and (lab[6:] == X).all()


def test_toques_sem_controle_nao_trocam():
    ev = _flat(
        _ge("FIRSTKICKOFF", 0, 0, True, "K", types=("PA",)),
        _ge("OTB", 0, 1, True, "K", types=("PA",)),
        _ge("OTB", 3, 3, False, "O", types=("CL",)),  # corte: não troca
        _ge("OTB", 5, 5, False, "O", types=("CH", "CL")),  # duelo + corte: não troca
        _ge("OTB", 7, 7, None, "O", types=("CH",)),  # duelo sem vencedor: não troca
        _ge("OTB", 9, 9, False, "O", types=("PA",), outcome="D"),  # passe de primeira incompleto: não
        _ge("OTB", 11, 12, True, "O"),
        _ge("OTB", 14, 14, False, "O", types=("PA",), outcome="C"),  # passe de primeira completo: troca
        _ge("OTB", 16, 16, True, "O", types=("CH", "PA"), outcome="D"),  # duelo + passe: troca
        _ge("END", 18, 18, types=()),
    )
    lab = labels_from_events(events_frame(ev, fps=1.0), 20)
    assert (lab[0:14] == H).all() and (lab[14:16] == A).all() and (lab[16:19] == H).all()


def test_parse_tracking_centers_dedups_and_estimated():
    def line(fr, ball, home):
        away = [{"jerseyNum": 9, "x": 1.0, "y": 1.0, "visibility": "ESTIMATED"}]
        return json.dumps({"frameNum": fr, "balls": ball, "homePlayers": home, "awayPlayers": away})

    home = [{"jerseyNum": 10, "x": -52.5, "y": 34.0, "visibility": "VISIBLE"}]
    lines = [
        line(1, [{"x": 0.0, "y": 0.0, "visibility": "VISIBLE"}], home),
        line(1, [{"x": 0.0, "y": 0.0, "visibility": "VISIBLE"}], home),  # duplicado
        line(2, [{"x": 1.0, "y": 1.0, "visibility": "ESTIMATED"}], []),
        line(3, [], [{"jerseyNum": 10, "x": None, "y": None}]),
    ]
    trk, ball_est, players_est = parse_tracking(lines, (105.0, 68.0))
    assert trk.ball.shape == (4, 2) and list(trk.team) == [HOME, AWAY]
    assert np.allclose(trk.ball[1], [52.5, 34.0]) and np.isnan(trk.ball[3]).all()
    assert np.allclose(trk.players[1, 0], [0.0, 68.0]) and np.allclose(trk.players[1, 1], [53.5, 35.0])
    assert np.isnan(trk.players[0]).all() and np.isnan(trk.players[[2, 3], 0]).all()
    assert list(ball_est) == [False, False, True, False]
    assert not players_est[:, 0].any() and list(players_est[:, 1]) == [False, True, True, True]
