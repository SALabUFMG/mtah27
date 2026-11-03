import json

import numpy as np

from mtah27.external import A, D, H, X, labels_from_events
from mtah27.pff import events_frame, parse_tracking
from mtah27.tracking import AWAY, HOME


def _ge(kind, t0, t1, home=None, setpiece=None, period=1):
    g = {"gameEventType": kind, "period": period, "homeTeam": home, "setpieceType": setpiece}
    return {"startTime": t0, "endTime": t1, "gameEvents": g}


def test_events_labels():
    ev = [
        _ge("FIRSTKICKOFF", 1, 1, True, "K"),
        _ge("OTB", 1, 2, True, "K"),
        _ge("OTB", 4, 5, False, "O"),  # A recupera
        _ge("OUT", 6, 6),
        _ge("OTB", 8, 9, True, "T"),  # lateral: D desde o OUT, sem FAULT RECEIVED extra
        _ge("OTB", 10, 11, True, "O"),
        _ge("OTB", 13, 14, False, "F"),  # falta: D desde o fim do OTB anterior (11)
        _ge("END", 16, 16),
    ]
    df = events_frame(ev, fps=1.0)
    assert (df["type"] == "FAULT RECEIVED").sum() == 1
    lab = labels_from_events(df, 20)
    assert lab[0] == X and lab[2] == H and lab[4] == A
    assert (lab[6:8] == D).all() and lab[8] == H
    assert (lab[11:13] == D).all() and lab[13] == A and lab[17] == X


def test_parse_tracking_centers_and_dedups():
    def line(fr, ball, home):
        return json.dumps(
            {
                "frameNum": fr,
                "balls": ball,
                "homePlayers": home,
                "awayPlayers": [{"jerseyNum": 9, "x": 1.0, "y": 1.0}],
            }
        )

    lines = [
        line(1, [{"x": 0.0, "y": 0.0}], [{"jerseyNum": 10, "x": -52.5, "y": 34.0}]),
        line(1, [{"x": 0.0, "y": 0.0}], [{"jerseyNum": 10, "x": -52.5, "y": 34.0}]),  # duplicado
        line(3, [], [{"jerseyNum": 10, "x": None, "y": None}]),
    ]
    trk = parse_tracking(lines, (105.0, 68.0))
    assert trk.ball.shape == (4, 2) and list(trk.team) == [HOME, AWAY]
    assert np.allclose(trk.ball[1], [52.5, 34.0]) and np.isnan(trk.ball[3]).all()
    assert np.allclose(trk.players[1, 0], [0.0, 68.0]) and np.allclose(trk.players[1, 1], [53.5, 35.0])
    assert np.isnan(trk.players[[0, 2]]).all() and np.isnan(trk.players[3, 0]).all()
