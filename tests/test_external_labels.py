import numpy as np
import pandas as pd

from mtah27.external import A, D, H, X, labels_from_events
from mtah27.tracking import AWAY, HOME, parse_epts


def _ev(rows):
    return pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start_frame", "end_frame"])


def test_recovery_changes_possession():
    ev = _ev([("H", "PASS", "", 1, 10, 12), ("A", "RECOVERY", "", 1, 20, 20), ("A", "PASS", "", 1, 30, 35)])
    lab = labels_from_events(ev, 50)
    assert lab[15] == H and lab[25] == A and lab[40] == X


def test_ball_lost_and_challenge_keep_possession():
    ev = _ev(
        [
            ("H", "PASS", "", 1, 10, 12),
            ("H", "BALL LOST", "", 1, 20, 20),
            ("A", "CHALLENGE", "", 1, 22, 22),
            ("H", "PASS", "", 1, 40, 41),
        ]
    )
    lab = labels_from_events(ev, 50)
    assert (lab[10:41] == H).all()


def test_dead_ball_until_set_piece():
    ev = _ev(
        [
            ("H", "PASS", "", 1, 10, 12),
            ("H", "BALL OUT", "", 1, 14, 20),
            ("A", "SET PIECE", "THROW IN", 1, 30, 30),
            ("A", "PASS", "", 1, 31, 35),
        ]
    )
    lab = labels_from_events(ev, 50)
    assert lab[19] == H and (lab[20:30] == D).all() and lab[30] == A


def test_dead_ball_until_goal_kick_pass():
    ev = _ev(
        [
            ("H", "PASS", "", 1, 10, 12),
            ("H", "BALL OUT", "", 1, 14, 20),
            ("A", "PASS", "GOAL KICK", 1, 30, 32),
            ("H", "SET PIECE", "CORNER KICK", 1, 400, 400),
        ]
    )
    lab = labels_from_events(ev, 450)
    assert (lab[20:30] == D).all() and (lab[30:400] == A).all() and lab[400] == H


def test_x_outside_periods():
    ev = _ev(
        [
            ("H", "PASS", "", 1, 10, 12),
            ("H", "PASS", "", 1, 18, 20),
            ("A", "PASS", "", 2, 40, 45),
            ("A", "PASS", "", 2, 50, 55),
        ]
    )
    lab = labels_from_events(ev, 70)
    assert (lab[:10] == X).all() and (lab[21:40] == X).all() and (lab[56:] == X).all()
    assert lab[15] == H and lab[42] == A


def test_parse_epts(tmp_path):
    meta = tmp_path / "m.xml"
    meta.write_text(
        '<r><Team id="T1"/><Team id="T2"/><Player id="a" teamId="T1"/><Player id="b" teamId="T2"/></r>'
    )
    raw = tmp_path / "t.txt"
    raw.write_text("5:0.5,0.5;NaN:0.1,0.2\n6:0.1,0.2,9;0.3,0.4,9:NaN\n", newline="\n")
    trk = parse_epts(meta, raw, 3, (100.0, 50.0), first_frame_id=5)
    assert list(trk.team) == [HOME, AWAY]
    assert np.allclose(trk.players[0, 0], [50, 25])
    assert np.allclose(trk.players[1, 1], [30, 20])
    assert np.isnan(trk.ball[1]).all() and np.allclose(trk.ball[0], [10, 10])
    assert np.isnan(trk.players[2]).all()
