import time

import numpy as np

from mtah27.features import FEATURES, window_features
from mtah27.tracking import AWAY, HOME, Tracking

PITCH = (105.0, 68.0)
HZ = 5.0
COL = {n: i for i, n in enumerate(FEATURES)}


def _rand(n=200, s=10, seed=0):
    rng = np.random.default_rng(seed)
    pl = rng.uniform([0, 0], PITCH, size=(n, s, 2))
    pl[rng.random((n, s)) < 0.1] = np.nan
    ball = rng.uniform([0, 0], PITCH, size=(n, 2))
    ball[rng.random(n) < 0.2] = np.nan
    team = np.array([HOME] * (s // 2) + [AWAY] * (s - s // 2))
    return Tracking(ball=ball, players=pl, team=team)


def _run(trk, pitch=PITCH):
    return window_features(trk, pitch, HZ, [(0, len(trk.ball) - 1)])


def _equal(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_shape_dtype_no_inf():
    x = _run(_rand())
    assert x.shape == (200, len(FEATURES)) and x.dtype == np.float32
    assert not np.isinf(x).any()


def test_slot_permutation_invariant():
    t = _rand()
    perm = np.random.default_rng(1).permutation(t.players.shape[1])
    _equal(_run(t), _run(Tracking(t.ball, t.players[:, perm], t.team[perm])))


def test_swap_teams_swaps_columns():
    t = _rand()
    x = _run(t)
    y = _run(Tracking(t.ball, t.players, 1 - t.team))
    for name in FEATURES:
        if name.startswith("home_"):
            _equal(x[:, COL[name]], y[:, COL["away_" + name[5:]]])


def test_ball_absent():
    t = _rand()
    t.ball[:] = np.nan
    x = _run(t)
    assert (x[:, COL["ball_present"]] == 0).all()
    for n in ("home_d1", "away_d1", "home_d2", "d_diff", "ball_d_side", "ball_out"):
        assert np.isnan(x[:, COL[n]]).all()


def test_nearest_distances_and_still_ball():
    n = 20
    ball = np.tile([50.0, 30.0], (n, 1))
    pl = np.tile(np.array([[51.0, 30.0], [55.0, 30.0], [10.0, 10.0]]), (n, 1, 1))
    x = _run(Tracking(ball, pl, np.array([HOME, AWAY, HOME])))
    mid = x[5:-5]
    np.testing.assert_allclose(mid[:, COL["home_d1"]], 1.0, atol=1e-5)
    np.testing.assert_allclose(mid[:, COL["away_d1"]], 5.0, atol=1e-5)
    np.testing.assert_allclose(mid[:, COL["ball_speed"]], 0.0, atol=1e-5)
    assert (mid[:, COL["home_n3"]] == 1).all() and (mid[:, COL["away_n8"]] == 1).all()


def test_mirror_x_keeps_line_distances():
    t = _rand()
    m = Tracking(t.ball * [-1, 1] + [PITCH[0], 0], t.players * [-1, 1] + [PITCH[0], 0], t.team)
    x, y = _run(t), _run(m)
    for n in ("ball_d_side", "ball_d_end", "home_cx_mid", "away_cx_mid", "home_dball", "away_dball"):
        _equal(x[:, COL[n]], y[:, COL[n]])


def test_speed_30000x46():
    t = _rand(30000, 46)
    t0 = time.perf_counter()
    x = _run(t)
    assert time.perf_counter() - t0 < 5 and x.shape[0] == 30000


def test_team_speed_ignores_appearing_slot():
    n = 40
    pl = np.full((n, 4, 2), np.nan)
    t = np.arange(n) / HZ
    for i in range(3):
        pl[:, i] = np.c_[10 + 5 * i + t, np.full(n, 20.0 + i)]
    pl[n // 2 :, 3] = np.c_[
        90.0 + t[n // 2 :], np.full(n - n // 2, 60.0)
    ]  # aparece de repente longe do centróide
    x = _run(Tracking(np.full((n, 2), np.nan), pl, np.array([HOME] * 4)))
    np.testing.assert_allclose(x[5:-5, COL["home_cspeed"]], 1.0, atol=1e-4)
    assert np.isnan(x[:, COL["away_cspeed"]]).all()


def test_fill_short_gaps_interpola_so_lacunas_curtas():
    from mtah27.features import fill_short_gaps

    ball = np.array([[np.nan] * 2, [0, 0], [np.nan] * 2, [np.nan] * 2, [3, 6], *[[np.nan] * 2] * 4, [9, 9]])
    out, gap, since = fill_short_gaps(ball.astype(np.float64), max_len=2)
    assert np.allclose(out[2:4], [[1, 2], [2, 4]])  # lacuna de 2: linha reta entre as pontas
    assert np.isnan(out[5:9]).all() and np.isnan(out[0]).all()  # lacuna de 4 e borda ficam vazias
    assert np.array_equal(gap[1:], [0, 2, 2, 0, 4, 4, 4, 4, 0]) and np.isnan(gap[0])
    assert np.array_equal(since[1:], [0, 1, 2, 0, 1, 2, 3, 4, 0])
    assert np.isnan(ball[2]).all()  # a entrada não muda
