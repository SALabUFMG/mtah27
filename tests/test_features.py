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
    for n in ("ball_d_side", "ball_d_end", "home_cx_mid", "away_cx_mid", "home_dball", "away_dball",
              "home_cx_att", "away_cx_att", "home_cvx_att", "away_cvx_att", "ball_x_att"):  # fmt: skip
        _equal(x[:, COL[n]], y[:, COL[n]])


def test_attack_side_per_block():
    """Casa à esquerda no 1º bloco e à direita no 2º: o sentido de ataque vira com o bloco."""
    n = 40
    pl = np.zeros((n, 2, 2))
    pl[:, 0], pl[:, 1] = [30.0, 34.0], [70.0, 34.0]  # casa, visitante
    pl[20:, 0, 0], pl[20:, 1, 0] = 70.0, 30.0
    pl[:, 0, 0] += np.arange(n) * 0.2  # casa anda +x o jogo todo (1 m/s)
    ball = np.tile([60.0, 34.0], (n, 1))
    x = window_features(Tracking(ball, pl, np.array([HOME, AWAY])), PITCH, HZ, [(0, 19), (20, n - 1)])
    h1, h2 = x[5:15], x[25:35]
    for h in (h1, h2):  # nos dois blocos cada time está no próprio campo
        assert (h[:, COL["home_cx_att"]] < 0).all() and (h[:, COL["away_cx_att"]] < 0).all()
    np.testing.assert_allclose(h1[:, COL["home_cvx_att"]], 1.0, atol=1e-5)
    np.testing.assert_allclose(h2[:, COL["home_cvx_att"]], -1.0, atol=1e-5)
    np.testing.assert_allclose(h1[:, COL["ball_x_att"]], 7.5, atol=1e-5)
    np.testing.assert_allclose(h2[:, COL["ball_x_att"]], -7.5, atol=1e-5)


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
