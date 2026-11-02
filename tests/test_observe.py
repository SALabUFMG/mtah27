import numpy as np

from mtah27.observe import OBS, degrade, observe
from mtah27.tracking import AWAY, HOME, Tracking

PITCH = (100.0, 60.0)


def _trk():
    nan = np.nan
    # slots: 0 = home, 1 = away
    ball = np.array(
        [[10, 10], [10, 10], [10, 10], [10, 10], [150, 10], [nan, nan], [10, 10], [-0.5, 10]],
        dtype=np.float32,
    )
    pl = np.full((8, 2, 2), nan, dtype=np.float32)
    pl[0] = [[11, 10], [40, 40]]  # home perto
    pl[1] = [[40, 40], [10, 12]]  # away perto
    pl[2] = [[20, 10], [40, 40]]  # home longe
    pl[3] = [[40, 40], [10, 20]]  # away longe
    pl[4] = [[11, 10], [40, 40]]  # bola fora
    pl[5] = [[11, 10], [40, 40]]  # bola NaN
    # frame 6: nenhum jogador
    pl[7] = [[11, 10], [40, 40]]  # dentro da margem de 1 m
    return Tracking(ball=ball, players=pl, team=np.array([HOME, AWAY], dtype=np.int8))


def test_observe_categories():
    got = observe(_trk(), r_close_m=5.0, pitch=PITCH)
    want = ["home_close", "away_close", "home_far", "away_far", "out", "missing", "missing", "home_far"]
    assert got.dtype == np.int8
    assert [OBS[i] for i in got] == want


def test_degrade_deterministic_rate_and_input():
    n = 100_000
    rng = np.random.default_rng(0)
    trk = Tracking(
        ball=rng.random((n, 2)).astype(np.float32),
        players=rng.random((n, 4, 2)).astype(np.float32),
        team=np.array([0, 0, 1, 1], dtype=np.int8),
    )
    ball0 = trk.ball.copy()
    a = degrade(trk, 0.3, 2.0, 25, seed=1)
    b = degrade(trk, 0.3, 2.0, 25, seed=1)
    assert np.array_equal(a.ball, b.ball, equal_nan=True)
    assert abs(np.isnan(a.ball).any(axis=1).mean() - 0.3) < 0.02
    assert np.array_equal(trk.ball, ball0)  # entrada intacta
    assert not np.array_equal(a.ball, degrade(trk, 0.3, 2.0, 25, seed=2).ball, equal_nan=True)
