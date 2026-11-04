import numpy as np

from mtah27.observe import (
    OBS,
    degrade,
    grid,
    grid_blocks,
    normalize_speed,
    observe,
    player_speed,
    speed_bin,
)
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
    trk.ball[:20_000] = np.nan  # pré-jogo sem bola: não conta para a meta
    ball0 = trk.ball.copy()
    blocks = [(20_000, 59_999), (70_000, 99_999)]
    a = degrade(trk, 0.3, 2.0, 25, seed=1, blocks=blocks)
    b = degrade(trk, 0.3, 2.0, 25, seed=1, blocks=blocks)
    assert np.array_equal(a.ball, b.ball, equal_nan=True)
    gone = np.isnan(a.ball).any(axis=1)
    assert abs(np.r_[gone[20_000:60_000], gone[70_000:]].mean() - 0.3) < 0.01
    assert not gone[60_000:70_000].any()  # fora dos blocos: intacto
    assert np.array_equal(trk.ball, ball0, equal_nan=True)  # entrada intacta
    c = degrade(trk, 0.3, 2.0, 25, seed=2, blocks=blocks)
    assert not np.array_equal(a.ball, c.ball, equal_nan=True)


def test_player_speed_e_faixas():
    n = 100
    pl = np.zeros((n, 3, 2), dtype=np.float32)
    pl[:, :, 0] = np.arange(n)[:, None] * 0.08  # 2 m/s a 25 fps
    pl[50:, :, 0] = pl[49, 0, 0]  # todos param no frame 50
    pl[30, 2, 0] = 999  # salto de troca de identidade: ignorado
    pl[70:, 1] = np.nan  # slot some
    trk = Tracking(ball=np.zeros((n, 2), np.float32), players=pl, team=np.array([0, 0, 1], np.int8))
    v = player_speed(trk, fps=25, window_s=0.2)
    assert np.allclose(v[10:25], 2.0, atol=1e-4) and np.allclose(v[60:], 0.0, atol=1e-4)
    assert list(speed_bin(np.array([0.5, 1.2, 3.0, np.nan]), [1.0, 1.5])) == [0, 1, 2, 2]


def test_grade_e_blocos():
    assert list(grid(11, 25, 5)) == [0, 5, 10]
    assert list(grid(40, 29.97, 5)) == [0, 6, 12, 18, 24, 30, 36]  # passo de 5,994 frames
    idx = np.array([0, 5, 10, 15])
    assert grid_blocks(idx, [(3, 12), (16, 18), (15, 20)]) == [(1, 2), (3, 3)]


def test_normalize_speed_por_bloco():
    s = np.array([1.0, 2.0, 3.0, 9.0, 4.0, 8.0, np.nan], dtype=np.float32)
    got = normalize_speed(s, [(0, 2), (4, 6)])
    assert np.allclose(got[:3], [0.5, 1.0, 1.5]) and np.isnan(got[3])
    assert np.allclose(got[4:6], [4 / 6, 8 / 6]) and np.isnan(got[6])
