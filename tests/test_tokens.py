import numpy as np

from mtah27.tokens import A, F, flip_tokens, velocity, window_tokens
from mtah27.tracking import AWAY, HOME, Tracking

PITCH = (100.0, 60.0)


def _trk(n=10, s=30, seed=0):
    rng = np.random.default_rng(seed)
    players = rng.uniform([0, 0], PITCH, (n, s, 2)).astype(np.float32)
    team = np.array([HOME] * (s // 2) + [AWAY] * (s - s // 2))
    ball = rng.uniform([0, 0], PITCH, (n, 2)).astype(np.float32)
    return Tracking(ball=ball, players=players, team=team)


def test_shapes_and_pad():
    trk = _trk()
    trk.ball[2] = np.nan
    trk.players[:, 20:] = np.nan  # visitante só com 5 slots
    tok, pad = window_tokens(trk, velocity(trk.ball, 5), velocity(trk.players, 5), 0, 8, PITCH)
    assert tok.shape == (8 * A, F) and pad.shape == (8 * A,)
    pad = pad.reshape(8, A)
    assert pad[2, 0] and not pad[3, 0]  # bola ausente vira pad
    assert not pad[:, 1:12].any()  # 11 mandantes escolhidos
    assert not pad[:, 12:17].any() and pad[:, 17:].all()  # 5 visitantes, resto pad
    assert np.all(tok.reshape(8, A, F)[pad] == 0)
    assert np.abs(tok[:, :2]).max() <= 1.0


def test_slots_follow_presence_and_x_order():
    trk = _trk(s=30)
    trk.players[:, 0] = np.nan  # slot do mandante sempre ausente: não entra
    tok, _ = window_tokens(trk, velocity(trk.ball, 5), velocity(trk.players, 5), 0, 10, PITCH)
    x = tok.reshape(10, A, F)[:, 1:12, 0].mean(axis=0)
    assert np.all(np.diff(x) >= 0)


def test_deterministic_and_flip():
    trk = _trk(seed=3)
    args = (velocity(trk.ball, 5), velocity(trk.players, 5), 1, 6, PITCH)
    a, _ = window_tokens(trk, *args)
    b, _ = window_tokens(trk, *args)
    assert np.array_equal(a, b)
    f = flip_tokens(a)
    assert np.array_equal(f[:, :4], -a[:, :4]) and np.array_equal(f[:, 4:], a[:, 4:])


def test_velocity_clips_jumps():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0]], np.float32)
    v = velocity(pos, 5)
    assert np.allclose(v[1], [5.0, 0.0]) and np.allclose(v[2], 0.0)
