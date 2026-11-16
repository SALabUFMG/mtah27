import numpy as np

from mtah27 import gaps
from mtah27.tracking import Tracking

HZ = 5
BLOCKS = [(50, 1999), (2100, 3999)]


def synth(seed=0, n=4000, miss_slow=0.8, miss_fast=0.2):
    """Jogadores alternam trechos parados e rápidos; a bola some mais nos parados."""
    rng = np.random.default_rng(seed)
    fast = (np.arange(n) // 200) % 2 == 1
    step = np.where(fast, 1.0, 0.05)[:, None, None] * rng.normal(size=(n, 6, 2))
    players = np.cumsum(step, axis=0).astype(np.float32)
    ball = rng.normal(size=(n, 2)).astype(np.float32)
    ball[rng.random(n) < np.where(fast, miss_fast, miss_slow)] = np.nan
    return Tracking(ball=ball, players=players, team=np.array([0, 0, 0, 1, 1, 1]))


def test_fit_separa_faixas_lentas_e_rapidas():
    fitted = gaps.fit(synth(), HZ, BLOCKS, n_bins=4)
    rate = fitted["missing_rate"]
    assert len(rate) == 4 and len(fitted["gap_len_q"]) == 4
    assert rate[0] > 0.6 and rate[-1] < 0.4  # faixa mais lenta perde mais a bola
    assert all(q[0] >= 1 for q in fitted["gap_len_q"])


def test_degrade_like_reproduz_fracoes_e_e_deterministico():
    fitted = gaps.fit(synth(), HZ, BLOCKS, n_bins=4)
    other = synth(seed=1, miss_slow=0.0, miss_fast=0.0)  # outro jogo, bola sempre presente
    out = gaps.degrade_like(other, fitted, HZ, seed=3, blocks=BLOCKS)
    again = gaps.degrade_like(other, fitted, HZ, seed=3, blocks=BLOCKS)
    assert np.array_equal(np.isnan(out.ball), np.isnan(again.ball))
    assert not np.isnan(other.ball).any()  # a entrada não muda
    bins = gaps.speed_bins(other, HZ, BLOCKS, 4)
    gone = np.isnan(out.ball).any(axis=1)
    assert not gone[bins < 0].any()  # nada apagado fora dos blocos
    got = [gone[bins == b].mean() for b in range(4)]
    assert np.allclose(got, fitted["missing_rate"], atol=0.08)
