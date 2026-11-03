import importlib

import numpy as np

from mtah27 import c0, paths

assert paths.OFFICIAL.exists()  # importar paths põe o repo oficial no sys.path
MISSING_P = 0.3
EMIT = np.array(
    [
        [0.35, 0.05, 0.20, 0.05, 0.05, MISSING_P],
        [0.05, 0.35, 0.05, 0.20, 0.05, MISSING_P],
        [0.10, 0.10, 0.15, 0.15, 0.20, MISSING_P],
    ]
)
MEMIT = np.array([[0.1, 0.3, 0.6], [0.1, 0.3, 0.6], [0.6, 0.3, 0.1]])  # D: jogadores parados
MIN_DUR = {"H": 1.0, "A": 1.0, "D": 2.0}


def _sample(probs, rng):
    return (rng.random(len(probs))[:, None] > np.cumsum(probs, axis=1)).sum(axis=1).astype(np.int8)


def synth(n=30000, seed=0):
    rng = np.random.default_rng(seed)
    labels = []
    while len(labels) < n:
        labels += [int(rng.integers(3))] * int(rng.integers(50, 300))
    labels = np.array(labels[:n], dtype=np.int8)
    return _sample(EMIT[labels], rng), _sample(MEMIT[labels], rng), labels


def test_fit_recupera_emissao_e_missing_neutro():
    obs, mot, labels = synth()
    p = c0.fit(obs, mot, labels, n_mot=3)
    em = np.array(p["emission"])
    assert np.abs(em[:2] - EMIT[:2]).max() < 0.05
    assert np.allclose(em[2], em[:2].mean(axis=0))
    assert np.abs(np.array(p["emission_motion"]) - MEMIT).max() < 0.05
    assert np.allclose(em.sum(axis=1), 1)
    assert np.ptp(em[:, c0.MISSING]) == 0
    assert np.allclose(np.array(p["transition"]).sum(axis=1), 1)
    assert p["n_frames"] == len(labels)
    assert p == c0.fit(obs, mot, labels, n_mot=3)


def test_decode_game_blocos_e_macro_f1():
    obs, mot, labels = synth()
    p = c0.fit(obs, mot, labels, n_mot=3)
    blocks = [(1000, 14999), (16000, 29000)]
    pred = c0.decode_game(obs, mot, p, MIN_DUR, 25, blocks)
    assert (pred[:1000] == 3).all() and (pred[15000:16000] == 3).all() and (pred[29001:] == 3).all()
    gt = np.full(len(labels), 3, dtype=np.int8)
    for a, b in blocks:
        gt[a : b + 1] = labels[a : b + 1]
    assert importlib.import_module("possession_metrics").evaluate(gt, pred)["macro_f1"] > 0.8
