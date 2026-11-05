import importlib

import numpy as np
import pytest

from mtah27 import c1, paths
from mtah27.observe import grid

assert paths.OFFICIAL.exists()  # importar paths põe o repo oficial no sys.path
MIN_DUR = {"H": 1.0, "A": 1.0, "D": 2.0}
FAST = {"max_iter": 30}


def synth(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    labels = []
    while len(labels) < n:
        labels += [int(rng.integers(3))] * int(rng.integers(50, 300))
    labels = np.array(labels[:n], dtype=np.int8)
    X = rng.normal(size=(n, 4)) * 1.5
    X[np.arange(n), labels] += 2.0  # coluna k carrega o estado k, com ruído
    X[rng.random(X.shape) < 0.1] = np.nan
    return X, labels


def test_decode_deterministico_blocos_e_macro_f1():
    X, labels = synth()
    idx = grid(len(labels), 25, 5)
    f1 = c1.fit(X[idx], labels[idx], params=FAST)
    f2 = c1.fit(X[idx], labels[idx], params=FAST)
    le = c1.log_emission(f1, X[idx])
    assert le.shape == (len(idx), 3) and np.isfinite(le).all()
    assert np.array_equal(le, c1.log_emission(f2, X[idx]))

    blocks = [(100, 2999), (3500, 5900)]
    pred = c1.decode_game(X[idx], f1, MIN_DUR, 5, idx, blocks, len(labels))
    assert pred.dtype == np.int8 and len(pred) == len(labels)
    assert (pred[:100] == 3).all() and (pred[3000:3500] == 3).all() and (pred[5901:] == 3).all()
    assert (pred[100:3000] != 3).all() and (pred[3500:5901] != 3).all()
    gt = np.full(len(labels), 3, dtype=np.int8)
    for a, b in blocks:
        gt[a : b + 1] = labels[a : b + 1]
    assert importlib.import_module("possession_metrics").evaluate(gt, pred)["macro_f1"] > 0.8


def test_falta_classe_da_erro_claro():
    X, labels = synth(n=600)
    y = np.where(labels == 2, 3, labels)  # D vira X: some do treino
    with pytest.raises(ValueError, match="3 classes"):
        c1.fit(X, y, params=FAST)
