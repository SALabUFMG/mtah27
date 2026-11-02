import numpy as np

from mtah27.viterbi import STATES, viterbi

LOG_SWITCH = np.log(np.array([[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]]))


def one_hot_emit(seq: str, p: float = 0.8) -> np.ndarray:
    idx = np.array(["HAD".index(c) for c in seq])
    e = np.full((len(seq), 3), (1 - p) / 2)
    e[np.arange(len(seq)), idx] = p
    return np.log(e)


def test_segue_observacoes_limpas():
    seq = "H" * 20 + "A" * 20 + "D" * 20
    out = "".join(STATES[viterbi(one_hot_emit(seq), [5, 5, 5], LOG_SWITCH)])
    assert out == seq


def test_duracao_minima_apaga_blip():
    seq = "H" * 20 + "A" * 2 + "H" * 20
    out = "".join(STATES[viterbi(one_hot_emit(seq), [5, 5, 5], LOG_SWITCH)])
    assert out == "H" * 42


def test_respeita_duracao_minima():
    rng = np.random.default_rng(0)
    out = viterbi(np.log(rng.dirichlet(np.ones(3), size=500)), [7, 7, 7], LOG_SWITCH)
    runs = np.diff(np.flatnonzero(np.diff(np.r_[-1, out, -1])))
    assert runs.min() >= 7
