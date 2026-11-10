"""Caminho C0: emissão categórica P(obs|estado) + transição, decodificados por Viterbi H/A/D.

Dois canais, independentes dado o estado: a observação da bola (observe.OBS) e a faixa de velocidade
dos jogadores (observe.speed_bin), que sinaliza bola morta mesmo quando a bola some.
"""

from __future__ import annotations

import numpy as np

from mtah27.viterbi import viterbi

# Espelham observe.OBS = (home_close, away_close, home_far, away_far, out, missing)
N_OBS = 6
MISSING = 5


def fit(obs: np.ndarray, mot: np.ndarray, labels: np.ndarray, n_mot: int, alpha: float = 1.0) -> dict:
    """Estima emissão da bola [3,6], do movimento [3,n_mot] e transição [3,3] em frames H/A/D (X ignorado).

    A coluna "missing" é neutra: vale o P(missing) agregado em todos os frames rotulados, igual nos
    3 estados. O alvo tem ~41% de bola ausente, bem mais que os jogos externos, então a ausência
    não pode carregar informação de estado. As demais colunas são renormalizadas para a linha somar 1.
    """
    obs = np.asarray(obs)
    labels = np.asarray(labels)
    lab = labels < 3
    counts = np.zeros((3, N_OBS))
    np.add.at(counts, (labels[lab], obs[lab]), 1)
    emit = counts + alpha
    emit /= emit.sum(axis=1, keepdims=True)

    p_missing = (counts[:, MISSING].sum() + alpha) / (counts.sum() + alpha * N_OBS)
    others = np.arange(N_OBS) != MISSING
    emit[:, others] *= ((1 - p_missing) / emit[:, others].sum(axis=1))[:, None]
    emit[:, MISSING] = p_missing
    # D não informa a bola: nos externos a bola quase não é vista em bola morta, e o pouco que se vê
    # é artefato (bola parada). D = média de H e A, então só o canal de movimento separa D de vivo.
    emit[2] = emit[:2].mean(axis=0)

    mcounts = np.zeros((3, n_mot))
    np.add.at(mcounts, (labels[lab], np.asarray(mot)[lab]), 1)
    memit = (mcounts + alpha) / (mcounts + alpha).sum(axis=1, keepdims=True)

    return {
        "emission": emit.tolist(),
        "emission_motion": memit.tolist(),
        "transition": fit_transition(labels, alpha).tolist(),
        "n_frames": int(lab.sum()),
    }


def fit_transition(labels: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Transição [3,3] entre H/A/D estimada em pares consecutivos de frames rotulados (X ignorado)."""
    labels = np.asarray(labels)
    lab = labels < 3
    both = lab[1:] & lab[:-1]
    trans = np.zeros((3, 3))
    np.add.at(trans, (labels[:-1][both], labels[1:][both]), 1)
    leave = trans.copy()
    np.fill_diagonal(leave, 0)
    stay = np.diag(trans) + alpha
    out = leave.sum(axis=1) + alpha
    p_leave = out / (out + stay)
    # P(trocar para j | sair de i), suavizado nas off-diagonais
    to = leave + alpha
    np.fill_diagonal(to, 0)
    to /= to.sum(axis=1, keepdims=True)
    T = to * p_leave[:, None]
    np.fill_diagonal(T, 1 - p_leave)
    return T


def to_native(dec: np.ndarray, idx_block: np.ndarray, a: int, b: int) -> np.ndarray:
    """Rótulos da grade (dec, em idx_block, nativos ordenados) para os frames a..b, pelo mais próximo."""
    f = np.arange(a, b + 1)
    right = np.searchsorted(idx_block, f).clip(0, len(idx_block) - 1)
    left = (right - 1).clip(0)
    nearest = np.where(f - idx_block[left] <= np.abs(idx_block[right] - f), left, right)
    return dec[nearest]


def decode(
    obs: np.ndarray, mot: np.ndarray, params: dict, min_dur_s: dict[str, float], fps: float
) -> np.ndarray:
    """Labels 0..2 (H,A,D) por Viterbi; min_dur_s dá a duração mínima por estado em segundos."""
    log_emit = np.log(np.asarray(params["emission"]))[:, np.asarray(obs)].T
    log_emit = log_emit + np.log(np.asarray(params["emission_motion"]))[:, np.asarray(mot)].T
    min_dur = [max(1, round(min_dur_s[s] * fps)) for s in "HAD"]
    log_switch = np.log(np.asarray(params["transition"]))
    return viterbi(log_emit, min_dur, log_switch).astype(np.int8)


def decode_game(
    obs: np.ndarray,
    mot: np.ndarray,
    params: dict,
    min_dur_s: dict[str, float],
    hz: float,
    idx: np.ndarray,
    blocks: list[tuple[int, int]],
    n: int,
) -> np.ndarray:
    """Decodifica na grade e devolve [n] rótulos nativos; X (3) fora dos blocos.

    obs/mot estão na grade de hz (amostra k = frame nativo idx[k]); blocks são nativos [first, last]
    (inclusivos). Cada bloco é decodificado separadamente e cada frame nativo dele recebe o rótulo
    da amostra mais próxima do mesmo bloco.
    """
    out = np.full(n, 3, dtype=np.int8)
    for a, b in blocks:
        ks = np.flatnonzero((idx >= a) & (idx <= b))
        if not len(ks):
            continue
        dec = decode(obs[ks], mot[ks], params, min_dur_s, hz)
        out[a : b + 1] = to_native(dec, idx[ks], a, b)
    return out
