"""Viterbi H/A/D com duração mínima por estado.

Cada estado s vira uma cadeia de m_s subestados: só se sai do último, então toda passagem por s dura
pelo menos m_s frames. Isso resolve a fragmentação que o spellF1 pune (uma posse quebrada em três
pedaços conta como erro).
"""

from __future__ import annotations

import numpy as np

STATES = np.array(["H", "A", "D"])


def viterbi(log_emit: np.ndarray, min_dur: list[int], log_switch: np.ndarray) -> np.ndarray:
    """Sequência de estados (índices em STATES) mais provável.

    log_emit:   [T, 3] log P(obs_t | estado)
    min_dur:    duração mínima de cada estado, em frames (>= 1)
    log_switch: [3, 3] log da transição saindo do último subestado; a diagonal é ficar.
    """
    T, n_states = log_emit.shape
    starts = np.concatenate(([0], np.cumsum(min_dur)[:-1]))
    lasts = starts + np.asarray(min_dur) - 1
    owner = np.repeat(np.arange(n_states), min_dur)  # subestado -> estado
    K = len(owner)

    delta = np.full(K, -np.inf)
    delta[starts] = log_emit[0]
    back = np.zeros((T, K), dtype=np.int32)
    chain = np.setdiff1d(np.arange(K), starts)  # subestados que só vêm do anterior na cadeia

    for t in range(1, T):
        new = np.full(K, -np.inf)
        bp = np.zeros(K, dtype=np.int32)
        new[chain] = delta[chain - 1]
        bp[chain] = chain - 1
        # saindo dos últimos subestados: ficar (diagonal) ou entrar no início de outro estado
        from_last = delta[lasts][:, None] + log_switch  # [de, para]
        stay = from_last[np.arange(n_states), np.arange(n_states)]
        better = stay > new[lasts]
        new[lasts] = np.where(better, stay, new[lasts])
        bp[lasts] = np.where(better, lasts, bp[lasts])
        enter = from_last.copy()
        np.fill_diagonal(enter, -np.inf)
        src = enter.argmax(axis=0)
        new[starts] = np.maximum(new[starts], enter[src, np.arange(n_states)])
        bp[starts] = np.where(new[starts] == enter[src, np.arange(n_states)], lasts[src], bp[starts])
        delta = new + log_emit[t][owner]
        back[t] = bp

    path = np.empty(T, dtype=np.int32)
    path[-1] = lasts[delta[lasts].argmax()]  # termina num estado completo
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return owner[path]
