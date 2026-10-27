"""Escrita do arquivo de submissão, no contrato exato do SUBMISSION.md oficial.

`frame,label`, 199.319 linhas, labels em {H, A, D, X}, frame 0..N-1 em ordem, LF.
"""

# ruff: noqa: I001  (a ordem importa: mtah27.paths precisa vir antes de `games`)
from __future__ import annotations

from pathlib import Path

import numpy as np

from mtah27 import paths  # noqa: F401  (configura sys.path para o repo oficial)
from games import GAME, GAMES  # type: ignore  # noqa: E402

VALID_LABELS = np.array(["H", "A", "D", "X"])


def n_frames(game: str = GAME) -> int:
    return int(GAMES[game]["n_video_frames"])


def empty_labels(game: str = GAME) -> np.ndarray:
    """Array de 'X' com o tamanho exato da submissão."""
    return np.full(n_frames(game), "X", dtype="<U1")


def write_submission(labels: np.ndarray, out_path: str | Path, game: str = GAME) -> Path:
    labels = np.asarray(labels).astype("<U1")
    n = n_frames(game)
    if labels.shape != (n,):
        raise ValueError(f"esperado {n} labels, recebido {labels.shape}")
    bad = ~np.isin(labels, VALID_LABELS)
    if bad.any():
        raise ValueError(f"labels inválidos: {sorted(set(labels[bad]))}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{lab}\n" for i, lab in enumerate(labels))
    return out_path
