"""Ponto de entrada do modelo: dados -> um label por frame.

Este é o único contrato que `scripts/make_submission.py` usa. A arquitetura do time entra aqui
(e em módulos irmãos dentro de src/mtah27/). Enquanto não há modelo, devolve tudo 'X' — uma
submissão válida que pontua 0, útil para testar o fluxo ponta a ponta.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mtah27.submission import empty_labels


def predict(config: dict[str, Any]) -> np.ndarray:
    """Retorna um array de 199.319 labels em {H, A, D, X}, indexado pelo frame do vídeo tático."""
    return empty_labels(config.get("game", "BAR-ATH"))
