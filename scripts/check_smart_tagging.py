"""Confronta uma submissão com o Smart Tagging do jogo da competição, só com agregados por código.

O Smart Tagging é automático e não é gabarito (DATA.md), mas é uma segunda opinião independente da
bola: cada fase tem um time, e fases de jogo acontecem com a bola viva. Para cada código, dentro dos
intervalos e dos tempos avaliados, imprime:

- a fração dos frames em que a submissão diz D, separando bola vista e bola ausente no ATD;
- nos frames em que a submissão diz H ou A, a fração em que o time previsto é o time da tag.

Serve para ver se o modelo chama de bola morta trechos que o tagging descreve como jogo, e qual é a
relação entre o time da tag e o time com a posse em cada código (em fases defensivas tende a ser o
oposto). Não usa os 5 min públicos e não escreve nada por frame.

Uso: uv run python scripts/check_smart_tagging.py [outputs/submissions/SALab_UFMG.csv ...]
"""

# ruff: noqa: I001  (a ordem importa: mtah27.paths precisa vir antes de `games`)
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mtah27.paths import SUBMISSIONS_DIR
from mtah27.tracking import load_tracking
from games import half_video_blocks  # type: ignore  # noqa: E402
from load_smart_tagging import load  # type: ignore  # noqa: E402

GAME = "BAR-ATH"
TEAM_SIDE = {"barcelona": "H", "atletic club": "A"}  # valores da tag Team (DATA.md); mandante = H


def tag_side(team: str | None) -> str:
    """H/A do time da tag; "" para N/A ou nome desconhecido."""
    return TEAM_SIDE.get((team or "").strip().lower(), "")


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or [SUBMISSIONS_DIR / "SALab_UFMG.csv"]
    trk = load_tracking(GAME)
    n = len(trk.ball)
    graded = np.zeros(n, bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a : b + 1] = True
    seen = ~np.isnan(trk.ball).any(axis=1)

    events = load(GAME)
    codes = sorted({e["code"] for e in events})
    cover = {c: np.zeros(n, bool) for c in codes}
    side = {c: np.full(n, "", dtype="<U1") for c in codes}
    for e in events:
        sl = slice(max(e["start_frame"], 0), min(e["end_frame"], n))
        cover[e["code"]][sl] = True
        side[e["code"]][sl] = tag_side(e["tags"].get("Team"))
    any_phase = np.any([cover[c] for c in codes if c not in ("SET PIECES", "GOALS")], axis=0)
    print(f"frames avaliados dentro de alguma fase (sem SET PIECES e GOALS): {any_phase[graded].mean():.3f}")

    for path in paths:
        pred = pd.read_csv(path)["label"].to_numpy()
        is_d = pred == "D"
        print(f"\n{path.name}: D em {is_d[graded].mean():.3f} dos frames avaliados")
        print(
            f"  D dentro de alguma fase: {is_d[graded & any_phase].mean():.3f}   "
            f"fora de toda fase: {is_d[graded & ~any_phase].mean():.3f}   "
            f"dos frames D, dentro de fase: {any_phase[graded & is_d].mean():.3f}"
        )
        print("  código                         cobertura  sem_bola  D|bola  D|sem_bola  time_previsto=tag")
        for c in codes:
            m = graded & cover[c]
            if not m.any():
                continue
            ha = m & ~is_d & (side[c] != "")
            agree = (pred[ha] == side[c][ha]).mean() if ha.any() else float("nan")
            d_seen = is_d[m & seen].mean() if (m & seen).any() else float("nan")
            d_miss = is_d[m & ~seen].mean() if (m & ~seen).any() else float("nan")
            print(
                f"  {c:30s} {m.sum() / graded.sum():.3f}      {(~seen)[m].mean():.3f}     {d_seen:.3f}   "
                f"{d_miss:.3f}       {agree:.3f}"
            )


if __name__ == "__main__":
    main()
