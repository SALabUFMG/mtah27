"""Mede no jogo da competição como a bola some, sem rótulo, e grava só agregados em configs/.

Por faixa de velocidade mediana dos jogadores (posto dentro de cada tempo): fração de amostras sem bola
e quantis do comprimento das lacunas (ver src/mtah27/gaps.py). Roda na grade de c0.hz, dentro dos dois
tempos. Não usa os 5 min públicos nem nenhum rótulo.

Saída: configs/ball_gaps_fitted.yaml. Para o C1 usar, aponte c1.gaps_fitted para esse arquivo.

Uso: uv run python scripts/measure_ball_gaps.py
"""

# ruff: noqa: I001  (a ordem importa: mtah27.paths precisa vir antes de `games`)
from __future__ import annotations

import yaml

from mtah27 import gaps, observe
from mtah27.config import load_config
from mtah27.paths import CONFIGS_DIR
from mtah27.tracking import load_tracking
from games import GAMES, half_video_blocks  # type: ignore  # noqa: E402

NAME = "ball_gaps_fitted.yaml"


def main() -> None:
    cfg = load_config()
    game, hz = cfg.get("game", "BAR-ATH"), cfg["c0"]["hz"]
    trk = load_tracking(game)
    idx = observe.grid(len(trk.ball), GAMES[game]["fps"], hz)
    blocks = observe.grid_blocks(idx, [(a, b) for _, a, b in half_video_blocks(game)])
    fitted = gaps.fit(observe.subsample(trk, idx), hz, blocks)
    fitted["missing_rate"] = [round(v, 6) for v in fitted["missing_rate"]]
    text = "# Gerado por scripts/measure_ball_gaps.py. Não edite à mão.\n" + yaml.safe_dump(
        fitted, sort_keys=True
    )
    with open(CONFIGS_DIR / NAME, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"gravado configs/{NAME}: {fitted['n_bins']} faixas, {len(fitted['gap_len_q'][0])} quantis")


if __name__ == "__main__":
    main()
