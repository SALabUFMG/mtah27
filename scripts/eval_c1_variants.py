"""Compara variantes do C1 em pares jogo x seed, no ponto fixo de c1_fitted.yaml.

Leave-one-game-out nos 3 jogos da Metrica para cada seed de degradação. Cada variante vale para treino
e teste; a primeira da lista é a referência das diferenças. Testes: "uniforme" (a degradação da
calibração) e "estresse" (lacunas só nos frames vivos, como em ablate_c1_ball.py). Com --gaps, treino e
teste "uniforme" trocam a degradação uniforme pela medida no alvo (configs/ball_gaps_fitted.yaml).
Não usa os 5 min públicos.

Variantes (VARIANTS): fill = fill_gap_s de features.window_features; n_aug = cópias degradadas por jogo
no treino; params = parâmetros extras do HistGradientBoosting.

Uso: uv run python scripts/eval_c1_variants.py --variants base fill1 [--seeds 0 1 2 3 4] [--gaps]
"""

# ruff: noqa: I001  (a ordem importa: calibrate_c0 importa mtah27.paths antes de possession_metrics)
from __future__ import annotations

import argparse
import time

import numpy as np

from calibrate_c0 import pm
from mtah27 import c1, external, features, observe
from mtah27.config import load_config

TEST = ("uniforme", "estresse")
VARIANTS = {
    "base": {},  # C1 da submissão
    "fill1": {"fill": 1.0},
    "fill2": {"fill": 2.0},
    "fill1_aug4": {"fill": 1.0, "n_aug": 4},
}


def prepare(trk, labels, hz, deg, window_s, n_aug, fill, gaps_fitted):
    """Features de treino (n_aug cópias) e dos dois testes de um jogo, com fill_gap_s = fill."""
    g = c1.game_features(
        trk, labels, 25, external.PITCH, hz, deg, window_s, n_aug, gaps_fitted=gaps_fitted, fill_gap_s=fill
    )
    gblocks = observe.grid_blocks(g["idx"], g["blocks"])
    sub = observe.subsample(trk, g["idx"])
    stress = observe.degrade(sub, **deg, fps=hz, blocks=c1.live_blocks(g["labels_grid"]))
    g["test"] = {
        "uniforme": g["feats"][0],
        "estresse": features.window_features(stress, external.PITCH, hz, gblocks, window_s, fill),
    }
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["base", "fill1"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--gaps", action="store_true", help="lacunas medidas no alvo no lugar das uniformes")
    args = ap.parse_args()
    t0 = time.time()
    cfg_c0 = load_config()["c0"]
    fitted_cfg = load_config(load_config()["c1"]["fitted"])
    hz, md, w = fitted_cfg["hz"], fitted_cfg["min_dur_s"], fitted_cfg["emission_weight"]
    raw = [external.load_sample_game(i) for i in (1, 2, 3)]
    gaps_fitted = load_config("ball_gaps_fitted.yaml") if args.gaps else None
    print(f"degradação de treino e do teste uniforme: {'alvo' if args.gaps else 'uniforme'}", flush=True)

    res = {(v, te): [] for v in args.variants for te in TEST}  # (spell, macro, D_prec, D_rec) por par
    for seed in args.seeds:
        deg = {**cfg_c0["degrade"], "seed": seed}
        for v in args.variants:
            var = VARIANTS[v]
            games = [
                prepare(
                    trk, labels, hz, deg, fitted_cfg["window_s"], var.get("n_aug", fitted_cfg["n_aug"]),
                    var.get("fill", 0.0), gaps_fitted,
                )
                for trk, labels in raw
            ]  # fmt: skip
            for i, g in enumerate(games):
                X, y = c1.train_rows(games[:i] + games[i + 1 :])
                fitted = c1.fit(X, y, seed=0, params=var.get("params"))
                for te in TEST:
                    pred = c1.decode_game(
                        g["test"][te], fitted, md, hz, g["idx"], g["blocks"], len(g["labels"]), w
                    )
                    ev = pm.evaluate(g["labels"], pred, bootstrap_n=1)
                    d = ev["per_class"]["D"]
                    res[v, te].append((ev["spell_f1_mean"], ev["macro_f1"], d["precision"], d["recall"]))
        print(f"seed {seed}: {time.time() - t0:.0f}s", flush=True)

    print(f"\npares jogo x seed: {3 * len(args.seeds)}")
    print("variante     teste      spellF1  macroF1  D_prec  D_rec")
    for (v, te), rows in res.items():
        m = np.mean(rows, axis=0)
        print(f"{v:11s}  {te:9s}  {m[0]:.4f}   {m[1]:.4f}   {m[2]:.3f}   {m[3]:.3f}")
    ref = args.variants[0]
    print(f"\ndiferença para {ref}:")
    for v in args.variants[1:]:
        for te in TEST:
            diff = np.array(res[v, te]) - np.array(res[ref, te])
            print(
                f"  {v:11s} teste {te:9s} spellF1 {diff[:, 0].mean():+.4f} ({(diff[:, 0] > 0).sum()} "
                f"vitórias)   macroF1 {diff[:, 1].mean():+.4f} ({(diff[:, 1] > 0).sum()} vitórias)"
            )
    print(f"tempo: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
