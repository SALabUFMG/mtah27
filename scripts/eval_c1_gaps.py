"""Compara a degradação uniforme da bola (observe.degrade) com a medida no alvo (gaps.degrade_like).

Leave-one-game-out nos 3 jogos da Metrica, no ponto fixo de c1_fitted.yaml, para cada seed de
degradação. Tabela cruzada: treino {uniforme, alvo} x teste {uniforme, alvo, estresse}. "alvo" é a
ausência medida por scripts/measure_ball_gaps.py; "estresse" é o de ablate_c1_ball.py (lacunas só nos
frames vivos). A linha que importa é o teste "alvo": é a melhor imitação que temos do jogo da competição.
Não usa os 5 min públicos.

Uso: uv run python scripts/eval_c1_gaps.py [--seeds 0 1 2 3 4]
"""

# ruff: noqa: I001  (a ordem importa: calibrate_c0 importa mtah27.paths antes de possession_metrics)
from __future__ import annotations

import argparse
import time

import numpy as np

from calibrate_c0 import pm
from mtah27 import c1, external, features, observe
from mtah27.config import load_config

TRAIN = ("uniforme", "alvo")
TEST = ("uniforme", "alvo", "estresse")


def prepare(trk, labels, hz, deg, window_s, n_aug, gaps_fitted):
    """Features de treino e teste de um jogo nas duas degradações, mais o teste de estresse."""
    args = (trk, labels, 25, external.PITCH, hz, deg, window_s, n_aug)
    g = c1.game_features(*args)
    like = c1.game_features(*args, gaps_fitted=gaps_fitted)
    gblocks = observe.grid_blocks(g["idx"], g["blocks"])
    sub = observe.subsample(trk, g["idx"])
    stress = observe.degrade(sub, **deg, fps=hz, blocks=c1.live_blocks(g["labels_grid"]))
    g["train"] = {"uniforme": g["feats"], "alvo": like["feats"]}
    g["test"] = {
        "uniforme": g["feats"][0],
        "alvo": like["feats"][0],
        "estresse": features.window_features(stress, external.PITCH, hz, gblocks, window_s),
    }
    live = g["labels_grid"] < 2
    dead = g["labels_grid"] == 2
    j = features.FEATURES.index("ball_present")
    g["miss"] = {t: (1 - X[live, j].mean(), 1 - X[dead, j].mean()) for t, X in g["test"].items()}
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config()
    cfg_c0 = cfg["c0"]
    fitted_cfg = load_config(cfg["c1"]["fitted"])
    gaps_fitted = load_config(gaps_name := cfg["c1"].get("gaps_fitted") or "ball_gaps_fitted.yaml")
    hz, md, w = fitted_cfg["hz"], fitted_cfg["min_dur_s"], fitted_cfg["emission_weight"]
    raw = [external.load_sample_game(i) for i in (1, 2, 3)]
    print(f"carga {time.time() - t0:.0f}s; lacunas de {gaps_name}", flush=True)

    res = {(tr, te): [] for tr in TRAIN for te in TEST}  # (spell, macro, D_prec, D_rec) por jogo x seed
    miss = {te: [] for te in TEST}
    for seed in args.seeds:
        deg = {**cfg_c0["degrade"], "seed": seed}
        games = [
            prepare(trk, labels, hz, deg, fitted_cfg["window_s"], fitted_cfg["n_aug"], gaps_fitted)
            for trk, labels in raw
        ]
        for te in TEST:
            miss[te] += [g["miss"][te] for g in games]
        for tr in TRAIN:
            for i, g in enumerate(games):
                others = [{**o, "feats": o["train"][tr]} for o in games[:i] + games[i + 1 :]]
                fitted = c1.fit(*c1.train_rows(others), seed=0)
                for te in TEST:
                    pred = c1.decode_game(
                        g["test"][te], fitted, md, hz, g["idx"], g["blocks"], len(g["labels"]), w
                    )
                    ev = pm.evaluate(g["labels"], pred, bootstrap_n=1)
                    d = ev["per_class"]["D"]
                    res[tr, te].append((ev["spell_f1_mean"], ev["macro_f1"], d["precision"], d["recall"]))
        print(f"seed {seed}: {time.time() - t0:.0f}s", flush=True)

    print("\nfração sem bola nos jogos de teste (vivo / bola morta):")
    for te in TEST:
        m = np.mean(miss[te], axis=0)
        print(f"  {te:9s} {m[0]:.2f} / {m[1]:.2f}")
    print("\ntreino     teste      spellF1  macroF1  D_prec  D_rec")
    for (tr, te), rows in res.items():
        m = np.mean(rows, axis=0)
        print(f"{tr:9s}  {te:9s}  {m[0]:.4f}   {m[1]:.4f}   {m[2]:.3f}   {m[3]:.3f}")
    print(f"\ntreino alvo - treino uniforme, pares jogo x seed (n = {3 * len(args.seeds)}):")
    for te in TEST:
        diff = np.array(res["alvo", te]) - np.array(res["uniforme", te])
        print(
            f"  teste {te:9s} spellF1 {diff[:, 0].mean():+.4f} ({(diff[:, 0] > 0).sum()} vitórias)   "
            f"macroF1 {diff[:, 1].mean():+.4f} ({(diff[:, 1] > 0).sum()} vitórias)"
        )
    print(f"tempo: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
