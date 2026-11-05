"""Calibra o C1 (emissão GBM + Viterbi) em leave-one-game-out nos 3 jogos da Metrica.

Por jogo de teste o modelo é treinado uma vez (seed 0, params default) nos outros jogos
(+ os 64 da PFF só como treino extra com --extra-train pff). A grade (duração mínima H/A e D, peso da
emissão) varre só a decodificação, sem retreinar. Treino usa n_aug cópias degradadas por jogo; a avaliação
usa só a cópia com a seed base, como no C0. Não usa os 5 min públicos. O cv_spell_f1 gravado é o máximo da
grade, então é otimista (viés de seleção): serve para comparar variantes, não como placar.

Saída: configs/c1_fitted.yaml (ou c1_fitted_pff.yaml). Não grava o modelo treinado.

Uso: uv run python scripts/calibrate_c1.py [--extra-train pff]
"""

# ruff: noqa: I001  (a ordem importa: calibrate_c0 importa mtah27.paths antes de possession_metrics)
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import yaml

from calibrate_c0 import load_games, pm, rnd
from mtah27 import c1, features
from mtah27.config import load_config
from mtah27.paths import CONFIGS_DIR

C0_REF = 0.4952
KEYS = (
    "spell_f1_mean",
    "macro_f1",
    "n_pred_spells",
    "n_gt_spells",
    "mean_spell_duration_s",
    "gt_mean_spell_duration_s",
)


def prepare(source, cfg_c0, window_s, n_aug, drop=()):
    """Por jogo: features das n_aug cópias degradadas, rótulos e índices. O tracking é descartado."""
    return [
        c1.game_features(trk, labels, fps, pitch, cfg_c0["hz"], cfg_c0["degrade"], window_s, n_aug)
        for trk, labels, fps, pitch in load_games(source, drop)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["metrica"], default="metrica")
    ap.add_argument("--extra-train", choices=["pff"], default=None)
    ap.add_argument("--drop-estimated", nargs="*", choices=["ball", "players"], default=["ball"])
    args = ap.parse_args()

    t0 = time.time()
    cfg_all = load_config()
    cfg_c0, cfg = cfg_all["c0"], cfg_all["c1"]
    hz, window_s, n_aug, grid = cfg_c0["hz"], cfg["window_s"], cfg["n_aug"], cfg["grid"]
    games = prepare(args.source, cfg_c0, window_s, n_aug)
    print(f"metrica: {len(games)} jogos, carga {time.time() - t0:.0f}s", flush=True)
    extra = []
    if args.extra_train:
        extra = prepare("pff", cfg_c0, window_s, n_aug, tuple(sorted(args.drop_estimated)))
        print(f"pff (treino extra): {len(extra)} jogos, carga {time.time() - t0:.0f}s", flush=True)

    points = list(itertools.product(grid["min_dur_ha_s"], grid["min_dur_d_s"], grid["emission_weight"]))
    results = {p: ([], [], [], [], [], []) for p in points}
    train_s = 0.0
    for i, g in enumerate(games):
        t1 = time.time()
        fitted = c1.fit(*c1.train_rows(games[:i] + games[i + 1 :] + extra), seed=0)
        train_s += time.time() - t1
        print(f"jogo {i + 1}: treino {time.time() - t1:.0f}s ({time.time() - t0:.0f}s)", flush=True)
        for p in points:
            ha, d, w = p
            pred = c1.decode_game(
                g["feats"][0], fitted, {"H": ha, "A": ha, "D": d}, hz, g["idx"], g["blocks"],
                len(g["labels"]), emission_weight=w,
            )  # fmt: skip
            ev = pm.evaluate(g["labels"], pred, bootstrap_n=1)
            for lst, k in zip(results[p], KEYS, strict=True):
                lst.append(ev[k])

    print(f"\nsklearn (treino total {train_s:.0f}s)")
    print("ha_s  d_s   w    spellF1  macroF1  spellF1 por jogo        n_pred/n_gt  dur_prev/dur_gt (s)")
    rows = []
    for (ha, d, w), (sp, mf, npd, ngt, dp, dg) in results.items():
        rows.append((ha, d, w, float(np.mean(sp)), float(np.mean(mf))))
        per = "/".join(f"{v:.4f}" for v in sp)
        print(
            f"{ha:4g}  {d:3g}  {w:3g}   {rows[-1][3]:.4f}   {rows[-1][4]:.4f}   {per:22s}  "
            f"{np.mean(npd):.0f}/{np.mean(ngt):.0f}  {np.mean(dp):.1f}/{np.mean(dg):.1f}"
        )
    best = max(rows, key=lambda t: (round(t[3], 6), t[0], t[1], -t[2]))  # empate: maior min_dur
    print(f"melhor: ha={best[0]:g} d={best[1]:g} w={best[2]:g} spellF1={best[3]:.4f} macroF1={best[4]:.4f}")
    print(f"referência: C0 = {C0_REF}")

    fitted_cfg = {
        "min_dur_s": {"H": best[0], "A": best[0], "D": best[1]},
        "emission_weight": best[2],
        "cv_spell_f1": round(best[3], 4),
        "cv_macro_f1": round(best[4], 4),
        "hz": hz,
        "window_s": window_s,
        "n_aug": n_aug,
        "extra_train": args.extra_train or "none",
        "features": list(features.FEATURES),
    }
    text = "# Gerado por scripts/calibrate_c1.py. Não edite à mão.\n" + yaml.safe_dump(
        rnd(fitted_cfg), sort_keys=True
    )
    name = "c1_fitted_pff.yaml" if args.extra_train else "c1_fitted.yaml"
    with open(CONFIGS_DIR / name, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"gravado configs/{name}; tempo: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
