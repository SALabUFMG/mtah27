"""Cruza fontes de treino e de teste do C1: Metrica (rótulo por evento) e DFL (rótulo do tracking).

Para cada fonte de treino (metrica, dfl, ambos) e cada jogo de teste, treina sem o jogo de teste e
avalia no ponto fixo de c1_fitted.yaml. Responde: (1) os dois tipos de rótulo são compatíveis? (2) somar
a DFL ajuda nos jogos da Metrica? (3) com 10 jogos, a interpolação da bola (--fill) continua ganhando?
A bola é degradada com c0.degrade em todos os jogos. Não usa os 5 min públicos.

Uso: uv run python scripts/eval_c1_sources.py [--fill 0 1]
"""

# ruff: noqa: I001  (a ordem importa: calibrate_c0 importa mtah27.paths antes de possession_metrics)
from __future__ import annotations

import argparse
import time

import numpy as np

from calibrate_c0 import load_games, pm
from mtah27 import c1
from mtah27.config import load_config

SOURCES = ("metrica", "dfl")
TRAIN = ("metrica", "dfl", "ambos")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", nargs="*", type=float, default=[0.0, 1.0])
    args = ap.parse_args()
    t0 = time.time()
    cfg_c0 = load_config()["c0"]
    fitted_cfg = load_config(load_config()["c1"]["fitted"])
    hz, md, w = fitted_cfg["hz"], fitted_cfg["min_dur_s"], fitted_cfg["emission_weight"]

    res = {}  # (fill, treino, fonte do teste) -> [(spell, macro, D_prec, D_rec, n_pred, n_gt)] por jogo
    for fill in args.fill:
        games = []  # (fonte, features do jogo)
        for src in SOURCES:
            for trk, labels, fps, pitch in load_games(src):
                g = c1.game_features(
                    trk, labels, fps, pitch, hz, cfg_c0["degrade"], fitted_cfg["window_s"],
                    fitted_cfg["n_aug"], fill_gap_s=fill,
                )  # fmt: skip
                games.append((src, g))
        if fill == args.fill[0]:
            describe(games)
        for tr in TRAIN:
            pool = [k for k, (src, _) in enumerate(games) if tr in (src, "ambos")]
            full = None  # modelo com a fonte inteira, para os jogos de teste que não estão nela
            for k, (src, g) in enumerate(games):
                if k in pool:
                    fitted = c1.fit(*c1.train_rows([games[j][1] for j in pool if j != k]), seed=0)
                else:
                    full = full or c1.fit(*c1.train_rows([games[j][1] for j in pool]), seed=0)
                    fitted = full
                pred = c1.decode_game(
                    g["feats"][0], fitted, md, hz, g["idx"], g["blocks"], len(g["labels"]), w
                )
                ev = pm.evaluate(g["labels"], pred, bootstrap_n=1)
                d = ev["per_class"]["D"]
                res.setdefault((fill, tr, src), []).append(
                    (ev["spell_f1_mean"], ev["macro_f1"], d["precision"], d["recall"],
                     ev["n_pred_spells"], ev["n_gt_spells"])
                )  # fmt: skip
            print(f"fill {fill:g} treino {tr}: {time.time() - t0:.0f}s", flush=True)

    print("\nfill_s  treino   teste    jogos  spellF1  macroF1  D_prec  D_rec   n_pred/n_gt")
    for (fill, tr, src), rows in res.items():
        m = np.mean(rows, axis=0)
        print(
            f"{fill:5g}   {tr:7s}  {src:7s}  {len(rows):3d}    {m[0]:.4f}   {m[1]:.4f}   {m[2]:.3f}   "
            f"{m[3]:.3f}   {m[4]:.0f}/{m[5]:.0f}"
        )
    if len(args.fill) > 1:
        base = args.fill[0]
        print(f"\ndiferença para fill_s = {base:g}, por jogo de teste:")
        for fill in args.fill[1:]:
            for tr in TRAIN:
                diff = np.concatenate(
                    [np.array(res[fill, tr, s]) - np.array(res[base, tr, s]) for s in SOURCES]
                )
                print(
                    f"  fill {fill:g} treino {tr:7s} spellF1 {diff[:, 0].mean():+.4f} "
                    f"({(diff[:, 0] > 0).sum()}/{len(diff)} vitórias)   macroF1 {diff[:, 1].mean():+.4f} "
                    f"({(diff[:, 1] > 0).sum()}/{len(diff)} vitórias)"
                )
    print(f"tempo: {time.time() - t0:.0f}s")


def describe(games):
    """Agregados dos rótulos por fonte: fração de cada estado, posses por jogo e duração média."""
    print("fonte    jogos  H      A      D      posses/jogo  duração média (s)")
    for src in SOURCES:
        rows = []
        for s, g in games:
            if s != src:
                continue
            y = g["labels"]
            ev = pm.evaluate(y, y, bootstrap_n=1)
            share = [(y[y < 3] == c).mean() for c in range(3)]
            rows.append((*share, ev["n_gt_spells"], ev["gt_mean_spell_duration_s"]))
        m = np.mean(rows, axis=0)
        print(f"{src:7s}  {len(rows):3d}    {m[0]:.3f}  {m[1]:.3f}  {m[2]:.3f}  {m[3]:8.0f}     {m[4]:.1f}")


if __name__ == "__main__":
    main()
