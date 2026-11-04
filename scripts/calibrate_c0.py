"""Calibra o C0 em jogos externos (K-fold por jogo) e grava os parâmetros em configs/.

Fontes: os 3 jogos da Metrica (K = 3 é o leave-one-game-out) ou os 64 da PFF (Copa 2022). Cada execução
usa uma fonte só. Tudo roda na grade de c0.hz (default.yaml) e a avaliação volta à taxa nativa. A bola é
degradada dentro dos blocos avaliados para imitar a ausência do alvo; a grade (r_close_m x duração mínima)
vem de configs/default.yaml. Escolhe o melhor spellF1 médio por jogo e reajusta em todos. Não usa os 5 min
públicos. O cv_spell_f1 gravado é o máximo da grade, então é otimista (viés de seleção): serve para
comparar variantes entre si, não como estimativa do placar.

Saída: configs/c0_fitted.yaml (metrica, a usada na submissão) ou c0_fitted_<fonte>[_sem_<estimado>].yaml.

Uso: uv run python scripts/calibrate_c0.py [--source metrica|pff] [--folds 8]
         [--drop-estimated ball players]   (só PFF: posições estimadas pela PFF viram NaN)
"""

# ruff: noqa: I001  (a ordem importa: mtah27.paths precisa vir antes de possession_metrics)
from __future__ import annotations

import argparse
import time

import numpy as np
import yaml

from mtah27 import c0, external, observe, pff
from mtah27.config import load_config
from mtah27.paths import CONFIGS_DIR
import possession_metrics as pm  # type: ignore  # noqa: E402

COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"  # metrica-sports/sample-data


def blocks_of(labels):
    """Blocos inclusivos [first, last] dos trechos não-X."""
    return [(a, b - 1) for a, b in pm.valid_runs(labels)]


def concat(parts):
    """Concatena (obs, mot, labels) de vários jogos com 1 frame X entre eles (sem transição cruzada)."""
    obs, mot, lab = [], [], []
    for o, mo, y in parts:
        obs += [o, np.zeros(1, o.dtype)]
        mot += [mo, np.zeros(1, mo.dtype)]
        lab += [y, np.full(1, 3, y.dtype)]
    return np.concatenate(obs), np.concatenate(mot), np.concatenate(lab)


def rnd(x):
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v) for v in x]
    return round(float(x), 6) if isinstance(x, (float, np.floating)) else x


def load_games(source, drop_estimated=()):
    """(tracking, rótulos, fps, campo) de cada jogo da fonte, em ordem fixa."""
    if source == "metrica":
        for i in (1, 2, 3):
            trk, labels = external.load_sample_game(i)
            yield trk, labels, 25, external.PITCH
    else:
        for gid in pff.game_ids():
            yield pff.load_game(gid, drop_estimated)


def prepare(source, cfg, drop_estimated=()):
    """Por jogo, só o que a calibração usa: obs por raio da grade, faixa de velocidade, rótulos, fps.

    O tracking é descartado logo após, para caber os 64 jogos da PFF em memória.
    """
    sp_cfg, hz = cfg["speed"], cfg["hz"]
    games = []
    for trk, labels, fps, pitch in load_games(source, drop_estimated):
        idx = observe.grid(len(labels), fps, hz)
        blocks = blocks_of(labels)
        gblocks = observe.grid_blocks(idx, blocks)
        trk = observe.degrade(observe.subsample(trk, idx), **cfg["degrade"], fps=hz, blocks=gblocks)
        speed = observe.player_speed(trk, hz, sp_cfg["window_s"])
        speed = observe.normalize_speed(speed, gblocks)
        games.append(
            {
                "obs": {r: observe.observe(trk, r, pitch) for r in cfg["grid"]["r_close_m"]},
                "mot": observe.speed_bin(speed, sp_cfg["bins_rel"]),
                "labels": labels,  # nativos: a avaliação é na taxa do jogo
                "labels_grid": labels[idx],  # na grade: o ajuste
                "idx": idx,
                "blocks": blocks,
            }
        )
    return games


def fit_on(games, r, n_mot):
    return c0.fit(*concat([(g["obs"][r], g["mot"], g["labels_grid"]) for g in games]), n_mot=n_mot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["metrica", "pff"], default="metrica")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--drop-estimated", nargs="*", choices=["ball", "players"], default=[])
    args = ap.parse_args()
    drop = tuple(sorted(args.drop_estimated))
    if drop and args.source != "pff":
        ap.error("--drop-estimated só vale para --source pff")

    t0 = time.time()
    cfg = load_config()["c0"]
    games = prepare(args.source, cfg, drop)
    k = min(args.folds, len(games))
    fold = [j % k for j in range(len(games))]
    n_mot = len(cfg["speed"]["bins_rel"]) + 1
    print(f"{args.source}: {len(games)} jogos, {k} folds, carga {time.time() - t0:.0f}s", flush=True)

    results = []
    for r in cfg["grid"]["r_close_m"]:
        params = [fit_on([g for g, f in zip(games, fold, strict=True) if f != i], r, n_mot) for i in range(k)]
        for m in cfg["grid"]["min_dur_s"]:
            md = {"H": m, "A": m, "D": 2 * m}
            sp, mf, npd, ngt = [], [], [], []
            for g, f in zip(games, fold, strict=True):
                pred = c0.decode_game(
                    g["obs"][r], g["mot"], params[f], md, cfg["hz"], g["idx"], g["blocks"], len(g["labels"])
                )
                ev = pm.evaluate(g["labels"], pred, bootstrap_n=1)
                sp.append(ev["spell_f1_mean"])
                mf.append(ev["macro_f1"])
                npd.append(ev["n_pred_spells"])
                ngt.append(ev["n_gt_spells"])
            results.append((r, m, float(np.mean(sp)), float(np.mean(mf)), sp, np.mean(npd), np.mean(ngt)))
            print(f"r={r:g} m={m:g} spellF1={results[-1][2]:.4f} ({time.time() - t0:.0f}s)", flush=True)

    print("r_close  min_dur  spellF1  macroF1  spellF1 por jogo (mín/mediana/máx)  n_pred/n_gt")
    for r, m, s, f, sp, npd, ngt in results:
        dist = f"{min(sp):.4f}/{np.median(sp):.4f}/{max(sp):.4f}"
        print(f"{r:7g}  {m:7g}  {s:.4f}   {f:.4f}   {dist:34s}  {npd:.0f}/{ngt:.0f}")
    best = max(results, key=lambda t: (t[2], -t[0], -t[1]))  # empate: menor r, depois menor m
    r, m = best[0], best[1]
    print(f"escolhido: r_close_m={r:g} min_dur_s={m:g} spellF1={best[2]:.4f} macroF1={best[3]:.4f}")

    fitted = fit_on(games, r, n_mot)
    fitted.update(
        r_close_m=r,
        min_dur_s={"H": m, "A": m, "D": 2 * m},
        speed=cfg["speed"],
        hz=cfg["hz"],
        source=args.source,
        data_commit=COMMIT if args.source == "metrica" else "pff_wc2022",
        n_games=len(games),
        drop_estimated=list(drop),
        cv_folds=k,
        cv_spell_f1=round(best[2], 4),
        cv_macro_f1=round(best[3], 4),
    )
    text = "# Gerado por scripts/calibrate_c0.py. Não edite à mão.\n" + yaml.safe_dump(
        rnd(fitted), sort_keys=True
    )
    suffix = "".join(f"_sem_{d}" for d in drop)
    name = "c0_fitted.yaml" if args.source == "metrica" else f"c0_fitted_{args.source}{suffix}.yaml"
    with open(CONFIGS_DIR / name, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"gravado configs/{name}; tempo: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
