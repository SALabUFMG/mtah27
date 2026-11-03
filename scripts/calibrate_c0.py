"""Calibra o C0 nos 3 jogos externos da Metrica (leave-one-game-out) e grava configs/c0_fitted.yaml.

A bola é degradada para imitar a ausência do alvo; a grade (r_close_m x duração mínima) vem de
configs/default.yaml. Escolhe o melhor spellF1 médio e reajusta nos 3 jogos. Não usa os 5 min públicos.

Uso: uv run python scripts/calibrate_c0.py
"""

# ruff: noqa: I001  (a ordem importa: mtah27.paths precisa vir antes de possession_metrics)
from __future__ import annotations

import time

import numpy as np
import yaml

from mtah27 import c0, external, observe
from mtah27.config import load_config
from mtah27.paths import CONFIGS_DIR
import possession_metrics as pm  # type: ignore  # noqa: E402

FPS = 25
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


def main():
    t0 = time.time()
    cfg = load_config()["c0"]
    games = []
    for i in (1, 2, 3):
        trk, labels = external.load_sample_game(i)
        games.append((observe.degrade(trk, **cfg["degrade"], fps=FPS), labels))
    blocks = [blocks_of(y) for _, y in games]
    sp_cfg = cfg["speed"]
    n_mot = len(sp_cfg["bins_ms"]) + 1
    mots = [
        observe.speed_bin(observe.player_speed(t, FPS, sp_cfg["window_s"]), sp_cfg["bins_ms"])
        for t, _ in games
    ]

    results = []
    for r in cfg["grid"]["r_close_m"]:
        obs = [observe.observe(trk, r, external.PITCH) for trk, _ in games]
        folds = [
            c0.fit(*concat([(obs[j], mots[j], games[j][1]) for j in range(3) if j != k]), n_mot=n_mot)
            for k in range(3)
        ]
        for m in cfg["grid"]["min_dur_s"]:
            md = {"H": m, "A": m, "D": 2 * m}
            sp, mf, npd, ngt = [], [], [], []
            for k in range(3):
                pred = c0.decode_game(obs[k], mots[k], folds[k], md, FPS, blocks[k])
                ev = pm.evaluate(games[k][1], pred, bootstrap_n=1)
                sp.append(ev["spell_f1_mean"])
                mf.append(ev["macro_f1"])
                npd.append(ev["n_pred_spells"])
                ngt.append(ev["n_gt_spells"])
            results.append((r, m, float(np.mean(sp)), float(np.mean(mf)), sp, np.mean(npd), np.mean(ngt)))

    print("r_close  min_dur  spellF1  macroF1  spellF1 por jogo        n_pred/n_gt")
    for r, m, s, f, sp, npd, ngt in results:
        print(f"{r:7g}  {m:7g}  {s:.4f}   {f:.4f}   {' '.join(f'{x:.4f}' for x in sp)}   {npd:.0f}/{ngt:.0f}")
    best = max(results, key=lambda t: (t[2], -t[0], -t[1]))  # empate: menor r, depois menor m
    r, m = best[0], best[1]
    print(f"escolhido: r_close_m={r:g} min_dur_s={m:g} spellF1={best[2]:.4f} macroF1={best[3]:.4f}")

    obs = [observe.observe(trk, r, external.PITCH) for trk, _ in games]
    fitted = c0.fit(*concat([(obs[j], mots[j], games[j][1]) for j in range(3)]), n_mot=n_mot)
    fitted.update(
        r_close_m=r,
        min_dur_s={"H": m, "A": m, "D": 2 * m},
        speed=sp_cfg,
        data_commit=COMMIT,
        logo_spell_f1=round(best[2], 4),
        logo_macro_f1=round(best[3], 4),
    )
    text = "# Gerado por scripts/calibrate_c0.py. Não edite à mão.\n" + yaml.safe_dump(
        rnd(fitted), sort_keys=True
    )
    with open(CONFIGS_DIR / "c0_fitted.yaml", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"tempo: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
