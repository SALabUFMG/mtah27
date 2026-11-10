"""Avaliação cruzada do C0: ajusta numa fonte (ou em várias) e avalia em outra.

Responde se a PFF ajuda no alvo: o CV dentro da PFF não diz isso, porque o domínio é outro (tracking
da transmissão, 22 jogadores sempre presentes). O alvo mais próximo que temos é a Metrica.

- --train pff --test metrica: ajusta nos 64 jogos da PFF e avalia nos 3 da Metrica.
- --train pff metrica --test metrica ("ambos"): deixa um jogo da Metrica de fora por vez e ajusta na
  PFF + nos outros 2. Os jogos entram juntos, sem peso por fonte (a PFF domina as contagens).
- --train metrica --test metrica reproduz o leave-one-game-out do calibrate_c0 (checagem).

Mostra a grade inteira (r_close_m x duração mínima) e não escolhe nada: os parâmetros finais vêm do CV
da fonte de treino (calibrate_c0). Não usa os 5 min públicos.

Uso: uv run python scripts/cross_eval_c0.py --train pff [metrica] [--test metrica]
         [--drop-estimated ball players]   (vale para os jogos da PFF)
"""

# ruff: noqa: I001  (a ordem importa: calibrate_c0 importa mtah27.paths antes de possession_metrics)
from __future__ import annotations

import argparse
import time

import numpy as np

from calibrate_c0 import fit_on, pm, prepare
from mtah27 import c0
from mtah27.config import load_config

SOURCES = ("metrica", "pff")


def splits(data, train, test):
    """[(jogos de ajuste, jogos de teste)]. Se a fonte de teste também treina, um jogo dela fica de fora
    por vez."""
    others = [g for s in train if s != test for g in data[s]]
    if test not in train:
        return [(others, data[test])]
    held = data[test]
    return [(others + held[:i] + held[i + 1 :], [g]) for i, g in enumerate(held)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", choices=SOURCES, required=True)
    ap.add_argument("--test", choices=SOURCES, default="metrica")
    ap.add_argument("--drop-estimated", nargs="*", choices=["ball", "players"], default=[])
    args = ap.parse_args()
    train = sorted(set(args.train))
    drop = tuple(sorted(args.drop_estimated))

    t0 = time.time()
    cfg = load_config()["c0"]
    data = {s: prepare(s, cfg, drop if s == "pff" else ()) for s in sorted(set(train) | {args.test})}
    sp = splits(data, train, args.test)
    n_mot = len(cfg["speed"]["bins_rel"]) + 1
    sizes = ", ".join(f"{s}={len(g)}" for s, g in data.items())
    print(
        f"treino {'+'.join(train)} -> teste {args.test}; jogos {sizes}; {len(sp)} ajuste(s); "
        f"drop_estimated={list(drop)}; carga {time.time() - t0:.0f}s",
        flush=True,
    )

    print("r_close  min_dur  spellF1  macroF1  spellF1 por jogo                    n_pred/n_gt")
    for r in cfg["grid"]["r_close_m"]:
        params = [fit_on(fit_games, r, n_mot) for fit_games, _ in sp]
        for m in cfg["grid"]["min_dur_s"]:
            md = {"H": m, "A": m, "D": 2 * m}
            s1, mf, npd, ngt = [], [], [], []
            for p, (_, test_games) in zip(params, sp, strict=True):
                for g in test_games:
                    pred = c0.decode_game(
                        g["obs"][r], g["mot"], p, md, cfg["hz"], g["idx"], g["blocks"], len(g["labels"])
                    )
                    ev = pm.evaluate(g["labels"], pred, bootstrap_n=1)
                    s1.append(ev["spell_f1_mean"])
                    mf.append(ev["macro_f1"])
                    npd.append(ev["n_pred_spells"])
                    ngt.append(ev["n_gt_spells"])
            per = (
                "/".join(f"{x:.4f}" for x in s1)
                if len(s1) <= 4
                else (f"{min(s1):.4f}/{np.median(s1):.4f}/{max(s1):.4f} (mín/med/máx)")
            )
            print(
                f"{r:7g}  {m:7g}  {np.mean(s1):.4f}   {np.mean(mf):.4f}   {per:34s}  "
                f"{np.mean(npd):.0f}/{np.mean(ngt):.0f}",
                flush=True,
            )
    print(f"tempo: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
