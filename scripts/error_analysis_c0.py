"""Análise de erro do C0: onde o spellF1 se perde (leave-one-game-out nos 3 jogos da Metrica).

Reproduz o CV do calibrate_c0 com os hiperparâmetros já escolhidos em configs/c0_fitted.yaml
(r_close_m, min_dur_s): para o jogo i, ajusta nos outros 2 e decodifica o i. Só imprime agregados.
Não usa os 5 min públicos.

Tudo é medido na taxa nativa (25 fps). A bola ausente (seção 3) é lida na grade de 5 Hz: cada frame
nativo herda a observação da amostra mais próxima da grade (mesmo mapeamento do c0.decode_game), e
"ausente" é obs == 5 depois do degrade. As seções 2 a 7 somam as contagens dos 3 jogos (pooled).

Uso: uv run python scripts/error_analysis_c0.py
"""

# ruff: noqa: I001  (a ordem importa: calibrate_c0 importa mtah27.paths antes de possession_metrics)
from __future__ import annotations

import numpy as np

from calibrate_c0 import blocks_of, fit_on, pm, prepare  # noqa: F401
from mtah27 import c0
from mtah27.config import load_config

FPS = 25
MISSING = 5  # código de "bola ausente" em obs
NAMES = "HAD"
SPELL_BINS = [("<4s", 0, 4), ("4-10s", 4, 10), ("10-20s", 10, 20), (">20s", 20, 1e9)]
D_BINS = [("<5s", 0, 5), ("5-15s", 5, 15), (">15s", 15, 1e9)]


def band(dur_s, bins):
    return next(i for i, (_, lo, hi) in enumerate(bins) if lo <= dur_s < hi)


def nearest_sample(idx, n):
    """Índice da amostra da grade mais próxima de cada frame nativo 0..n-1."""
    f = np.arange(n)
    right = np.searchsorted(idx, f).clip(0, len(idx) - 1)
    left = (right - 1).clip(0)
    return np.where(f - idx[left] <= np.abs(idx[right] - f), left, right)


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def decode_all(games, cfg, fitted):
    """[(game, pred)] em leave-one-game-out com r_close_m e min_dur_s do ajustado."""
    r, md = fitted["r_close_m"], fitted["min_dur_s"]
    n_mot = len(cfg["speed"]["bins_rel"]) + 1
    out = []
    for i, g in enumerate(games):
        p = fit_on(games[:i] + games[i + 1 :], r, n_mot)
        pred = c0.decode_game(
            g["obs"][r], g["mot"], p, md, cfg["hz"], g["idx"], g["blocks"], len(g["labels"])
        )
        out.append((g, pred))
    return out, r


def main():
    cfg = load_config()["c0"]
    fitted = load_config("c0_fitted.yaml")
    games = prepare("metrica", cfg)
    res, r = decode_all(games, cfg, fitted)
    print(f"C0 leave-one-game-out, metrica; r_close_m={r:g} min_dur_s={fitted['min_dur_s']}\n")

    cm = np.zeros((3, 4), dtype=np.int64)  # verdade H/A/D x previsto H/A/D/X, pooled
    cm_g, acc_miss = [], []  # por jogo; [acertos, total] ausente e presente
    spell_gt = np.zeros((4, 2), int)  # por faixa: [total, casadas]
    spell_pr = np.zeros((4, 2), int)
    frag = []
    d_gt = [[] for _ in D_BINS]  # por faixa: (d_inicio_s, d_fim_s) ou None
    sw_gt, sw_pairs = 0, []

    print("1. Por jogo")
    print("jogo  spellF1  macroF1  n_pred/n_gt")
    s1 = []
    for k, (g, pred) in enumerate(res, 1):
        gt = g["labels"]
        ev = pm.evaluate(gt, pred, bootstrap_n=1)
        s1.append((ev["spell_f1_mean"], ev["macro_f1"]))
        print(f"{k:4d}  {s1[-1][0]:.4f}   {s1[-1][1]:.4f}   {ev['n_pred_spells']}/{ev['n_gt_spells']}")

        c = pm.scored_confusion(pm.confusion(gt, pred))
        cm += c
        cm_g.append(c)

        # 3. acurácia por frame, bola ausente x presente (herdada da amostra mais próxima da grade)
        near = nearest_sample(g["idx"], len(gt))
        miss = g["obs"][r][near] == MISSING
        graded = gt != 3
        ok = gt == pred
        acc_miss.append([[int((ok & graded & m).sum()), int((graded & m).sum())] for m in (miss, ~miss)])

        runs = pm.valid_runs(gt)
        filled = pm.fill_abstentions(pred, runs)
        gt_sp, pr_sp = pm.possession_spells(gt, runs), pm.possession_spells(filled, runs)
        dur = lambda s: (s[1] - s[0]) / FPS  # noqa: E731

        # 4. posses por faixa de duração, casadas a IoU >= 0,5
        gflag = pm.matched_gt_flags(pm.spell_overlaps(gt_sp, pr_sp), len(gt_sp), 0.5)
        pflag = pm.matched_gt_flags(pm.spell_overlaps(pr_sp, gt_sp), len(pr_sp), 0.5)
        for s, f in zip(gt_sp, gflag, strict=True):
            spell_gt[band(dur(s), SPELL_BINS)] += (1, int(f))
        for s, f in zip(pr_sp, pflag, strict=True):
            spell_pr[band(dur(s), SPELL_BINS)] += (1, int(f))

        # 5. fragmentação: segmentos previstos (qualquer rótulo) que tocam cada posse verdadeira
        pr_seg = pm.segments(filled, runs)
        for a, b, _ in gt_sp:
            frag.append(sum(1 for pa, pb, _ in pr_seg if pa < b and a < pb))

        # 6. fronteiras de D: o D previsto de maior sobreposição com cada D verdadeiro
        pr_d = [s for s in pr_seg if s[2] == 2]
        for a, b, c_ in pm.segments(gt, runs):
            if c_ != 2:
                continue
            ov = [(min(b, pb) - max(a, pa), pa, pb) for pa, pb, _ in pr_d if pa < b and a < pb]
            best = max(ov) if ov else None
            d_gt[band((b - a) / FPS, D_BINS)].append(
                None if best is None else (abs(best[1] - a) / FPS, abs(best[2] - b) / FPS)
            )

        # 7. trocas H<->A diretas
        sw = lambda t: [x for x in t if x[1] != x[2] and x[1] < 2 and x[2] < 2]  # noqa: E731
        gt_sw, pr_sw = sw(pm.transitions(gt, runs)), sw(pm.transitions(filled, runs))
        sc = pm.transition_scores(gt_sw, pr_sw, 2 * FPS, typed="to")
        sw_gt += len(gt_sw)
        sw_pairs += [p[0] - q[0] for q, p in sc["pairs"]]

    print(f"média {np.mean([s[0] for s in s1]):.4f}   {np.mean([s[1] for s in s1]):.4f}\n")

    print("2. Confusão (linha = verdade, normalizada por linha; colunas = previsto H A D X), pooled")
    for i in range(3):
        print(
            f"  {NAMES[i]}: " + "  ".join(f"{v:.3f}" for v in cm[i] / cm[i].sum()) + f"   (n={cm[i].sum()})"
        )
    print(
        "  recall por jogo (H A D): "
        + " | ".join(" ".join(f"{c[i, i] / c[i].sum():.3f}" for i in range(3)) for c in cm_g)
    )

    print(
        "\n3. Acurácia por frame avaliado, bola ausente x presente na obs (taxa nativa, amostra mais próxima)"
    )
    for name, j in (("ausente", 0), ("presente", 1)):
        per = [a[j][0] / a[j][1] if a[j][1] else float("nan") for a in acc_miss]
        tot = np.sum([a[j] for a in acc_miss], axis=0)
        print(
            f"  {name:8s}: pooled {tot[0] / tot[1]:.4f} (n={tot[1]})  por jogo "
            + " ".join(f"{x:.4f}" for x in per)
        )

    print("\n4. Posses por faixa de duração (casadas a IoU >= 0,5)")
    print("  faixa    verdadeiras  frac_casada | previstas  frac_que_casa")
    for i, (nm, _, _) in enumerate(SPELL_BINS):
        (tg, mg), (tp, mp) = spell_gt[i], spell_pr[i]
        print(f"  {nm:7s}  {tg:11d}  {mg / max(tg, 1):11.3f} | {tp:9d}  {mp / max(tp, 1):13.3f}")

    f = np.array(frag)
    print(
        f"\n5. Fragmentação: média de segmentos previstos por posse verdadeira = {f.mean():.3f}; "
        f"fração com >1 = {(f > 1).mean():.3f} (n={len(f)})"
    )

    print("\n6. Fronteiras de D (|delta| em s do D previsto de maior sobreposição)")
    print("  faixa D   n   sem_D_prev  início med/p90   fim med/p90")
    for i, (nm, _, _) in enumerate(D_BINS):
        v = [x for x in d_gt[i] if x is not None]
        st, en = [x[0] for x in v], [x[1] for x in v]
        print(
            f"  {nm:7s} {len(d_gt[i]):4d}  {1 - len(v) / max(len(d_gt[i]), 1):10.3f}  "
            f"{pct(st, 50):6.2f}/{pct(st, 90):6.2f}   {pct(en, 50):6.2f}/{pct(en, 90):6.2f}"
        )

    lat = np.array(sw_pairs) / FPS
    print(
        f"\n7. Trocas H<->A diretas: n_gt={sw_gt}; com troca prevista em +-2s (mesmo destino) "
        f"{len(lat) / max(sw_gt, 1):.3f}; atraso mediano com sinal (prev - verd) "
        f"{np.median(lat) if len(lat) else float('nan'):.2f}s"
    )


if __name__ == "__main__":
    main()
