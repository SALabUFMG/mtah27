"""Testa se o C1 aprendeu "bola ausente = D", um artefato dos jogos externos.

Nos jogos da Metrica a bola quase só some em bola morta. Se o modelo usa isso, ele prevê D demais
quando a bola some com o jogo vivo, como acontece no alvo. Variantes de features (cada uma treinada
em leave-one-game-out nos 3 jogos, no ponto de c1_fitted.yaml):

- full: as 33 features;
- sem_presenca: sem ball_present e ball_presence_win;
- sem_bola: sem nenhuma feature que use a bola (limite inferior);
- aug_vivo: as 33 features, mas a última das n_aug cópias de treino tem a bola apagada só nos frames
  vivos (c1.game_features com live_aug).

Condições de teste: "padrão" (a degradação da calibração, 41% sem bola nos blocos) e "estresse"
(lacunas sorteadas só nos frames vivos até 41% deles sem bola). Com --target, imprime também a fração
de D prevista no jogo da competição por variante (só esse agregado). Não usa os 5 min públicos.

Uso: uv run python scripts/ablate_c1_ball.py [--target]
"""

# ruff: noqa: I001  (a ordem importa: calibrate_c0 importa mtah27.paths antes de possession_metrics)
from __future__ import annotations

import argparse
import time

import numpy as np

from calibrate_c0 import pm
from mtah27 import c1, external, features, observe
from mtah27.config import load_config

BALL_FEATURES = {
    "ball_present", "ball_presence_win", "ball_speed", "ball_speed_win", "ball_out", "ball_d_side",
    "ball_d_end", "home_d1", "home_d2", "away_d1", "away_d2", "d_diff", "d_diff_win", "home_n3",
    "home_n8", "away_n3", "away_n8", "nearest_home_win", "home_dball", "away_dball",
}  # fmt: skip
VARIANTS = {  # nome: (features retiradas, live_aug no treino)
    "full": (set(), False),
    "sem_presenca": ({"ball_present", "ball_presence_win"}, False),
    "sem_bola": (BALL_FEATURES, False),
    "aug_vivo": (set(), True),
}


def load(cfg_c0, fitted_cfg):
    """Por jogo da Metrica: features de treino (n_aug cópias, com e sem live_aug), de teste padrão e de
    estresse."""
    hz, window_s, deg = fitted_cfg["hz"], fitted_cfg["window_s"], cfg_c0["degrade"]
    games = []
    for i in (1, 2, 3):
        trk, labels = external.load_sample_game(i)
        g = c1.game_features(trk, labels, 25, external.PITCH, hz, deg, window_s, fitted_cfg["n_aug"])
        aug = c1.game_features(trk, labels, 25, external.PITCH, hz, deg, window_s, fitted_cfg["n_aug"], True)
        g["feats_aug"] = aug["feats"]
        sub = observe.subsample(trk, g["idx"])
        stress = observe.degrade(sub, **deg, fps=hz, blocks=c1.live_blocks(g["labels_grid"]))
        gblocks = observe.grid_blocks(g["idx"], g["blocks"])
        g["stress"] = features.window_features(stress, external.PITCH, hz, gblocks, window_s)
        live = g["labels_grid"] < 2
        g["miss_live"] = float(np.isnan(stress.ball[live, 0]).mean())
        games.append(g)
    return games


def summary(ev):
    d = ev["per_class"]["D"]
    return ev["spell_f1_mean"], ev["macro_f1"], d["precision"], d["recall"], ev["n_pred_spells"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="store_true", help="também a fração de D prevista no alvo")
    args = ap.parse_args()
    t0 = time.time()
    cfg_c0 = load_config()["c0"]
    fitted_cfg = load_config(load_config()["c1"]["fitted"])
    hz, md, w = fitted_cfg["hz"], fitted_cfg["min_dur_s"], fitted_cfg["emission_weight"]
    games = load(cfg_c0, fitted_cfg)
    miss = "/".join(f"{g['miss_live']:.2f}" for g in games)
    print(f"carga {time.time() - t0:.0f}s; estresse: fração sem bola nos frames vivos {miss}", flush=True)

    cols = {
        v: [j for j, f in enumerate(features.FEATURES) if f not in drop] for v, (drop, _) in VARIANTS.items()
    }
    print("variante      teste     spellF1  macroF1  D_prec  D_rec   n_pred/n_gt")
    for v, keep in cols.items():
        res = {"padrão": [], "estresse": []}
        for i, g in enumerate(games):
            X, y = c1.train_rows(train_games(games[:i] + games[i + 1 :], VARIANTS[v][1]))
            fitted = c1.fit(X[:, keep], y, seed=0)
            for cond, Xt in (("padrão", g["feats"][0]), ("estresse", g["stress"])):
                pred = c1.decode_game(Xt[:, keep], fitted, md, hz, g["idx"], g["blocks"], len(g["labels"]), w)
                ev = pm.evaluate(g["labels"], pred, bootstrap_n=1)
                res[cond].append((*summary(ev), ev["n_gt_spells"]))
        for cond, rows in res.items():
            m = np.mean(rows, axis=0)
            print(
                f"{v:12s}  {cond:8s}  {m[0]:.4f}   {m[1]:.4f}   {m[2]:.3f}   {m[3]:.3f}   "
                f"{m[4]:.0f}/{m[5]:.0f}",
                flush=True,
            )
    if args.target:
        target_d_share(cfg_c0, fitted_cfg, games, cols)
    print(f"tempo: {time.time() - t0:.0f}s")


def train_games(games, live_aug):
    """Os jogos com as cópias de treino da variante (feats_aug quando live_aug)."""
    return [{**g, "feats": g["feats_aug"]} if live_aug else g for g in games]


def target_d_share(cfg_c0, fitted_cfg, games, cols):
    """Fração de D prevista nos frames avaliados do alvo, por variante (treino nos 3 jogos)."""
    from games import GAMES, half_video_blocks  # type: ignore
    from mtah27.tracking import load_tracking

    trk = load_tracking("BAR-ATH")
    n, hz = len(trk.ball), fitted_cfg["hz"]
    idx = observe.grid(n, GAMES["BAR-ATH"]["fps"], hz)
    blocks = [(a, b) for _, a, b in half_video_blocks("BAR-ATH")]
    Xt = features.window_features(
        observe.subsample(trk, idx), GAMES["BAR-ATH"]["pitch"], hz, observe.grid_blocks(idx, blocks),
        fitted_cfg["window_s"],
    )  # fmt: skip
    inside = np.zeros(n, bool)
    for a, b in blocks:
        inside[a : b + 1] = True
    for v, keep in cols.items():
        X, y = c1.train_rows(train_games(games, VARIANTS[v][1]))
        fitted = c1.fit(X[:, keep], y, seed=0)
        pred = c1.decode_game(
            Xt[:, keep], fitted, fitted_cfg["min_dur_s"], hz, idx, blocks, n, fitted_cfg["emission_weight"]
        )
        print(f"alvo {v:12s}: D = {(pred[inside] == 2).mean():.3f} dos frames avaliados", flush=True)


if __name__ == "__main__":
    main()
