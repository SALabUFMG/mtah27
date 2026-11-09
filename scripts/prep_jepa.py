"""Grava os jogos na grade de hz para o pré-treino do FC-JEPA (C2): DATA_DIR/interim/jepa/<nome>.npz.

Jogos: os 3 da Metrica, os 64 da PFF (bola estimada vira NaN, como o ATD do alvo, que não estima) e o
jogo da competição (só o tracking, sem rótulo: o RULES.md permite autossupervisionado no jogo dado).
Cada arquivo tem ball, players, team, pitch, blocks (trechos [first, last] na grade onde há janelas) e,
nos externos, labels (H/A/D/X na grade, para a cabeça). Saída na tela: só nomes e contagens.

Uso: uv run python scripts/prep_jepa.py [--only metrica|pff|target]
"""

# ruff: noqa: I001  (a ordem importa: mtah27.paths precisa vir antes de `games`)
from __future__ import annotations

import argparse

import numpy as np

from mtah27 import external, observe, pff
from mtah27.config import load_config
from mtah27.paths import INTERIM_DIR
from mtah27.tracking import load_tracking
from games import GAMES, half_video_blocks  # type: ignore  # noqa: E402
import possession_metrics as pm  # type: ignore  # noqa: E402

OUT = INTERIM_DIR / "jepa"


def save(name, trk, fps, pitch, hz, blocks, labels=None):
    idx = observe.grid(len(trk.ball), fps, hz)
    sub = observe.subsample(trk, idx)
    gblocks = np.array(observe.grid_blocks(idx, blocks), np.int64).reshape(-1, 2)
    extra = {} if labels is None else {"labels": labels[idx].astype(np.int8)}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / f"{name}.npz", ball=sub.ball, players=sub.players, team=sub.team,
        pitch=np.array(pitch, np.float32), blocks=gblocks, **extra,
    )  # fmt: skip
    print(f"{name}: {len(idx)} amostras, {len(gblocks)} blocos")


def labeled_blocks(labels):
    return [(a, b - 1) for a, b in pm.valid_runs(labels)]


def main(only: str | None) -> None:
    hz = load_config()["c0"]["hz"]
    if only in (None, "metrica"):
        for i in (1, 2, 3):
            trk, labels = external.load_sample_game(i)
            save(f"metrica_{i}", trk, 25, external.PITCH, hz, labeled_blocks(labels), labels)
    if only in (None, "pff"):
        for gid in pff.game_ids():
            trk, labels, fps, pitch = pff.load_game(gid, drop_estimated=("ball",))
            save(f"pff_{gid}", trk, fps, pitch, hz, labeled_blocks(labels), labels)
    if only in (None, "target"):
        game = load_config()["game"]
        blocks = [(first, last) for _, first, last in half_video_blocks(game)]
        save(f"target_{game}", load_tracking(game), GAMES[game]["fps"], GAMES[game]["pitch"], hz, blocks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["metrica", "pff", "target"])
    main(ap.parse_args().only)
