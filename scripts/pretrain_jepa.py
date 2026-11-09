"""Pré-treino do FC-JEPA (C2) nos caches de scripts/prep_jepa.py.

Treina em todos os jogos de DATA_DIR/interim/jepa, menos os n_val_pff primeiros da PFF (ordem por
nome), que só medem a perda. Janelas de T amostras, contínuas, dentro dos blocos, com rotação de 180°
sorteada. Salva a cada época em outputs/jepa/: c2_latest.pth (para retomar) e c2_encoder.pth (só o
context_encoder + config, o que a cabeça usa). Na tela: só perdas e contagens.

Uso (máquina com GPU): uv run python scripts/pretrain_jepa.py [--epochs N] [--device cuda] [--smoke]
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mtah27.config import load_config
from mtah27.jepa.model import FCJEPAModel
from mtah27.paths import INTERIM_DIR, OUTPUTS_DIR
from mtah27.tokens import A, F, flip_tokens, velocity, window_tokens
from mtah27.tracking import Tracking

CACHE = INTERIM_DIR / "jepa"
OUT = OUTPUTS_DIR / "jepa"


class Windows(Dataset):
    def __init__(self, names, T, stride, hz, flip, seed=0):
        self.games, self.index, self.T, self.flip, self.seed = [], [], T, flip, seed
        for g, name in enumerate(names):
            z = np.load(CACHE / f"{name}.npz")
            trk = Tracking(ball=z["ball"], players=z["players"], team=z["team"])
            self.games.append((trk, velocity(trk.ball, hz), velocity(trk.players, hz), tuple(z["pitch"])))
            for a, b in z["blocks"]:
                self.index += [(g, s) for s in range(int(a), int(b) - T + 2, stride)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        g, s = self.index[i]
        trk, vb, vp, pitch = self.games[g]
        tok, pad = window_tokens(trk, vb, vp, s, self.T, pitch)
        if self.flip and np.random.default_rng((self.seed, i, torch.initial_seed() % 2**31)).random() < 0.5:
            tok = flip_tokens(tok)
        return torch.from_numpy(tok), torch.from_numpy(pad)


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def run_epoch(model, loader, T, mask, device, opt=None, sched=None):
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    kw = dict(time_mask_ratio=mask["time_mask_ratio"], agent_mask_ratio=mask["agent_mask_ratio"],
              num_blocks=mask["num_blocks"])  # fmt: skip
    for tok, pad in loader:
        tok, pad = tok.to(device, non_blocking=True), pad.to(device, non_blocking=True)
        keep = ~pad.all(dim=1)
        if not keep.any():
            continue
        tok, pad = tok[keep], pad[keep]
        with torch.set_grad_enabled(train):
            loss = model(tok, pad, T=T, A=A, mask_mode=mask["mode"], mask_kwargs=kw)[0]
        if train:
            sched(opt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), sched.clip)
            opt.step()
            model.update_target_encoder(sched.momentum())
            sched.step += 1
        total, n = total + float(loss.detach()) * len(tok), n + len(tok)
    return total / max(1, n)


class Sched:
    def __init__(self, total, warmup, lr, ema, clip):
        self.total, self.warmup, self.lr, self.ema, self.clip, self.step = total, warmup, lr, ema, clip, 0

    def __call__(self, opt):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(self.step, self.total, self.warmup, self.lr)

    def momentum(self):
        a, b = self.ema
        return a + (b - a) * min(1.0, self.step / max(1, self.total))


def main(epochs, device, smoke, num_workers):
    cfg = load_config()
    c2, hz = cfg["c2"], cfg["c0"]["hz"]
    p, T = c2["pretrain"], c2["window_frames"]
    torch.manual_seed(p["seed"])
    names = sorted(f.stem for f in CACHE.glob("*.npz"))
    pff = [n for n in names if n.startswith("pff_")]
    val_names = pff[: p["n_val_pff"]]
    train_names = [n for n in names if n not in val_names]
    if smoke:
        train_names, val_names = train_names[:2], val_names[:1]
    train_ds = Windows(train_names, T, p["stride"], hz, flip=True, seed=p["seed"])
    val_ds = Windows(val_names, T, p["stride"] * 5, hz, flip=False)
    if smoke:
        train_ds.index, val_ds.index = train_ds.index[:512], val_ds.index[:256]
    print(f"treino: {len(train_names)} jogos, {len(train_ds)} janelas")
    print(f"val: {len(val_names)} jogos, {len(val_ds)} janelas")

    g = torch.Generator().manual_seed(p["seed"])
    dl = dict(batch_size=p["batch_size"], num_workers=num_workers, pin_memory=device == "cuda")
    train_dl = DataLoader(train_ds, shuffle=True, generator=g, drop_last=True, **dl)
    val_dl = DataLoader(val_ds, shuffle=False, **dl)

    model = FCJEPAModel(in_dim_enc=F, d_model=p["d_model"], depth=p["depth"], n_heads=p["n_heads"],
                        same_agent_bias=True, A=A).to(device)  # fmt: skip
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=p["lr"], weight_decay=p["weight_decay"])
    epochs = epochs or p["epochs"]
    sched = Sched(
        epochs * len(train_dl), p["warmup_epochs"] * len(train_dl), p["lr"], p["ema"], p["grad_clip"]
    )

    OUT.mkdir(parents=True, exist_ok=True)
    latest, start = OUT / "c2_latest.pth", 0
    if latest.exists() and not smoke:
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.step, start = ck["step"], ck["epoch"] + 1
        print(f"retomando da época {start}")

    meta = {"pretrain": p, "window_frames": T, "hz": hz, "in_dim": F, "A": A, "train_games": train_names}
    for ep in range(start, epochs):
        t0 = time.time()
        tr = run_epoch(model, train_dl, T, p["mask"], device, opt, sched)
        torch.manual_seed(p["seed"])  # mesma máscara na validação em toda época
        with torch.no_grad():
            va = run_epoch(model, val_dl, T, p["mask"], device)
        print(f"época {ep}: perda treino {tr:.4f} | val {va:.4f} | {time.time() - t0:.0f} s", flush=True)
        if smoke:
            continue
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "step": sched.step, "epoch": ep}, latest
        )
        torch.save({"encoder": model.context_encoder.state_dict(), "meta": meta, "epoch": ep,
                    "val_loss": va}, OUT / "c2_encoder.pth")  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="2 jogos, 512 janelas, 1 época, não salva")
    a = ap.parse_args()
    main(1 if a.smoke else a.epochs, a.device, a.smoke, a.num_workers)
