"""Features de janela por amostra da grade (bola, disputa, times, movimento) para o C1.

Tudo depende só do time dos slots, nunca do índice do slot. NaN = sem dado; nunca inf.
"""

from __future__ import annotations

import numpy as np

from mtah27.observe import normalize_speed, player_speed
from mtah27.tracking import AWAY, HOME, Tracking

_TEAMS = (("home", HOME), ("away", AWAY))
_BALL = (
    "ball_present",
    "ball_presence_win",
    "ball_speed",
    "ball_speed_win",
    "ball_out",
    "ball_d_side",
    "ball_d_end",
)
_CONTEST = (
    "home_d1",
    "home_d2",
    "away_d1",
    "away_d2",
    "d_diff",
    "d_diff_win",
    "home_n3",
    "home_n8",
    "away_n3",
    "away_n8",
    "nearest_home_win",
)
_TEAM = ("cx_mid", "cy", "spread", "cspeed", "n", "dball")
_MOVE = ("speed_med", "speed_med_norm", "slow_frac")
FEATURES: tuple[str, ...] = (
    *_BALL,
    *_CONTEST,
    *(f"{t}_{n}" for t, _ in _TEAMS for n in _TEAM),
    *_MOVE,
)

BALL_V_MAX = 40.0  # m/s; acima disso é salto do tracking
PLAYER_V_MAX = 12.0  # igual a observe.player_speed
SLOW_MS = 0.5


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Média móvel centrada de k amostras ignorando NaN (NaN se a janela não tem nada)."""
    ok = ~np.isnan(x)
    ker = np.ones(k)
    den = np.convolve(ok, ker, "same")
    num = np.convolve(np.where(ok, x, 0.0), ker, "same")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def _two_nearest(d: np.ndarray) -> np.ndarray:
    """[N, 2] 1ª e 2ª menores distâncias por linha (NaN se não houver)."""
    d = np.where(np.isnan(d), np.inf, d)
    d = np.concatenate([d, np.full((len(d), 2), np.inf)], axis=1)  # time com <2 slots
    out = np.sort(np.partition(d, 1, axis=1)[:, :2], axis=1)
    return np.where(np.isinf(out), np.nan, out)


def window_features(
    trk: Tracking,
    pitch: tuple[float, float],
    hz: float,
    blocks: list[tuple[int, int]],
    window_s: float = 2.0,
) -> np.ndarray:
    """Features float32 [N, len(FEATURES)] na grade de hz; janela = média móvel centrada de window_s."""
    length, width = pitch
    k = max(1, round(window_s * hz))
    ball, pl = trk.ball.astype(np.float64), trk.players.astype(np.float64)
    n = len(ball)
    f: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        # bola
        bx, by = ball[:, 0], ball[:, 1]
        present = ~np.isnan(ball).any(axis=1)
        f["ball_present"] = present.astype(float)
        f["ball_presence_win"] = _smooth(f["ball_present"], k)
        v = np.r_[np.nan, np.hypot(np.diff(bx), np.diff(by)) * hz]
        v = np.where(v > BALL_V_MAX, np.nan, v)
        f["ball_speed"] = v
        f["ball_speed_win"] = _smooth(v, k)
        out = (bx < -1) | (bx > length + 1) | (by < -1) | (by > width + 1)
        f["ball_out"] = np.where(present, out, np.nan)
        f["ball_d_side"] = np.minimum(by, width - by)
        f["ball_d_end"] = np.minimum(bx, length - bx)

        # disputa
        d = np.hypot(pl[..., 0] - bx[:, None], pl[..., 1] - by[:, None])  # [N, S]
        near = {}
        for name, t in _TEAMS:
            dt = d[:, trk.team == t]
            near[name] = _two_nearest(dt)
            f[f"{name}_d1"], f[f"{name}_d2"] = near[name][:, 0], near[name][:, 1]
            for r in (3, 8):
                cnt = (dt < r).sum(axis=1).astype(float)
                f[f"{name}_n{r}"] = np.where(present, cnt, np.nan)
        f["d_diff"] = f["home_d1"] - f["away_d1"]
        f["d_diff_win"] = _smooth(f["d_diff"], k)
        h, a = np.nan_to_num(f["home_d1"], nan=np.inf), np.nan_to_num(f["away_d1"], nan=np.inf)
        valid = present & ~(np.isinf(h) & np.isinf(a))
        f["nearest_home_win"] = _smooth(np.where(valid, (h < a).astype(float), np.nan), k)

        # times
        for name, t in _TEAMS:
            p = pl[:, trk.team == t]
            ok = ~np.isnan(p).any(axis=2)
            cnt = ok.sum(axis=1)
            safe = np.maximum(cnt, 1)
            pz = np.where(ok[..., None], p, 0.0)
            c = np.where((cnt > 0)[:, None], pz.sum(axis=1) / safe[:, None], np.nan)  # [N, 2]
            dist = np.where(ok, np.hypot(p[..., 0] - c[:, None, 0], p[..., 1] - c[:, None, 1]), 0.0)
            m1 = dist.sum(axis=1) / safe
            var = np.maximum((dist**2).sum(axis=1) / safe - m1**2, 0.0)
            f[f"{name}_cx_mid"] = np.abs(c[:, 0] - length / 2)  # não depende do lado de ataque
            f[f"{name}_cy"] = c[:, 1]
            f[f"{name}_dball"] = np.hypot(c[:, 0] - bx, c[:, 1] - by)
            f[f"{name}_spread"] = np.where(cnt > 0, np.sqrt(var), np.nan)
            # velocidade do time: deslocamento médio dos slots presentes em amostras consecutivas
            # (o centróide salta quando um slot aparece/some)
            dp = np.diff(p, axis=0) * hz
            okv = ~np.isnan(dp).any(axis=2) & (np.hypot(dp[..., 0], dp[..., 1]) <= PLAYER_V_MAX)
            nv = okv.sum(axis=1)
            mv = np.where(okv[..., None], dp, 0.0).sum(axis=1) / np.maximum(nv, 1)[:, None]
            cv = np.r_[np.nan, np.where(nv > 0, np.hypot(mv[:, 0], mv[:, 1]), np.nan)]
            f[f"{name}_cspeed"] = _smooth(cv, k)
            f[f"{name}_n"] = cnt.astype(float)

        # movimento
        sp = player_speed(trk, hz, 1.0, PLAYER_V_MAX).astype(np.float64)
        f["speed_med"] = sp
        f["speed_med_norm"] = normalize_speed(sp, blocks)
        pv = np.linalg.norm(np.diff(pl, axis=0), axis=2) * hz
        pv = np.where(pv > PLAYER_V_MAX, np.nan, pv)
        nv = (~np.isnan(pv)).sum(axis=1)
        slow = (pv < SLOW_MS).sum(axis=1) / np.maximum(nv, 1)
        slow = np.r_[np.nan, np.where(nv > 0, slow, np.nan)]
        f["slow_frac"] = _smooth(slow, k)

    X = np.stack([f[name] for name in FEATURES], axis=1) if n else np.empty((0, len(FEATURES)))
    return np.where(np.isfinite(X), X, np.nan).astype(np.float32)
