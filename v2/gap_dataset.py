"""Part 2, step 2: 10 Hz training material for the ball-gap model, one file per match.

    .venv\\Scripts\\python.exe v2/gap_dataset.py

Needs the degraded bench (v2/sportec_degrade.py, smoothed by v2/bench.py) and the true ball behind
its gaps (v2/gap_truth.py). Same steps and inputs as v1/ball_dataset.py, per match:
  - players: the degraded copy, smoothed as the bench smooths it; position, velocity, speed,
    acceleration, team, scaled by that match's own pitch. A slot is absent where it has no track
    or moves faster than a human;
  - ball input: the degraded raw ball and a known flag. Its gaps were placed like ours (by cause);
  - truth: the true ball on every step (Sportec), so the loss sees the real gaps, not only the
    stretches training hides on purpose. For ours the truth is only the seen ball;
  - who has the ball, made from the TRUE ball: nearest present player within HAS_BALL_M, the next
    receiver while it travels, nobody when out;
  - val: one 5-minute block per half (the second), the validation material when the match trains.
Our match is built too (no labels needed): the 7 models fill its gaps later.

Writes cache/v2_gap/<match>.npz. Prints totals only.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bench  # noqa: E402
from bench import MATCHES  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "v1"))
import ball_dataset as bd  # noqa: E402
from smooth import acceleration, runs  # noqa: E402
from hidden_gaps import FOLD_BLOCK, HUMAN_SPEED  # noqa: E402
from dead_balls import OUT_M  # noqa: E402

# Params
OUT_DIR = os.path.join(ROOT, "cache", "v2_gap")
DEGRADED = os.path.join(ROOT, "cache", "sportec_degraded")
VAL_BLOCK = 1            # the second 5-minute block of each half: clear of the kickoff
NAMES = tuple(MATCHES) + ("ours",)


def path(name):
    return os.path.join(OUT_DIR, name + ".npz")


def players(m):
    """Smoothed player arrays plus speed and acceleration, as v1/smooth.py makes them."""
    vx, vy = m["player_vx"].astype(np.float64), m["player_vy"].astype(np.float64)
    ax, ay = np.full_like(vx, np.nan), np.full_like(vy, np.nan)
    for j in range(vx.shape[1]):
        ax[:, j], ay[:, j] = acceleration(vx[:, j], vy[:, j], runs(~np.isnan(vx[:, j])))
    return {"player_x": m["player_x"], "player_y": m["player_y"], "player_vx": vx, "player_vy": vy,
            "player_speed": np.hypot(vx, vy), "player_accel": np.hypot(ax, ay), "team": m["team"]}


def scaled(sm, frames, pitch):
    """v1's player inputs, scaled by this match's pitch."""
    length, width = pitch
    speed = sm["player_speed"][frames]
    x = (sm["player_x"][frames] - length / 2) / (length / 2)
    y = (sm["player_y"][frames] - width / 2) / (width / 2)
    vx, vy = sm["player_vx"][frames] / bd.SPEED_SCALE, sm["player_vy"][frames] / bd.SPEED_SCALE
    present = np.isfinite(x) & np.isfinite(y) & np.isfinite(vx) & np.isfinite(vy) & (speed <= HUMAN_SPEED)
    accel = np.clip(np.nan_to_num(sm["player_accel"][frames]) / bd.ACCEL_SCALE, 0, bd.ACCEL_LIMIT)
    team = np.broadcast_to(sm["team"].astype(np.float32), x.shape)
    feats = np.stack([x, y, vx, vy, speed / bd.SPEED_SCALE, accel, team], axis=-1).astype(np.float32)
    feats[~present] = 0
    return feats, present


def who_has_ball(sm, tx, ty, inside, pitch):
    """v1's label, from the true ball: a slot, NOBODY (out or in no one's reach yet) or UNKNOWN."""
    length, width = pitch
    n = len(tx)
    near = np.full(n, -1, dtype=np.int16)
    frames = np.flatnonzero(inside & ~np.isnan(tx))
    for chunk in np.array_split(frames, 200):
        d = np.hypot(sm["player_x"][chunk] - tx[chunk, None], sm["player_y"][chunk] - ty[chunk, None])
        d[~(sm["player_speed"][chunk] <= HUMAN_SPEED)] = np.inf
        best = d.argmin(axis=1)
        close = d[np.arange(len(chunk)), best] <= bd.HAS_BALL_M
        near[chunk[close]] = best[close]
    with np.errstate(invalid="ignore"):
        out = (tx < -OUT_M) | (tx > length + OUT_M) | (ty < -OUT_M) | (ty > width + OUT_M)
    labels = np.full(n, bd.UNKNOWN, dtype=np.int16)
    receiver = bd.NOBODY
    for t in range(n - 1, -1, -1):
        if not inside[t] or np.isnan(tx[t]):
            receiver = bd.NOBODY
        elif out[t]:
            receiver = bd.NOBODY
            labels[t] = bd.NOBODY
        else:
            if near[t] >= 0:
                receiver = near[t]
            labels[t] = receiver
    return labels


def build(name):
    m = bench.load(name if name == "ours" else name + "~")
    inside = m["half"] > 0
    frames, half = bd.step_frames(inside)
    sm = players(m)
    feats, present = scaled(sm, frames, m["pitch"])
    bx, by = m["raw_ball_x"].astype(np.float32), m["raw_ball_y"].astype(np.float32)
    if name == "ours":
        tx, ty = bx, by
    else:
        t = np.load(os.path.join(DEGRADED, name, "ball_truth.npz"))
        tx, ty = t["x"], t["y"]
    labels = who_has_ball(sm, tx, ty, inside, m["pitch"])[frames]
    steps = np.arange(len(frames))
    labels[(labels >= 0) & ~present[steps, np.maximum(labels, 0)]] = bd.UNKNOWN
    # 5-minute blocks counted from the start of each half
    block = np.zeros(len(frames), dtype=np.int16)
    for h in (0, 1):
        f = frames[half == h]
        block[half == h] = (f - f[0]) // FOLD_BLOCK
    return {"frames": frames, "half": half, "block": block, "val": block == VAL_BLOCK,
            "label": labels.astype(np.int8), "ball_x": bx[frames], "ball_y": by[frames],
            "known": np.isfinite(bx[frames]), "truth_x": tx[frames], "truth_y": ty[frames],
            "truth_known": np.isfinite(tx[frames]), "player": feats, "present": present,
            "pitch": np.array(m["pitch"], dtype=np.float32), "waiting": np.zeros(len(frames), dtype=bool)}


def load(name):
    return dict(np.load(path(name)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("match    steps  ball known  truth known  slots/step  label: slot  nobody  none   val steps")
    for name in NAMES:
        ds = build(name)
        np.savez_compressed(path(name), **ds)
        lab = ds["label"]
        print("{:7s} {:6d}    {:5.1%}      {:5.1%}       {:5.1f}       {:5.1%}  {:5.1%}  {:5.1%}  {:6d}".format(
            name, len(lab), ds["known"].mean(), ds["truth_known"].mean(), ds["present"].sum(axis=1).mean(),
            (lab >= 0).mean(), (lab == bd.NOBODY).mean(), (lab == bd.UNKNOWN).mean(), ds["val"].sum()))
    print("wrote cache/v2_gap/<match>.npz")


if __name__ == "__main__":
    main()
