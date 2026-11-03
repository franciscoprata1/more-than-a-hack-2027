"""Phase 3, step 2: three simple ways to guess the ball inside the hidden gaps, no machine learning.

    .venv\\Scripts\\python.exe v1/ball_baselines.py

Needs cache/hidden_gaps.npz (v1/hidden_gaps.py).
  A. straight line between the last seen and the next seen ball;
  B. follow the carrier: stay on the player nearest the ball when it disappeared. If that track
     ends, move to the player nearest the last guess. With the ball hidden nobody else can be
     seen becoming nearest, so this is the only handover;
  C. pressure point: start at the last seen ball and keep moving the guess towards the spot where
     fast opponents near it are heading, the closest ones counting most. Opponents = the team
     without the player nearest the guess.
A knows both ends of the gap. B and C only use the past, as the plan describes them.

The ball is blanked inside the gaps before any baseline runs. Settings come from football sense,
fixed before the first run: this test is the target for step 3, so it never tunes them.

Writes cache/ball_baselines.csv. Prints totals only.
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from smooth import FPS, PITCH  # noqa: E402
from hidden_gaps import GAP_SECONDS, HUMAN_SPEED, SEED  # noqa: E402

# Params
CACHE_DIR = os.path.join(ROOT, "cache")
OUT_PATH = os.path.join(CACHE_DIR, "ball_baselines.csv")
PRESS_RADIUS_M = 15.0    # opponents further from the guess than this aren't reacting to the ball
PRESS_SPEED = 2.0        # m/s: above a walk, so they are going somewhere
CLOSE_M = 5.0            # a player's weight halves about every 3.5 m from the guess
ANCHOR = 0.1             # weak pull to the current guess, so one or two runners can't throw it far
FOLLOW_RATE = 1.0        # per second: the guess closes this share of the distance to the spot
BOOTSTRAPS = 1000


def hide(v, gaps, length):
    """Copy of v with the ball blanked inside every gap."""
    out = v.astype(np.float64).copy()
    for a in gaps:
        out[a:a + length] = np.nan
    return out


def straight_line(last, nxt, length):
    """A: the line from the last seen ball to the next seen ball."""
    t = (np.arange(1, length + 1) / (length + 1))[:, None]
    return last + t * (nxt - last)


def follow_carrier(last, px, py):
    """B: px, py cover the last seen frame then the gap. Returns the carrier's path over the gap."""
    guess = last.copy()
    slot = None
    out = np.zeros((len(px) - 1, 2))
    for t in range(len(px)):
        if slot is None or not np.isfinite(px[t, slot]):
            d = np.hypot(px[t] - guess[0], py[t] - guess[1])
            slot = int(np.nanargmin(d)) if np.isfinite(d).any() else None
        if slot is not None:
            guess = np.array([px[t, slot], py[t, slot]])
        if t > 0:
            out[t - 1] = guess
    return out


def pressure_target(guess, px, py, vx, vy, speed, team):
    """Where the fast opponents near `guess` are converging, or None if there are none."""
    dx, dy = guess[0] - px, guess[1] - py
    d = np.hypot(dx, dy)
    ok = np.isfinite(d) & np.isfinite(vx)
    if not ok.any():
        return None
    nearest_team = team[np.flatnonzero(ok)[np.argmin(d[ok])]]
    fast = ok & (team != nearest_team) & (d <= PRESS_RADIUS_M) & (speed >= PRESS_SPEED) & (speed <= HUMAN_SPEED)
    fast &= dx * vx + dy * vy > 0  # running towards the guess, not away from it
    if not fast.any():
        return None
    # The point closest to all their running lines, weighted by closeness.
    ux, uy = vx[fast] / speed[fast], vy[fast] / speed[fast]
    w = np.exp(-d[fast] / CLOSE_M)
    a11, a12, a22 = w * (1 - ux * ux), -w * ux * uy, w * (1 - uy * uy)
    A = np.array([[a11.sum() + ANCHOR, a12.sum()], [a12.sum(), a22.sum() + ANCHOR]])
    b = np.array([(a11 * px[fast] + a12 * py[fast]).sum() + ANCHOR * guess[0],
                  (a12 * px[fast] + a22 * py[fast]).sum() + ANCHOR * guess[1]])
    return np.linalg.solve(A, b)


def pressure_point(last, px, py, vx, vy, speed, team):
    """C: px.. cover the gap. Returns the guess per frame."""
    guess = last.copy()
    step = FOLLOW_RATE / FPS
    out = np.zeros((len(px), 2))
    for t in range(len(px)):
        target = pressure_target(guess, px[t], py[t], vx[t], vy[t], speed[t], team)
        if target is not None:
            guess = np.clip(guess + step * (target - guess), (0, 0), PITCH)
        out[t] = guess
    return out


def run_length(seconds, gaps, sm, x_true, y_true):
    """Error in metres per gap, one row per gap and one column per hidden frame, for each baseline."""
    length = seconds * FPS
    x, y = hide(x_true, gaps, length), hide(y_true, gaps, length)
    team = sm["team"]
    errors = {"A": [], "B": [], "C": []}
    for a in gaps:
        b = a + length
        last, nxt = np.array([x[a - 1], y[a - 1]]), np.array([x[b], y[b]])
        assert np.isfinite(last).all() and np.isfinite(nxt).all() and np.isnan(x[a:b]).all()
        truth = np.c_[x_true[a:b], y_true[a:b]]
        px, py = sm["player_x"], sm["player_y"]
        paths = {"A": straight_line(last, nxt, length),
                 "B": follow_carrier(last, px[a - 1:b], py[a - 1:b]),
                 "C": pressure_point(last, px[a:b], py[a:b], sm["player_vx"][a:b], sm["player_vy"][a:b],
                                     sm["player_speed"][a:b], team)}
        for name, path in paths.items():
            errors[name].append(np.hypot(*(path - truth).T))
    return {k: np.array(v) for k, v in errors.items()}


def interval(errors, rng):
    """95% range of the median, resampling whole gaps: frames inside one gap are not independent."""
    picks = rng.integers(0, len(errors), (BOOTSTRAPS, len(errors)))
    medians = [np.median(errors[p]) for p in picks]
    return np.percentile(medians, [2.5, 97.5])


def main():
    sm = dict(np.load(os.path.join(CACHE_DIR, "smooth.npz")))
    raw = np.load(os.path.join(CACHE_DIR, "ball.npz"))
    hidden = np.load(os.path.join(CACHE_DIR, "hidden_gaps.npz"))
    x_true, y_true = raw["x"].astype(np.float64), raw["y"].astype(np.float64)
    rng = np.random.default_rng(SEED)

    print("gap    n   baseline   median m  (95% range)    mean m   first frame m")
    rows = []
    for seconds in GAP_SECONDS:
        gaps = hidden["gap_{}".format(seconds)]
        if not len(gaps):
            print("{:2d} s    0".format(seconds))
            continue
        errors = run_length(seconds, gaps, sm, x_true, y_true)
        for name, e in errors.items():
            low, high = interval(e, rng)
            print("{:2d} s {:4d}      {}      {:6.1f}   ({:4.1f} - {:4.1f})   {:6.1f}    {:6.1f}".format(
                seconds, len(gaps), name, np.median(e), low, high, e.mean(), np.median(e[:, 0])))
            rows.append({"gap_s": seconds, "n": len(gaps), "baseline": name, "median_m": round(float(np.median(e)), 2),
                         "median_low": round(float(low), 2), "median_high": round(float(high), 2),
                         "mean_m": round(float(e.mean()), 2)})
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nwrote cache/ball_baselines.csv")


if __name__ == "__main__":
    main()
