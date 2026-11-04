"""Stage 3 (owner version) cues: who clearly has the ball, read from where the seen ball is.

    .venv\\Scripts\\python.exe v2/owner.py         # transfer check: ours against the degraded bench

Replaces v2/touches.py (touches need sharp changes of ball speed, which our smoother tracker does
not show). Only frames where the ball is really seen (v1 smoothing state 0) are read.

    owner(m, r, margin)  -> per frame: slot of the player who has the ball, -1 if nobody clearly does
      the nearest player has it when the ball is within r of him and the nearest opponent is at
      least margin further away (a crowded spot has no owner: no clue, by choice);
    episodes(own, team)  -> runs of one owner lasting at least MIN_HOLD frames (shorter is flicker)
                            as (start, end, slot, team), and the start of each turnover: an owner
                            whose team differs from the previous owner's (a change inside a team
                            is a pass, possession stays). Ownership is forgotten across an unscored
                            frame, so nothing chains over half time.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
from bench import MATCHES  # noqa: E402
from smooth import runs  # noqa: E402

# Params
FPS = 25
MIN_HOLD = 3             # frames (0.12 s): shorter owner runs are tracking flicker
RADII = (1.0, 1.5, 2.0)  # m: about a stride; tried in the grid
MARGINS = (0.0, 0.25, 0.5, 1.0, 2.0)   # 0 and 0.25 added: the first grid picked its edge, 0.5
CHECK = (1.5, 1.0)       # radius, margin used for the transfer check


def distances_all(m):
    """(frames, slots) distance from the seen ball to every tracked player; NaN where unknown."""
    seen = (m["ball_state"] == 0) & (m["half"] > 0)
    bx = np.where(seen, m["ball_x"], np.nan).astype(np.float64)
    by = np.where(seen, m["ball_y"], np.nan).astype(np.float64)
    return np.hypot(m["player_x"] - bx[:, None], m["player_y"] - by[:, None])


def owner(d, team, r, margin):
    inf = np.where(np.isnan(d), np.inf, d)
    j = inf.argmin(axis=1)
    d1 = inf[np.arange(len(j)), j]
    opp = np.full(len(j), np.inf)
    for t in (0, 1):
        dt = inf[:, team == t].min(axis=1)
        opp = np.where(team[j] != t, dt, opp)
    return np.where((d1 < r) & (opp - d1 >= margin), j, -1)


def episodes(own, team, half):
    eps, turn = [], []
    prev_team = -1
    for s, e in runs(own >= 0):
        if s > 0 and half[s - 1] == 0:
            prev_team = -1
        # split a run where the owner changes
        o = own[s:e]
        cuts = np.r_[0, np.flatnonzero(np.diff(o)) + 1, e - s]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < MIN_HOLD:
                continue
            slot = o[a]
            t = int(team[slot])
            if prev_team >= 0 and t != prev_team:
                turn.append((s + a, t))
            eps.append((s + a, s + b, slot, t))
            prev_team = t
        if e < len(half) and half[e] == 0:
            prev_team = -1
    return eps, turn


def check():
    rows = []
    for name in ("ours",) + tuple(mt + "~" for mt in MATCHES):
        m = bench.load(name)
        d = distances_all(m)
        own = owner(d, m["team"], *CHECK)
        seen = (m["ball_state"] == 0) & (m["half"] > 0)
        eps, turn = episodes(own, m["team"], m["half"])
        minutes = (m["half"] > 0).sum() / FPS / 60
        changes = sum(1 for a, b in zip(eps[:-1], eps[1:]) if a[2] != b[2])
        rows.append(dict(match=name, owner_share_of_seen=(own[seen] >= 0).mean(),
                         crowded_share=((np.sort(np.where(np.isnan(d), np.inf, d), axis=1)[:, 1] < CHECK[0] + CHECK[1])[seen]).mean(),
                         owner_runs_per_min=len(eps) / minutes, owner_run_median_s=np.median([b - a for a, b, _, _ in eps]) / FPS,
                         owner_changes_per_min=changes / minutes, turnovers_per_min=len(turn) / minutes,
                         passes_per_min=(changes - len(turn)) / minutes))
    t = pd.DataFrame(rows)
    bench_mean = t[t.match != "ours"].mean(numeric_only=True)
    bench_mean["match"] = "bench mean"
    return pd.concat([t, bench_mean.to_frame().T], ignore_index=True)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 3)
    print("radius {} m, margin {} m".format(*CHECK))
    print(check().to_string(index=False))
