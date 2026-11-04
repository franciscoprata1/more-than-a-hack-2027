"""Stage 3 cues: touches, possession zone and duels, read from the seen ball only.

    .venv\\Scripts\\python.exe v2/touches.py          # touch check against Sportec's on-ball events

Used by v2/stage3.py. Only frames where the ball is really seen (v1 smoothing state 0, never filled
or missing) are used, as v1 said: never detect touches in a gap.

    touches(m, a_touch, dist)  -> frames, team (0 H / 1 A), duel (both teams close), confirmed
      a touch is a peak of the ball's change of velocity (|v after - v before|, each over K frames)
      above a_touch m/s, with the nearest player within R_TOUCH; it belongs to that player's team.
      Two-touch rule: a touch is "confirmed" (the team has the ball, not just a deflection) when the
      same team touches again within CONFIRM_S, or keeps the ball in its zone for HOLD_S after it.
    zone(dist)                 -> per frame 0 / 1 (only that team within ZONE_M of the ball),
                                  2 duel (both), -1 nobody close
dist is stage1.distances(m): nearest home and away player to the (smoothed) ball, FAR if none.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import deadball as db  # noqa: E402
import stage1 as s1  # noqa: E402
from bench import MATCHES  # noqa: E402

# Params
FPS = 25
K = 3                    # frames each side for the velocity before and after (0.12 s)
R_TOUCH = 1.5            # m: a touch needs a player this close (a stretched leg plus tracking error)
ZONE_M = 1.0             # m: the ball at a player's feet
PEAK_GAP = 5             # frames: two touches closer than this are one
CONFIRM_S = 3.0
HOLD_S = 0.4
A_TOUCHES = (2.0, 4.0, 7.0)   # m/s, tried in the Stage 3 grid
EVENT_TOUCHES = ("Play", "OtherBallAction", "ShotAtGoal", "BallClaiming", "ThrowIn", "FreeKick",
                 "GoalKick", "CornerKick", "KickOff")
MATCH_FRAMES = 25        # an event within 1 s of a touch is that touch (hand-set event clocks scatter;
                         # no steady offset: median -1 to +2 frames)


def velocity_change(m):
    """|v after - v before| per frame (m/s), NaN unless the ball is seen from t-K to t+K."""
    seen = (m["ball_state"] == 0) & (m["half"] > 0) & ~np.isnan(m["ball_x"])
    x, y = m["ball_x"].astype(np.float64), m["ball_y"].astype(np.float64)
    n = len(x)
    dv = np.full(n, np.nan)
    t = np.arange(K, n - K)
    vbx, vby = (x[t] - x[t - K]) * FPS / K, (y[t] - y[t - K]) * FPS / K
    vax, vay = (x[t + K] - x[t]) * FPS / K, (y[t + K] - y[t]) * FPS / K
    dv[t] = np.hypot(vax - vbx, vay - vby)
    ok = db.held_for(seen, 2 * K + 1)          # seen over [t-K, t+K]
    ok = np.r_[ok[K:], np.zeros(K, bool)]
    dv[~ok] = np.nan
    return dv


def zone(dist):
    dh, da, _ = dist
    h, a = dh < ZONE_M, da < ZONE_M
    return np.where(h & a, 2, np.where(h, 0, np.where(a, 1, -1)))


def touches(m, a_touch, dist, dv=None):
    dv = velocity_change(m) if dv is None else dv
    v = np.nan_to_num(dv, nan=0.0)
    big = v > a_touch
    w = 2 * PEAK_GAP + 1
    pad = np.r_[np.zeros(PEAK_GAP), v, np.zeros(PEAK_GAP)]
    local = np.lib.stride_tricks.sliding_window_view(pad, w).max(axis=1)
    f = np.flatnonzero(big & (v >= local))
    dh, da, _ = dist
    near = np.minimum(dh[f], da[f]) < R_TOUCH
    f = f[near]
    team = (da[f] < dh[f]).astype(np.int8)
    duel = (dh[f] < R_TOUCH) & (da[f] < R_TOUCH)
    z = zone(dist)
    hold = int(HOLD_S * FPS)
    conf = np.zeros(len(f), dtype=bool)
    for i, (fr, t) in enumerate(zip(f, team)):
        nxt = i + 1
        if nxt < len(f) and f[nxt] - fr <= CONFIRM_S * FPS and team[nxt] == t:
            conf[i] = True
        elif (z[fr + 1:fr + 1 + hold] == t).all():
            conf[i] = True
    return f, team, duel, conf


def check(a_touches=A_TOUCHES):
    """Touches found against Sportec's on-ball events (degraded bench, event frame seen)."""
    rows = []
    for mt in MATCHES:
        m = bench.load(mt + "~")
        dist = s1.distances(m)
        dv = velocity_change(m)
        ev = m["events"]
        ev = ev[ev["kind"].isin(EVENT_TOUCHES) & ev["team"].isin(["H", "A"]) & (ev["in_half"] == 1)]
        fr = ev["frame"].values.astype(int)
        seen_ev = ~np.isnan(dv[fr])
        fr, et = fr[seen_ev], (ev["team"].values[seen_ev] == "A").astype(int)
        minutes = (m["half"] > 0).sum() / FPS / 60
        for a in a_touches:
            f, team, duel, conf = touches(m, a, dist, dv)
            j = np.searchsorted(f, fr)
            best = np.full(len(fr), -1)
            for k, (x, jj) in enumerate(zip(fr, j)):
                c = [i for i in (jj - 1, jj) if 0 <= i < len(f) and abs(f[i] - x) <= MATCH_FRAMES]
                if c:
                    best[k] = min(c, key=lambda i: abs(f[i] - x))
            hit = best >= 0
            used = np.zeros(len(f), bool)
            used[best[hit]] = True
            rows.append(dict(match=mt, a_touch=a, touches_per_min=len(f) / minutes,
                             event_recall=hit.mean(), touch_on_event=used.mean(),
                             team_right=(team[best[hit]] == et[hit]).mean(),
                             confirmed=conf.mean(), duel=duel.mean()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)
    t = check()
    print(t.to_string(index=False))
    print(t.groupby("a_touch").mean(numeric_only=True).to_string())
