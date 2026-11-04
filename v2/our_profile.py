"""Stage 0: the shape of our match's tracking, as totals, so a degraded Sportec copy can copy it.

    .venv\Scripts\python.exe v2/our_profile.py

Reads cache/ball.npz, players.npz, scored.npy, tags.csv and smooth.npz. Writes cache/v2_profile.npz:
lengths of ball-missing runs and ball-seen runs on scored frames (the gap pattern), players tracked
per frame, the position noise (raw minus a light smoothing), and why the ball goes missing (below).
No labels, no answer key. Prints only totals and percentiles.

measure(m, restarts) is the same set of totals for any match from v2/bench.load, so our match, the
old bench and the new bench are measured by one piece of code (v2/sportec_degrade.py prints the table):
  - gap causes: each missing run is put down to where the ball was last seen (the frame before it):
      out    past a line by more than OUT_CLASS_M (the ball left the pitch)
      line   within NEAR_LINE_M of a line, inside
      crowd  CROWD_N or more players (either team) within CROWD_M of the ball
      none   anything else
    and its frames are counted under that cause (gaps that open a half have no last seen ball: "start");
  - missing share around restarts: frames within RESTART_WIN of a restart (ours: Smart Tagging set
    piece start + 3.5 s; Sportec: its set piece events);
  - still ball: share of seen frames with the smoothed ball under STILL_SPEED;
  - player dropouts: a tracked run that ends inside a half with no teammate track starting within
    RELINK_FRAMES and RELINK_M (that would be the tracker renaming the player, not losing him). The
    rate is per tracked player-minute, by band of y (distance across the pitch from the y = 0
    touchline, the top of the picture) and by distance along the pitch from the ball (the tactical
    camera follows the ball, so its edges are far from the ball in x; only frames with a seen ball).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "v1"))
from smooth import runs, gaussian, centred  # noqa: E402

# Params
FPS = 25
CACHE_DIR = os.path.join(ROOT, "cache")
PITCH = (105.3, 68.0)
NOISE_SIGMA = 3.0  # frames; the smoothing that "noise" is measured against
OUT_MARGIN = 1.0   # metres past a line before a seen ball counts as out of the pitch
OUT_CLASS_M = 0.5  # gap cause "out": same margin as Stage 1's out (a little generous for noise)
NEAR_LINE_M = 5.0
CROWD_M = 5.0      # football sense: a duel plus a teammate in support
CROWD_N = 3
CAUSES = ("out", "line", "crowd", "none")
RESTART_WIN = 2 * FPS
TAG_LEAD = int(3.5 * FPS)  # Smart Tagging opens ~3.5 s before the restart
STILL_SPEED = 1.0          # m/s, as Stage 2's still ball
RELINK_FRAMES = 5
RELINK_M = 3.0
Y_BANDS = (0.0, 10.0, 24.0, 44.0, 58.0)          # from the y = 0 touchline; last band runs to the far one
DX_BANDS = (0.0, 20.0, 35.0)                     # |x - ball x|, metres


def scored_runs(mask, scored):
    """Lengths of True runs of mask that lie inside scored frames (cut at the scored edges)."""
    return np.array([e - s for s, e in runs(mask & scored)])


def noise_rms(x, y):
    """RMS of position minus a Gaussian smoothing, over unbroken stretches of >= 25 frames."""
    w = gaussian(NOISE_SIGMA)
    res = []
    for s, e in runs(~np.isnan(x)):
        if e - s < 25:
            continue
        norm = centred(np.ones(e - s), w)
        rx = x[s:e] - centred(x[s:e], w) / norm
        ry = y[s:e] - centred(y[s:e], w) / norm
        res.append(np.hypot(rx, ry)[10:-10])
    r = np.concatenate(res)
    return float(np.sqrt(np.mean(r ** 2)))


def ball_classes(bx, by, px, py, pitch):
    """Per frame: 0 out, 1 near a line, 2 crowd, 3 none (see CAUSES); -1 where the ball is unknown."""
    L, W = pitch
    inside = np.minimum.reduce([bx, L - bx, by, W - by])
    with np.errstate(invalid="ignore"):
        n_near = (np.hypot(px - bx[:, None], py - by[:, None]) < CROWD_M).sum(axis=1)
        c = np.where(inside < -OUT_CLASS_M, 0, np.where(inside < NEAR_LINE_M, 1,
                     np.where(n_near >= CROWD_N, 2, 3)))
    c[np.isnan(bx)] = -1
    return c


def gap_causes(seen, half, cls):
    """[(start, length, cause)] of each missing run inside a half; cause -1 if it opens the half."""
    out = []
    for s, e in runs(~seen & (half > 0)):
        c = cls[s - 1] if s > 0 and half[s - 1] == half[s] else -1
        out.append((s, e - s, c))
    return out


def player_drops(px, py, team, half):
    """(frame, slot) of each lost player: last tracked frame of a run that ends inside a half and is
    not picked up by a teammate's track nearby (RELINK_*)."""
    tracked = ~np.isnan(px)
    n, k = px.shape
    st = [(s, j) for j in range(k) for s, e in runs(tracked[:, j])]
    st.sort()
    sf = np.array([s for s, _ in st])
    sj = np.array([j for _, j in st])
    drops = []
    for j in range(k):
        for s, e in runs(tracked[:, j]):
            if e >= n or half[e] == 0 or half[e] != half[e - 1]:
                continue
            lo, hi = np.searchsorted(sf, e - RELINK_FRAMES), np.searchsorted(sf, e + RELINK_FRAMES + 1)
            cand = [(f, i) for f, i in zip(sf[lo:hi], sj[lo:hi]) if i != j and team[i] == team[j]]
            if any(np.hypot(px[f, i] - px[e - 1, j], py[f, i] - py[e - 1, j]) < RELINK_M for f, i in cand):
                continue
            drops.append((e - 1, j))
    return drops


def band_of(v, edges):
    return np.searchsorted(np.asarray(edges), v, side="right") - 1


def band_names(edges, last="far"):
    return ["{:.0f}-{}".format(a, "{:.0f}".format(b) if b is not None else last)
            for a, b in zip(edges, list(edges[1:]) + [None])]


def measure(m, restarts):
    """The profile numbers for one match from bench.load (raw positions; smoothed ball speed)."""
    half, pitch = m["half"], m["pitch"]
    sc = half > 0
    bx, by = m["raw_ball_x"].astype(np.float64), m["raw_ball_y"].astype(np.float64)
    px, py = m["raw_player_x"].astype(np.float64), m["raw_player_y"].astype(np.float64)
    seen = ~np.isnan(bx) & sc
    L, W = pitch
    out = {}
    out["ball seen share"] = seen[sc].mean()
    gaps = scored_runs(~seen, sc)
    sr = scored_runs(seen, sc)
    out["gaps per 90 min"] = len(gaps) / (sc.sum() / FPS / 60) * 90
    out["gap median s"] = np.median(gaps) / FPS if len(gaps) else np.nan
    out["seen run median s"] = np.median(sr) / FPS
    cls = ball_classes(bx, by, px, py, pitch)
    gc = gap_causes(seen, half, cls)
    tot = max(1, sum(n for _, n, _ in gc))
    for k, name in enumerate(CAUSES):
        out["gap frames: " + name] = sum(n for _, n, c in gc if c == k) / tot
    out["gap frames: opens a half"] = sum(n for _, n, c in gc if c == -1) / tot
    win = np.zeros(len(bx), dtype=bool)
    for r in restarts:
        win[max(0, r - RESTART_WIN):r + RESTART_WIN + 1] = True
    win &= sc
    out["missing around restarts"] = (~seen[win]).mean()
    out["restarts"] = len(restarts)
    inside = np.minimum.reduce([bx, L - bx, by, W - by])
    out["seen past a line > 1 m"] = (inside[seen] < -OUT_MARGIN).mean()
    spd = np.where(seen & (m["ball_state"] == 0), m["ball_speed"], np.nan)
    ok = ~np.isnan(spd)
    out["still ball share of seen"] = (spd[ok] < STILL_SPEED).mean()
    tracked = ~np.isnan(px) & sc[:, None]
    out["players tracked"] = tracked[sc].sum(axis=1).mean()
    drops = player_drops(np.where(sc[:, None], px, np.nan), py, m["team"], half)
    expo = tracked.sum()
    rate = lambda nd, ne: nd / max(ne, 1) * FPS * 60
    out["player drops per player-min"] = rate(len(drops), expo)
    df, dj = np.array([f for f, _ in drops], dtype=int), np.array([j for _, j in drops], dtype=int)
    yb = band_of(py, Y_BANDS)
    for b, name in enumerate(band_names(Y_BANDS, "68")):
        r = rate(np.sum(yb[df, dj] == b), np.sum(tracked & (yb == b)))
        out["drop rate y " + name] = r / out["player drops per player-min"]
    dx = np.abs(px - bx[:, None])
    xb = band_of(dx, DX_BANDS)
    seen_t = tracked & seen[:, None]
    all_seen = rate(np.sum(seen[df]), seen_t.sum())
    for b, name in enumerate(band_names(DX_BANDS)):
        r = rate(np.sum(seen[df] & (xb[df, dj] == b)), np.sum(seen_t & (xb == b)))
        out["drop rate dx " + name] = r / all_seen
    out["ball noise m"] = noise_rms(bx, by)
    return out, gc, drops


def our_restarts(m):
    t = m["tags"]
    t = t[t["code"] == "SET PIECES"]
    return np.sort(t["start_frame"].astype(int).values + TAG_LEAD)


def main():
    import bench
    m = bench.load("ours")
    scored = m["half"] > 0
    bx = m["raw_ball_x"].astype(np.float64)
    vis = ~np.isnan(bx)
    gaps = scored_runs(~vis, scored)
    seen = scored_runs(vis, scored)
    players = np.load(os.path.join(CACHE_DIR, "players.npz"))
    n_tracked = (~np.isnan(players["x"][scored])).sum(axis=1)
    px, py = players["x"].astype(np.float64), players["y"].astype(np.float64)
    p_noise = np.mean([noise_rms(px[:, k], py[:, k]) for k in range(px.shape[1])
                       if (~np.isnan(px[:, k])).sum() > 1000])
    prof, gc, drops = measure(m, our_restarts(m))
    by_cause = {"gap_lengths_" + name: np.array([n for _, n, c in gc if c == k]) for k, name in enumerate(CAUSES)}
    np.savez(os.path.join(CACHE_DIR, "v2_profile.npz"), gap_lengths=gaps, seen_lengths=seen,
             share_seen=vis[scored].mean(), players_mean=n_tracked.mean(),
             ball_noise=prof["ball noise m"], player_noise=p_noise,
             share_out_of_seen=prof["seen past a line > 1 m"], **by_cause,
             names=np.array(list(prof)), values=np.array(list(prof.values()), dtype=float))

    print("missing runs: {}, length s: median {:.2f}, mean {:.2f}, 90% {:.1f}, max {:.0f}".format(
        len(gaps), *(np.array([np.median(gaps), gaps.mean(), np.percentile(gaps, 90), gaps.max()]) / 25)))
    for k, name in enumerate(CAUSES):
        g = by_cause["gap_lengths_" + name]
        print("  cause {:5s}: {:3d} gaps, median {:.1f} s".format(name, len(g), np.median(g) / FPS if len(g) else 0))
    print("player noise RMS {:.3f} m".format(p_noise))
    for k, v in prof.items():
        print("  {:32s} {:.4g}".format(k, v))


if __name__ == "__main__":
    main()
