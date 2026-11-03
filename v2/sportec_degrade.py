"""Stage 0: a degraded copy of each Sportec match that looks like our match's tracking.

    .venv\\Scripts\\python.exe v2/sportec_degrade.py [MATCH ...]

Needs cache/sportec/<match>/ (v2/sportec_cache.py) and cache/v2_profile.npz (v2/our_profile.py,
totals of our match only). Writes cache/sportec_degraded/<match>/ with the same files; labels,
events, half and pitch are copied unchanged.

What is copied from our match (choices, written down in the action plan):
  - ball gaps: alternate seen and missing runs, each length drawn from our match's own run lengths
    (so the share seen is ~59% and a typical gap ~11 s, almost none shorter than 1 s);
  - gaps start near a line more often than chance (43% of ours do, 20% of seen frames are near a
    line): with probability NEAR_LINE_SHIFT a gap start moves to the nearest frame within 2 s where
    the ball is within 5 m of a line;
  - the ball past a line: ours is seen there on 2.7% of seen frames, Sportec on ~6%, so each stretch
    of ball more than 1 m out is hidden whole with probability OUT_HIDE;
  - players: each player drops out in runs (mean OFF_MEAN_S) so that ~20.5 of 22 are tracked, as ours;
  - noise: none added. Sportec is already noisier than ours (ball 0.30-0.37 m against 0.17 m, raw
    minus a light smoothing), so adding any would move it further away.
Fixed seed per match. Prints totals.
"""

import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "v1"))
from sportec_download import MATCHES  # noqa: E402
from smooth import runs  # noqa: E402

# Params
FPS = 25
SRC = os.path.join(ROOT, "cache", "sportec")
DST = os.path.join(ROOT, "cache", "sportec_degraded")
PROFILE = os.path.join(ROOT, "cache", "v2_profile.npz")
SEED = 2027
NEAR_LINE_M = 5.0
NEAR_LINE_SHIFT = 0.35   # set so ~43% of gaps start near a line, as ours (printed below)
SHIFT_WIN = 2 * FPS
OUT_M = 1.0
OUT_HIDE = 0.55          # set so ~2.7% of seen ball frames are past a line, as ours (printed below)
OFF_MEAN_S = 3.0         # a player lost by the tracker comes back after a few seconds
COPY = ("labels.npy", "half.npy", "scored.npy", "pitch.npy", "events.csv")


def line_distance(x, y, pitch):
    L, W = pitch
    return np.minimum.reduce([x, L - x, y, W - y])


def ball_mask(n_half, near, rng, prof):
    """Seen (True) / hidden (False) for one half, from our run lengths, gap starts pulled to lines."""
    seen = np.ones(n_half, dtype=bool)
    t = 0
    state = rng.random() < prof["share_seen"]
    while t < n_half:
        if state:
            t += int(rng.choice(prof["seen_lengths"]))
        else:
            start = t
            if rng.random() < NEAR_LINE_SHIFT:
                lo, hi = max(0, t - SHIFT_WIN), min(n_half, t + SHIFT_WIN)
                cand = np.flatnonzero(near[lo:hi]) + lo
                if len(cand):
                    start = int(cand[np.argmin(np.abs(cand - t))])
            gap = int(rng.choice(prof["gap_lengths"]))
            seen[start:start + gap] = False
            t = max(t, start) + gap
        state = not state
    return seen


def player_mask(on, rng, share_off):
    """Tracked (True) for one player's on-pitch frames: off runs of mean OFF_MEAN_S."""
    tracked = on.copy()
    off_mean = OFF_MEAN_S * FPS
    on_mean = off_mean * (1 - share_off) / share_off
    for s, e in runs(on):
        t = s + int(rng.exponential(on_mean))
        while t < e:
            gap = max(1, int(rng.exponential(off_mean)))
            tracked[t:min(e, t + gap)] = False
            t += gap + int(rng.exponential(on_mean))
    return tracked


def degrade(match, k, prof):
    rng = np.random.default_rng(SEED + k)
    src, dst = os.path.join(SRC, match), os.path.join(DST, match)
    os.makedirs(dst, exist_ok=True)
    for f in COPY:
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    ball = dict(np.load(os.path.join(src, "ball.npz")))
    pl = dict(np.load(os.path.join(src, "players.npz")))
    half, pitch = np.load(os.path.join(src, "half.npy")), np.load(os.path.join(src, "pitch.npy"))
    bx, by = ball["x"].astype(np.float64), ball["y"].astype(np.float64)
    dist = line_distance(bx, by, pitch)

    seen = np.zeros(len(bx), dtype=bool)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        seen[idx] = ball_mask(len(idx), dist[idx] < NEAR_LINE_M, rng, prof)
    for s, e in runs(dist < -OUT_M):
        if rng.random() < OUT_HIDE:
            seen[s:e] = False
    seen &= half > 0

    share_off = 1 - prof["players_mean"] / 22.0
    px, py = pl["x"].copy(), pl["y"].copy()
    for j in range(px.shape[1]):
        keep = player_mask(~np.isnan(px[:, j]), rng, share_off)
        px[~keep, j] = np.nan
        py[~keep, j] = np.nan

    def speed(x, y):
        v = np.full(x.shape, np.nan, dtype=np.float32)
        v[1:] = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)) * FPS
        return v

    bx[~seen], by[~seen] = np.nan, np.nan
    z = ball["z"].copy()
    z[~seen] = np.nan
    np.savez_compressed(os.path.join(dst, "ball.npz"), x=bx.astype(np.float32), y=by.astype(np.float32),
                        z=z, speed=speed(bx, by), visible=seen)
    np.savez_compressed(os.path.join(dst, "players.npz"), x=px, y=py, speed=speed(px, py),
                        team=pl["team"], person=pl["person"])

    sc = half > 0
    gaps = np.array([e - s for s, e in runs(~seen & sc)])
    starts = [s for s, e in runs(~seen & sc) if s > 0 and half[s - 1] > 0]
    print("{}  ball seen {:.1%}  gaps {} median {:.1f} s  gap starts near a line {:.0%}  seen past a line "
          "{:.1%}  players tracked {:.2f}".format(
              match, seen[sc].mean(), len(gaps), np.median(gaps) / FPS,
              np.mean(dist[np.array(starts) - 1] < NEAR_LINE_M),
              np.mean(dist[seen] < -OUT_M), (~np.isnan(px[sc])).sum(1).mean()), flush=True)


if __name__ == "__main__":
    prof = dict(np.load(PROFILE))
    for k, m in enumerate(MATCHES):
        if len(sys.argv) == 1 or m in sys.argv[1:]:
            degrade(m, k, prof)
