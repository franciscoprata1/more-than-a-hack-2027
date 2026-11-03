"""Stage 0: the shape of our match's tracking, as totals, so a degraded Sportec copy can copy it.

    .venv\Scripts\python.exe v2/our_profile.py

Reads cache/ball.npz, players.npz, scored.npy (raw, not smoothed). Writes cache/v2_profile.npz:
lengths of ball-missing runs and ball-seen runs on scored frames (the gap pattern), players tracked
per frame, and the position noise (raw minus a light smoothing). No labels, no answer key.
Prints only totals and percentiles.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "v1"))
from smooth import runs, gaussian, centred  # noqa: E402

# Params
CACHE_DIR = os.path.join(ROOT, "cache")
PITCH = (105.3, 68.0)
NOISE_SIGMA = 3.0  # frames; the smoothing that "noise" is measured against
OUT_MARGIN = 1.0   # metres past a line before a seen ball counts as out of the pitch


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


def main():
    ball = np.load(os.path.join(CACHE_DIR, "ball.npz"))
    players = np.load(os.path.join(CACHE_DIR, "players.npz"))
    scored = np.load(os.path.join(CACHE_DIR, "scored.npy"))
    bx, by = ball["x"].astype(np.float64), ball["y"].astype(np.float64)
    vis = ~np.isnan(bx)

    gaps = scored_runs(~vis, scored)
    seen = scored_runs(vis, scored)
    out = vis & scored & ((bx < -OUT_MARGIN) | (bx > PITCH[0] + OUT_MARGIN)
                          | (by < -OUT_MARGIN) | (by > PITCH[1] + OUT_MARGIN))
    n_tracked = (~np.isnan(players["x"][scored])).sum(axis=1)
    px = players["x"].astype(np.float64)
    py = players["y"].astype(np.float64)
    p_noise = np.mean([noise_rms(px[:, k], py[:, k]) for k in range(px.shape[1])
                       if (~np.isnan(px[:, k])).sum() > 1000])
    b_noise = noise_rms(bx, by)

    np.savez(os.path.join(CACHE_DIR, "v2_profile.npz"), gap_lengths=gaps, seen_lengths=seen,
             share_seen=vis[scored].mean(), players_mean=n_tracked.mean(),
             ball_noise=b_noise, player_noise=p_noise, share_out_of_seen=out.sum() / (vis & scored).sum())

    print("ball seen on {:.1%} of scored frames".format(vis[scored].mean()))
    print("missing runs: {}, length s: median {:.2f}, mean {:.2f}, 90% {:.1f}, max {:.0f}".format(
        len(gaps), *(np.array([np.median(gaps), gaps.mean(), np.percentile(gaps, 90), gaps.max()]) / 25)))
    print("  share of missing frames in runs <= 1 s: {:.1%}, > 10 s: {:.1%}".format(
        gaps[gaps <= 25].sum() / gaps.sum(), gaps[gaps > 250].sum() / gaps.sum()))
    print("seen runs: {}, length s: median {:.2f}, mean {:.2f}, 90% {:.1f}".format(
        len(seen), np.median(seen) / 25, seen.mean() / 25, np.percentile(seen, 90) / 25))
    print("seen ball beyond a line by > {} m: {:.2%} of seen scored frames".format(
        OUT_MARGIN, out.sum() / (vis & scored).sum()))
    print("players tracked per scored frame: mean {:.2f}, 10% {:.0f}, 90% {:.0f}".format(
        n_tracked.mean(), *np.percentile(n_tracked, [10, 90])))
    print("noise RMS (raw - smooth sigma {} fr): ball {:.3f} m, players {:.3f} m".format(
        NOISE_SIGMA, b_noise, p_noise))


if __name__ == "__main__":
    main()
