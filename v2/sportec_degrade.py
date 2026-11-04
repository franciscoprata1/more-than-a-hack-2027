"""Stage 0: a degraded copy of each Sportec match that looks like our match's tracking (version 2).

    .venv\\Scripts\\python.exe v2/sportec_degrade.py

Needs cache/sportec/<match>/ (v2/sportec_cache.py) and cache/v2_profile.npz (v2/our_profile.py,
totals of our match only). Writes cache/sportec_degraded/<match>/ with the same files; labels,
events, half and pitch are copied unchanged. Then run v2/bench_profile.py for the table.

What is copied from our match (choices, written down in the action plan):
  - the dead ball (kept from version 1): Sportec freezes the ball where play stopped for the whole
    stoppage, so in every stoppage the ball rolls on OVERRUN_M past the line if it went out, wanders
    to the restart spot at about CARRY_SPEED, then lies there for 1-3 s with 1 cm jitter. New: the
    walk stays within WANDER_OUT_M of the lines (once fetched, the ball is brought back), and
    CARRY_SPEED is chosen from CARRY_SPEEDS so the share of seen frames with a still ball (< 1 m/s)
    is ours (10%);
  - ball gaps by cause: every frame where the ball is seen may start a gap, with a probability that
    depends on where the ball is (our_profile.ball_classes: out, near a line, crowd, none) and is
    raised by K_RESTART within 2 s of a set piece restart (Sportec's set piece events). Each gap's
    length is drawn from our gaps of the same cause. On top, each stretch of ball more than 1 m out
    is hidden with chance OUT_HIDE, from up to OUT_EDGE_S before it to up to OUT_EDGE_S after (the
    camera loses the ball near the line first). The four probabilities, K_RESTART and OUT_HIDE are
    fitted (all 7 matches pooled) so the bench has our share of ball seen, our split of gap frames
    by cause, our missing share around restarts and our share of seen ball past a line. Version 1's
    random gap placement and near-line shift are gone;
  - players: each player is lost with a probability per frame that depends on his band across the
    pitch (y) and his distance along the pitch from the ball (dx), and stays lost for an exponential
    time of mean OFF_MEAN. Band factors, the base rate and OFF_MEAN are fitted so the bench has our
    drop rate per player-minute, our relative rate in each band and our players tracked per frame;
  - noise: none added (Sportec is already noisier than ours).
Fitting uses only totals of our match (cache/v2_profile.npz) and Sportec. The fit is saved in
cache/v2_degrade_fit.json. Fixed seed per match.
"""

import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "v1"))
import our_profile as op  # noqa: E402
from sportec_download import MATCHES  # noqa: E402
from smooth import runs, fill_ball, smooth_and_velocity, BALL_SIGMA, MISSING  # noqa: E402

# Params
FPS = 25
SRC = os.path.join(ROOT, "cache", "sportec")
DST = os.path.join(ROOT, "cache", "sportec_degraded")
PROFILE = os.path.join(ROOT, "cache", "v2_profile.npz")
FIT = os.path.join(ROOT, "cache", "v2_degrade_fit.json")
SEED = 2027
PLACED_S = (1.0, 3.0)    # the ball lies at the restart spot this long before the restart
CARRY_SPEEDS = (1.2, 1.6, 2.0, 2.5, 3.0, 4.0)   # m/s, the one giving our still share is kept
CARRY_SIGMA = 25         # frames: how smoothly its heading changes
JITTER_M = 0.01
OVERRUN_M = 3.0          # a ball that went out rolls on past the line before it is fetched
OVERRUN_S = 1.0
WANDER_OUT_M = 1.0       # the walk to the restart spot stays within this of the lines
OUT_EDGE_S = 2.0         # a hidden exit starts up to this long before the ball is out, ends this long after
FIT_ITERS = 25
TARGETS = ("gap frames: out", "gap frames: line", "gap frames: crowd", "gap frames: none",
           "ball seen share", "missing around restarts", "still ball share of seen",
           "players tracked", "player drops per player-min")
COPY = ("labels.npy", "half.npy", "scored.npy", "pitch.npy", "events.csv")


def line_distance(x, y, pitch):
    L, W = pitch
    return np.minimum.reduce([x, L - x, y, W - y])


def fold(v, lo, hi):
    """Reflect values back inside [lo, hi] (a start or end already outside stays where it is)."""
    lo, hi = min(lo, v[0], v[-1]), max(hi, v[0], v[-1])
    v = np.where(v < lo, 2 * lo - v, v)
    return np.clip(np.where(v > hi, 2 * hi - v, v), lo, hi)


def dead_ball_path(bx, by, labels, half, pitch, rng, carry_speed):
    """Replace Sportec's frozen dead ball by a walk to the restart spot, then a short rest there."""
    bx, by = bx.copy(), by.copy()
    for s, e in runs((labels == 2) & (half > 0)):
        if e >= len(bx) or half[e] != half[s]:
            continue  # no restart in this half to walk to
        n = e - s
        rest = min(n // 2, int(rng.uniform(*PLACED_S) * FPS))
        x0, y0 = bx[s], by[s]
        go = 0
        if line_distance(np.array([x0]), np.array([y0]), pitch)[0] < 1.5 and n > 3 * OVERRUN_S * FPS:
            go = int(OVERRUN_S * FPS)  # outwards, perpendicular to the nearest line
            L, W = pitch
            k = int(np.argmin([x0, L - x0, y0, W - y0]))
            ox, oy = [(-1, 0), (1, 0), (0, -1), (0, 1)][k]
            t = np.arange(1, go + 1) / go
            bx[s:s + go] = x0 + t * ox * OVERRUN_M
            by[s:s + go] = y0 + t * oy * OVERRUN_M
            x0, y0 = bx[s + go - 1], by[s + go - 1]
        move = n - rest - go
        if move > 0:
            # A wandering walk from (x0, y0) that ends at the spot: smooth random steps at about
            # carry_speed, then the drift is taken out so it arrives (a Brownian bridge).
            v = rng.normal(0, 1, (move + 6 * CARRY_SIGMA, 2))
            k = np.exp(-0.5 * (np.arange(-3 * CARRY_SIGMA, 3 * CARRY_SIGMA + 1) / CARRY_SIGMA) ** 2)
            v = np.stack([np.convolve(v[:, j], k, mode="valid")[:move] for j in (0, 1)], axis=1)
            v *= carry_speed / FPS / np.mean(np.hypot(v[:, 0], v[:, 1]))
            path = np.cumsum(v, axis=0)
            t = (np.arange(1, move + 1) / move)[:, None]
            path = path - t * path[-1] + t * np.array([bx[e] - x0, by[e] - y0])
            # Once fetched the ball is brought back: the walk folds back at WANDER_OUT_M past a line.
            L, W = pitch
            bx[s + go:s + go + move] = fold(x0 + path[:, 0], -WANDER_OUT_M, L + WANDER_OUT_M)
            by[s + go:s + go + move] = fold(y0 + path[:, 1], -WANDER_OUT_M, W + WANDER_OUT_M)
        move = max(move, 0) + go
        bx[s + move:e], by[s + move:e] = bx[e], by[e]
        bx[s:e] += rng.normal(0, JITTER_M, n)
        by[s:e] += rng.normal(0, JITTER_M, n)
    return bx, by


def restart_window(frames, half):
    w = np.zeros(len(half), dtype=bool)
    for r in frames:
        w[max(0, r - op.RESTART_WIN):r + op.RESTART_WIN + 1] = True
    return w & (half > 0)


def ball_mask(cls, rwin, half, hz, k_restart, lengths, rng):
    """Seen (True) / hidden (False): a gap may start after any seen frame, chance hz[cause] (times
    k_restart near a restart), length drawn from our gaps of that cause."""
    n = len(cls)
    p = np.clip(hz[np.maximum(cls, 0)] * np.where(rwin, k_restart, 1.0), 0, 0.5)
    p[(cls < 0) | (half == 0)] = 0
    u, pick = rng.random(n), rng.random(n)
    seen = half > 0
    t = 0
    for c in np.flatnonzero(u < p):
        if c < t:
            continue
        g = lengths[cls[c]]
        ln = int(g[int(pick[c] * len(g))])
        end = c + 1 + ln
        h = half[c]
        seen[c + 1:end] &= half[c + 1:end] != h
        t = end + 1
    return seen


def player_mask(on, band, base, factor, off_mean, rng):
    """Tracked (True) for one player: lost with chance base * factor[band] per frame, for an
    exponential time of mean off_mean frames. Returns (tracked, list of loss frames)."""
    n = len(on)
    p = np.where(on, base * factor[band], 0.0)
    u, e = rng.random(n), rng.exponential(1.0, n)
    tracked = on.copy()
    drops = []
    t = 0
    for c in np.flatnonzero(u < p):
        if c < t:
            continue
        ln = max(1, int(e[c] * off_mean))
        stop = c + 1 + ln
        seg = tracked[c + 1:stop]
        k = np.flatnonzero(~on[c + 1:stop])
        if len(k):
            seg = seg[:k[0]]
        seg[:] = False
        drops.append(c)
        t = stop + 1
    return tracked, drops


class Match:
    """What the fit needs of one Sportec match, computed once."""

    def __init__(self, k, mt):
        self.k, self.mt = k, mt
        src = os.path.join(SRC, mt)
        self.ball = dict(np.load(os.path.join(src, "ball.npz")))
        self.pl = dict(np.load(os.path.join(src, "players.npz")))
        self.half = np.load(os.path.join(src, "half.npy"))
        self.pitch = tuple(np.load(os.path.join(src, "pitch.npy")))
        self.labels = np.load(os.path.join(src, "labels.npy"))
        import pandas as pd
        ev = pd.read_csv(os.path.join(src, "events.csv"), keep_default_na=False)
        ev = ev[(ev["set_piece"] == 1) & (ev["in_half"] == 1)]
        self.rwin = restart_window(ev["frame"].values.astype(int), self.half)
        px, py = self.pl["x"].astype(np.float64), self.pl["y"].astype(np.float64)
        self.yb = op.band_of(py, op.Y_BANDS)

    def set_carry(self, speed):
        rng = np.random.default_rng(SEED + self.k)
        self.bx, self.by = dead_ball_path(self.ball["x"].astype(np.float64), self.ball["y"].astype(np.float64),
                                          self.labels, self.half, self.pitch, rng, speed)
        px, py = self.pl["x"].astype(np.float64), self.pl["y"].astype(np.float64)
        self.cls = op.ball_classes(self.bx, self.by, px, py, self.pitch)
        self.xb = op.band_of(np.abs(px - self.bx[:, None]), op.DX_BANDS)
        self.out = (line_distance(self.bx, self.by, self.pitch) < -op.OUT_MARGIN) & (self.half > 0)
        self.out_runs = runs(self.out)

    def ball_seen(self, hz, k_restart, out_hide, lengths):
        rng = np.random.default_rng(SEED + 100 + self.k)
        seen = ball_mask(self.cls, self.rwin, self.half, hz, k_restart, lengths, rng)
        # A ball that leaves the pitch may be lost with it: the camera tends to lose it near the
        # line first, so the hidden stretch opens up to OUT_EDGE_S before it is 1 m out and closes
        # up to OUT_EDGE_S after it is back.
        r2 = np.random.default_rng(SEED + 300 + self.k)
        u, before, after = (r2.random(len(self.out_runs)) for _ in range(3))
        edge = OUT_EDGE_S * FPS
        for (s, e), ui, b, a in zip(self.out_runs, u, before, after):
            if ui < out_hide:
                lo, hi = max(0, s - int(b * edge)), e + int(a * edge)
                seen[lo:hi] &= self.half[lo:hi] != self.half[s]
        return seen

    def players_tracked(self, base, fy, fdx, off_mean):
        rng = np.random.default_rng(SEED + 200 + self.k)
        px = self.pl["x"]
        on = ~np.isnan(px) & (self.half > 0)[:, None]
        tracked = np.zeros_like(on)
        drops = []
        nb_x = len(op.DX_BANDS)
        for j in range(px.shape[1]):
            band = np.clip(self.yb[:, j], 0, len(fy) - 1) * nb_x + np.clip(self.xb[:, j], 0, nb_x - 1)
            factor = np.outer(fy, fdx).ravel()
            tr, dr = player_mask(on[:, j], band, base, factor, off_mean, rng)
            tracked[:, j] = tr
            drops += [(f, j) for f in dr]
        return on, tracked, drops


def ball_totals(ms, masks):
    """Pooled gap totals over matches: seen share, gap frames per cause, missing around restarts."""
    sc_n = seen_n = rw_n = rw_miss = out_n = 0
    frames = np.zeros(4)
    for m, seen in zip(ms, masks):
        sc = m.half > 0
        sc_n += sc.sum()
        seen_n += seen[sc].sum()
        out_n += (seen & m.out).sum()
        rw_n += m.rwin.sum()
        rw_miss += (~seen[m.rwin]).sum()
        for _, ln, c in op.gap_causes(seen, m.half, m.cls):
            if c >= 0:
                frames[c] += ln
    return seen_n / sc_n, frames / frames.sum(), rw_miss / rw_n, out_n / seen_n


def still_share(ms, masks):
    """Share of seen frames whose smoothed ball (as v2/bench.py smooths it) is under 1 m/s."""
    still = n = 0
    for m, seen in zip(ms, masks):
        x, y = np.where(seen, m.bx, np.nan), np.where(seen, m.by, np.nan)
        fx, fy, state, _ = fill_ball(x, y, m.half > 0)
        _, _, vx, vy = smooth_and_velocity(fx, fy, runs(state != MISSING), BALL_SIGMA)
        ok = seen & (state == 0)
        still += (np.hypot(vx, vy)[ok] < op.STILL_SPEED).sum()
        n += ok.sum()
    return still / n


def fit_ball(ms, target, lengths, hz=None, k_restart=1.0, out_hide=0.5):
    """Multiplicative updates until the bench's gap totals are ours."""
    t_frames = np.array([target["gap frames: " + c] for c in op.CAUSES])
    t_frames /= t_frames.sum()
    hz = np.full(4, 2e-3) if hz is None else hz.copy()
    for it in range(FIT_ITERS):
        masks = [m.ball_seen(hz, k_restart, out_hide, lengths) for m in ms]
        share, frames, rmiss, out = ball_totals(ms, masks)
        miss_ratio = (1 - target["ball seen share"]) / (1 - share)
        hz *= (t_frames / np.maximum(frames, 1e-4)) ** 0.7 * miss_ratio ** 0.7
        k_restart *= (target["missing around restarts"] / max(rmiss, 1e-3)) ** 1.5 / miss_ratio ** 0.7
        keep = (1 - out_hide) * (target["seen past a line > 1 m"] / max(out, 1e-4)) ** 0.8
        out_hide = float(np.clip(1 - keep, 0, 0.99))
    return hz, k_restart, out_hide, masks, (share, frames, rmiss, out)


def fit_players(ms, target):
    nb_y, nb_x = len(op.Y_BANDS), len(op.DX_BANDS)
    t_y = np.array([target["drop rate y " + n] for n in op.band_names(op.Y_BANDS, "68")])
    t_x = np.array([target["drop rate dx " + n] for n in op.band_names(op.DX_BANDS)])
    fy, fdx = t_y.copy(), t_x.copy()
    rate_t = target["player drops per player-min"] / (FPS * 60)
    base, off_mean = rate_t, 7.0 * FPS
    for it in range(12):
        n_drop, expo = 0, 0
        dy, ey, dx_, ex = np.zeros(nb_y), np.zeros(nb_y), np.zeros(nb_x), np.zeros(nb_x)
        tracked_sum, frames = 0, 0
        for m in ms:
            on, tr, drops = m.players_tracked(base, fy, fdx, off_mean)
            sc = m.half > 0
            tracked_sum += tr[sc].sum()
            frames += sc.sum()
            expo += tr.sum()
            n_drop += len(drops)
            for b in range(nb_y):
                ey[b] += (tr & (m.yb == b)).sum()
            for b in range(nb_x):
                ex[b] += (tr & (m.xb == b)).sum()
            for f, j in drops:
                dy[min(max(m.yb[f, j], 0), nb_y - 1)] += 1
                dx_[min(max(m.xb[f, j], 0), nb_x - 1)] += 1
        rate = n_drop / expo
        ry, rx = (dy / np.maximum(ey, 1)) / rate, (dx_ / np.maximum(ex, 1)) / rate
        tracked = tracked_sum / frames
        fy *= (t_y / np.maximum(ry, 1e-3)) ** 0.8
        fdx *= (t_x / np.maximum(rx, 1e-3)) ** 0.8
        base *= (rate_t / rate) ** 0.8
        lost_t = 21.8 - target["players tracked"]   # 21.8: players on the pitch in clean Sportec
        off_mean *= (lost_t / max(21.8 - tracked, 1e-3)) ** 0.8 * (rate / rate_t) ** 0.8
    return base, fy, fdx, off_mean, (rate * FPS * 60, ry, rx, tracked)


def write(m, seen, tracked):
    dst = os.path.join(DST, m.mt)
    if os.path.isdir(dst):
        shutil.rmtree(dst)  # also drops the old smooth.npz, so v2/bench.py smooths the new copy
    os.makedirs(dst)
    for f in COPY:
        shutil.copy(os.path.join(SRC, m.mt, f), os.path.join(dst, f))
    bx, by = m.bx.copy(), m.by.copy()
    px, py = m.pl["x"].copy(), m.pl["y"].copy()
    px[~tracked], py[~tracked] = np.nan, np.nan

    def speed(x, y):
        v = np.full(x.shape, np.nan, dtype=np.float32)
        v[1:] = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)) * FPS
        return v

    bx[~seen], by[~seen] = np.nan, np.nan
    z = m.ball["z"].copy()
    z[~seen] = np.nan
    np.savez_compressed(os.path.join(dst, "ball.npz"), x=bx.astype(np.float32), y=by.astype(np.float32),
                        z=z, speed=speed(bx, by), visible=seen)
    np.savez_compressed(os.path.join(dst, "players.npz"), x=px, y=py, speed=speed(px, py),
                        team=m.pl["team"], person=m.pl["person"])


def main():
    prof = np.load(PROFILE)
    target = dict(zip(prof["names"], prof["values"]))
    lengths = [prof["gap_lengths_" + c] for c in op.CAUSES]
    ms = [Match(k, mt) for k, mt in enumerate(MATCHES)]

    # Gaps are fitted at each carry speed (the ball's path moves the causes and the out frames);
    # the speed whose fitted bench has our still share is kept.
    fits = {}
    for v in CARRY_SPEEDS:
        for m in ms:
            m.set_carry(v)
        fits[v] = fit_ball(ms, target, lengths)
        st = still_share(ms, fits[v][3])
        fits[v] += (st,)
        print("carry {:.1f} m/s: still ball {:.1%} of seen (ours {:.1%})".format(
            v, st, target["still ball share of seen"]), flush=True)
    carry = min(fits, key=lambda v: abs(fits[v][-1] - target["still ball share of seen"]))
    for m in ms:
        m.set_carry(carry)
    hz, kr, hide, masks, (share, frames, rmiss, out), still = fits[carry]
    print("ball fit at carry {:.1f} m/s: hazard per frame out {:.4f} line {:.4f} crowd {:.4f} none {:.4f}, "
          "K_RESTART {:.2f}, OUT_HIDE {:.2f}".format(carry, *hz, kr, hide))
    print("  seen {:.1%} (ours {:.1%}), gap frames out/line/crowd/none {} (ours {}), missing near restarts "
          "{:.1%} (ours {:.1%}), seen past a line {:.1%} (ours {:.1%}), still {:.1%}".format(
              share, target["ball seen share"], np.round(frames, 3),
              np.round([target["gap frames: " + c] for c in op.CAUSES], 3), rmiss,
              target["missing around restarts"], out, target["seen past a line > 1 m"], still), flush=True)

    base, fy, fdx, off_mean, (rate, ry, rx, tracked) = fit_players(ms, target)
    print("player fit: base {:.2e}/frame, y factors {}, dx factors {}, off mean {:.1f} s".format(
        base, np.round(fy, 2), np.round(fdx, 2), off_mean / FPS))
    print("  drops {:.3f}/player-min (ours {:.3f}), y rel {}, dx rel {}, tracked {:.2f} (ours {:.2f})".format(
        rate, target["player drops per player-min"], np.round(ry, 2), np.round(rx, 2), tracked,
        target["players tracked"]), flush=True)

    with open(FIT, "w") as f:
        json.dump(dict(carry_speed=carry, hazards=dict(zip(op.CAUSES, hz.tolist())), k_restart=kr, out_hide=hide,
                       player_base=base, y_factors=fy.tolist(), dx_factors=fdx.tolist(),
                       off_mean_s=off_mean / FPS), f, indent=1)
    for m, seen in zip(ms, masks):
        _, tracked, _ = m.players_tracked(base, fy, fdx, off_mean)
        write(m, seen, tracked)
        print("wrote", m.mt, flush=True)


if __name__ == "__main__":
    main()
