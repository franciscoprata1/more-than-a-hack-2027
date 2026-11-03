"""Phase 3, step 1: clean and smooth the ball and player tracks, then work out speed again.

    .venv\\Scripts\\python.exe v1/smooth.py

Rules (action plan, Phase 3, step 1):
  - ball gaps up to 1 s are filled by joining the positions before and after, and each filled
    frame keeps the length of its gap. Longer gaps stay empty: they are the "missing ball"
    sequences that step 2 looks at;
  - gaps that reach outside the scored frames (before kickoff, halftime) are not filled;
  - players are filled only inside their own track slot, and only gaps up to 2 s (longer ones
    would draw ghosts of players who moved to another slot; joining slots is Phase 5);
  - positions are smoothed with a small Gaussian looking both ways in time, inside each unbroken
    stretch only, and speed and direction come from the smoothed positions. The given speed is
    not used: it exists on only 10% of ball frames and does not agree with the positions;
  - every player also gets an acceleration, from the change in that smoothed velocity. It is noisy
    where a filled stretch joins a seen one: those frames have player_gap > 0.

Writes cache/smooth.npz. Prints only totals.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
FPS = 25
CACHE_DIR = os.path.join(ROOT, "cache")
PITCH = (105.3, 68.0)
BALL_MAX_GAP = 1 * FPS  # a guess across a second is fair; beyond that anything may have happened
PLAYER_MAX_GAP = 2 * FPS  # longer player gaps stay empty: the player has usually moved to another
                          # slot, so joining would draw a ghost (29.8 tracks per frame when filling all)
BALL_SIGMA = 1.0      # frames; small, so real kicks stay sharp
PLAYER_SIGMA = 2.0     # frames; players change speed more gently than the ball

# Ball state per frame.
SEEN, FILLED, MISSING = 0, 1, 2


def runs(mask):
    """[(start, end)] of each True run, end exclusive."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def fillable(start, end, n, scored):
    """A gap can be joined if it has a known point on both sides, all inside the scored frames."""
    return start > 0 and end < n and scored[start - 1:end + 1].all()


def join(v, start, end):
    """Straight line across v[start:end] from the known points either side, in place."""
    t = np.arange(1, end - start + 1) / (end - start + 1)
    v[start:end] = v[start - 1] + t * (v[end] - v[start - 1])


def fill_ball(x, y, scored):
    """Join ball gaps up to BALL_MAX_GAP. Returns x, y, state and gap length per frame."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    state = np.where(np.isnan(x), MISSING, SEEN).astype(np.int8)
    gap = np.zeros(len(x), dtype=np.int32)
    for start, end in runs(state == MISSING):
        gap[start:end] = end - start
        if end - start <= BALL_MAX_GAP and fillable(start, end, len(x), scored):
            join(x, start, end)
            join(y, start, end)
            state[start:end] = FILLED
    return x, y, state, gap


def fill_track(x, y, scored):
    """Join gaps up to PLAYER_MAX_GAP in one track slot. Returns x, y and gap length per frame (0 = seen)."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    gap = np.zeros(len(x), dtype=np.int32)
    for start, end in runs(np.isnan(x)):
        gap[start:end] = end - start
        if end - start <= PLAYER_MAX_GAP and fillable(start, end, len(x), scored):
            join(x, start, end)
            join(y, start, end)
    return x, y, gap


def gaussian(sigma):
    half = int(3 * sigma)
    w = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    return w / w.sum()


def centred(v, w):
    """Convolution of v with the odd-length kernel w, same length as v (also when v is shorter)."""
    half = len(w) // 2
    return np.convolve(v, w, mode="full")[half:half + len(v)]


def smooth_and_velocity(x, y, segments, sigma):
    """Smooth x, y inside each segment and return (x, y, vx, vy) in metres and m/s."""
    w = gaussian(sigma)
    sx, sy = np.full_like(x, np.nan), np.full_like(y, np.nan)
    vx, vy = np.full_like(x, np.nan), np.full_like(y, np.nan)
    for start, end in segments:
        if end - start < 2:
            sx[start:end], sy[start:end] = x[start:end], y[start:end]
            continue
        norm = centred(np.ones(end - start), w)  # fewer neighbours at the edges
        sx[start:end] = centred(x[start:end], w) / norm
        sy[start:end] = centred(y[start:end], w) / norm
        vx[start:end] = np.gradient(sx[start:end]) * FPS
        vy[start:end] = np.gradient(sy[start:end]) * FPS
    return sx, sy, vx, vy


def acceleration(vx, vy, segments):
    """(ax, ay) in m/s2: the change in velocity, inside each segment."""
    ax, ay = np.full_like(vx, np.nan), np.full_like(vy, np.nan)
    for start, end in segments:
        if end - start >= 2:
            ax[start:end] = np.gradient(vx[start:end]) * FPS
            ay[start:end] = np.gradient(vy[start:end]) * FPS
    return ax, ay


def main():
    ball = np.load(os.path.join(CACHE_DIR, "ball.npz"))
    players = np.load(os.path.join(CACHE_DIR, "players.npz"))
    scored = np.load(os.path.join(CACHE_DIR, "scored.npy"))

    bx, by, state, ball_gap = fill_ball(ball["x"], ball["y"], scored)
    bx, by, bvx, bvy = smooth_and_velocity(bx, by, runs(state != MISSING), BALL_SIGMA)

    px_raw, py_raw = players["x"], players["y"]
    n_frames, n_slots = px_raw.shape
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py, pvx, pvy, pax, pay = px.copy(), px.copy(), px.copy(), px.copy(), px.copy()
    player_gap = np.zeros((n_frames, n_slots), dtype=np.int32)
    for slot in range(n_slots):
        fx, fy, player_gap[:, slot] = fill_track(px_raw[:, slot], py_raw[:, slot], scored)
        segs = runs(~np.isnan(fx))
        sx, sy, vx, vy = smooth_and_velocity(fx, fy, segs, PLAYER_SIGMA)
        px[:, slot], py[:, slot], pvx[:, slot], pvy[:, slot] = sx, sy, vx, vy
        pax[:, slot], pay[:, slot] = acceleration(vx, vy, segs)

    np.savez_compressed(
        os.path.join(CACHE_DIR, "smooth.npz"),
        ball_x=bx.astype(np.float32), ball_y=by.astype(np.float32),
        ball_vx=bvx.astype(np.float32), ball_vy=bvy.astype(np.float32),
        ball_speed=np.hypot(bvx, bvy).astype(np.float32), ball_state=state, ball_gap=ball_gap,
        player_x=px, player_y=py, player_vx=pvx, player_vy=pvy,
        player_speed=np.hypot(pvx, pvy).astype(np.float32), player_gap=player_gap,
        player_ax=pax, player_ay=pay, player_accel=np.hypot(pax, pay).astype(np.float32),
        team=players["team"])

    # Totals only.
    s, g = state[scored], ball_gap[scored]
    print("ball on scored frames: seen {:.1%}, filled {:.1%}, still missing {:.1%}".format(
        *[(s == k).mean() for k in (SEEN, FILLED, MISSING)]))
    long_gaps = runs((state == MISSING) & scored)
    print("missing ball sequences for step 2: {}, median {:.1f} s".format(
        len(long_gaps), np.median([e - st for st, e in long_gaps]) / FPS))
    raw_seen = (~np.isnan(px_raw[scored])).sum(axis=1)
    now_seen = (~np.isnan(px[scored])).sum(axis=1)
    print("player tracks per scored frame: {:.1f} raw -> {:.1f} after filling; frames with more than 22: "
          "{:.1%} raw -> {:.1%}".format(raw_seen.mean(), now_seen.mean(), (raw_seen > 22).mean(), (now_seen > 22).mean()))

    tracked = player_gap[scored] == 0
    speed = np.hypot(pvx, pvy)[scored][tracked]
    accel = np.hypot(pax, pay)[scored][tracked]
    for label, v in (("speed m/s", speed), ("acceleration m/s2", accel)):
        print("player {} on seen frames: median {:.2f}, 99% {:.2f}, 99.9% {:.2f}, max {:.1f}".format(
            label, *np.nanpercentile(v, [50, 99, 99.9, 100])))
    raw_speed = np.hypot(np.diff(ball["x"]), np.diff(ball["y"])) * FPS
    seen_both = (state[1:] == SEEN) & (state[:-1] == SEEN) & scored[1:]
    print("ball speed on seen frames, median: {:.1f} m/s frame-to-frame raw -> {:.1f} m/s smoothed".format(
        np.median(raw_speed[seen_both]), np.median(np.hypot(bvx, bvy)[1:][seen_both])))
    print("wrote cache/smooth.npz ({:.0f} MB)".format(os.path.getsize(os.path.join(CACHE_DIR, "smooth.npz")) / 1e6))


if __name__ == "__main__":
    main()
