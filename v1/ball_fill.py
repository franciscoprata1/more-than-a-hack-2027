"""Phase 3, step 3, stage 4: predict the ball in every real long gap and ask if the result is believable.

    .venv\\Scripts\\python.exe v1/ball_fill.py

Needs cache/ball_model_fold<k>_all.pt and cache/ball_results_fold<k>_all.npz (v1/ball_train.py --all),
cache/ball_dataset.npz, cache/smooth.npz, cache/ball.npz, cache/scored.npy, cache/dead_events.csv.

There is no true ball in a real gap, so this scores nothing. It reports how the path behaves, next to
the same measures on the true ball and on the model in the 10 s and 20 s hidden gaps.

Each gap is filled by the plain model of the fold holding its middle step, which never trained on
that block. A gap too long for one window gets overlapping windows, each counting most at its centre,
so there is no jump where they meet. Far inside such a gap a window has no ball at all, which the
model never met in training and which no test has covered: read those gaps by their own row.

Writes cache/ball_predicted.npz: x, y per frame (the smoothed ball, or the model where predicted is
True), predicted, and gap_start, gap_end, gap_fold per gap. Prints totals only.
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ball_dataset as bd  # noqa: E402
from ball_model import BallModel  # noqa: E402
from ball_train import ALL_TAG, BATCH, EDGE_S, make_batch, predict, results_path, set_seed  # noqa: E402
from dead_balls import read_tags  # noqa: E402
from hidden_gaps import FOLDS, HUMAN_SPEED, SEED, second_half_wait, stoppage_mask  # noqa: E402
from smooth import FPS, MISSING, PITCH, runs  # noqa: E402

# Params
CACHE_DIR = os.path.join(ROOT, "cache")
OUT_PATH = os.path.join(CACHE_DIR, "ball_predicted.npz")
OVERLAP_STRIDE = bd.WINDOW // 2   # long gaps: a new window every 15 s, so every step sits near a centre
TURN_FRAMES = 10                  # direction is read over 0.4 s: shorter than that is tracking noise
TURN_DEGREES = 45.0               # a clear change of direction, not a curve
TURN_MOVE_M = 1.0                 # both moves at least this far (2.5 m/s): a still ball has no direction
NO_PLAYER_M = 3.0                 # further than a stride and a stretch: nobody can have touched it
LENGTH_GROUPS_S = ((0, 20), (20, 30), (30, np.inf))  # up to the longest length tested, then beyond
REFERENCE_GAPS_S = (10, 20)       # real gaps are nearly all over 5 s


def load_models():
    models = []
    for fold in range(FOLDS):
        model = BallModel(on_line=False)
        model.load_state_dict(torch.load(os.path.join(CACHE_DIR, "ball_model_fold{}{}.pt".format(fold, ALL_TAG))))
        models.append(model.eval())
    return models


def window_starts_for(p, q, half):
    """Windows covering steps p..q-1. One if the gap fits inside the window's edges, else overlapping ones."""
    edge = EDGE_S * bd.HZ
    if q - p <= bd.WINDOW - 2 * edge:
        return [bd.window_around(p, q - 1, half)]
    in_half = np.flatnonzero(half == half[p])
    starts = np.arange(p - edge, q + edge - bd.WINDOW + OVERLAP_STRIDE, OVERLAP_STRIDE)
    return sorted(set(np.clip(starts, in_half[0], in_half[-1] + 1 - bd.WINDOW).tolist()))


def predict_steps(model, ds, p, q):
    """Ball at steps p..q-1, in metres, windows weighted towards their centres."""
    starts = window_starts_for(p, q, ds["half"])
    weight = 1 - np.abs(np.arange(bd.WINDOW) - (bd.WINDOW - 1) / 2) / bd.WINDOW
    total, weights = np.zeros((q - p, 2)), np.zeros(q - p)
    no_hiding = np.zeros(bd.WINDOW, dtype=bool)
    for i in range(0, len(starts), BATCH):
        chunk = starts[i:i + BATCH]
        batch = make_batch(ds, chunk, [no_hiding] * len(chunk))
        with torch.no_grad():
            pred = predict(model, batch)[1].numpy()
        for path, s in zip(pred, chunk):
            a, b = max(p, s), min(q, s + bd.WINDOW)
            if a < b:
                total[a - p:b - p] += weight[a - s:b - s, None] * path[a - s:b - s]
                weights[a - p:b - p] += weight[a - s:b - s]
    assert (weights > 0).all()
    return total / weights[:, None]


def fill(models, ds, gaps, raw):
    """Per gap: path at 25 fps, its frames, its fold, and the seen ball just before and after."""
    frames = ds["frames"]
    x_raw, y_raw = raw["x"].astype(np.float64), raw["y"].astype(np.float64)
    out = []
    for a, b in gaps:
        p, q = np.searchsorted(frames, [a, b])
        fold = int(ds["block"][(p + q - 1) // 2]) % FOLDS
        steps = predict_steps(models[fold], ds, p, q)
        before, after = np.array([x_raw[a - 1], y_raw[a - 1]]), np.array([x_raw[b], y_raw[b]])
        anchors = [(f, v) for f, v in ((a - 1, before), (b, after)) if np.isfinite(v).all()]
        at = np.r_[frames[p:q], [f for f, _ in anchors]]
        xy = np.vstack([steps] + [v[None] for _, v in anchors])
        order = np.argsort(at)
        wanted = np.arange(a, b)
        out.append({"frames": wanted, "fold": fold, "before": before, "after": after,
                    "path": np.c_[np.interp(wanted, at[order], xy[order, 0]), np.interp(wanted, at[order], xy[order, 1])]})
    return out


def nearest_player(sm, frames, path):
    """Metres from each point to the nearest player. inf where no player is tracked."""
    d = np.hypot(sm["player_x"][frames] - path[:, 0, None], sm["player_y"][frames] - path[:, 1, None])
    d[~(sm["player_speed"][frames] <= HUMAN_SPEED)] = np.inf
    return d.min(axis=1)


def turns(path, near):
    """Direction changes with no player near, over moves of TURN_FRAMES."""
    points = path[::TURN_FRAMES]
    close = near[::TURN_FRAMES]
    if len(points) < 3:
        return 0
    move = np.diff(points, axis=0)
    size = np.hypot(*move.T)
    a, b, sa, sb = move[:-1], move[1:], size[:-1], size[1:]
    far_enough = (sa >= TURN_MOVE_M) & (sb >= TURN_MOVE_M)
    cosine = (a * b).sum(axis=1) / np.maximum(sa * sb, 1e-9)
    turned = cosine < np.cos(np.radians(TURN_DEGREES))
    return int((far_enough & turned & (close[1:-1] > NO_PLAYER_M)).sum())


def measure(gaps, sm):
    """The believability numbers for a list of gaps."""
    path = np.vstack([g["path"] for g in gaps])
    near = np.concatenate([nearest_player(sm, g["frames"], g["path"]) for g in gaps])
    count = sum(turns(g["path"], nearest_player(sm, g["frames"], g["path"])) for g in gaps)
    enter = [np.hypot(*(g["path"][TURN_FRAMES - 1] - g["before"])) for g in gaps if len(g["path"]) >= TURN_FRAMES]
    leave = [np.hypot(*(g["after"] - g["path"][-TURN_FRAMES])) for g in gaps if len(g["path"]) >= TURN_FRAMES]
    length, width = PITCH
    inside = (path[:, 0] >= 0) & (path[:, 0] <= length) & (path[:, 1] >= 0) & (path[:, 1] <= width)
    minutes = len(path) / FPS / 60
    ends = np.array(enter + leave)
    return {"gaps": len(gaps), "minutes": minutes, "inside": inside.mean(), "near": np.median(near[np.isfinite(near)]),
            "turns": count / minutes, "ends": np.median(ends[np.isfinite(ends)])}


def row(label, gaps, sm):
    if not gaps:
        print("{:34s}    0".format(label))
        return
    m = measure(gaps, sm)
    print("{:34s} {:4d} {:6.1f}   {:6.1%}    {:5.1f} m     {:6.1f}      {:5.1f} m".format(
        label, m["gaps"], m["minutes"], m["inside"], m["near"], m["turns"], m["ends"]))


def hidden_reference(results, x_true, y_true, which):
    """The 10 s and 20 s hidden gaps as the same kind of list, with the model's or the true path."""
    out = []
    for r in results:
        for seconds in REFERENCE_GAPS_S:
            size = seconds * FPS
            for a, path in zip(r["gaps_{}".format(seconds)], r["path_{}_{}".format(which, seconds)]):
                out.append({"frames": np.arange(a, a + size), "path": path.astype(np.float64),
                            "before": np.array([x_true[a - 1], y_true[a - 1]]),
                            "after": np.array([x_true[a + size], y_true[a + size]])})
    return out


def report(filled, sm, stopped, results, x_true, y_true):
    print("\n{:34s} gaps    min   inside     nearest     turns with   first / last".format(""))
    print("{:34s}              pitch      player      no player    0.4 s move".format(""))
    print("{:34s}                         median      per minute   median".format(""))
    row("all real gaps", filled, sm)
    for low, high in LENGTH_GROUPS_S:
        size = [g for g in filled if low * FPS < len(g["frames"]) <= high * FPS]
        row("  {:.0f} to {} s".format(low, "longest" if np.isinf(high) else int(high)), size, sm)
    row("  mostly open play", [g for g in filled if stopped[g["frames"]].mean() <= 0.5], sm)
    row("  mostly in a stoppage", [g for g in filled if stopped[g["frames"]].mean() > 0.5], sm)
    row("hidden 10, 20 s: the model", hidden_reference(results, x_true, y_true, "model"), sm)
    row("hidden 10, 20 s: the true ball", hidden_reference(results, x_true, y_true, "truth"), sm)


def save(filled, sm, count):
    x, y = sm["ball_x"].astype(np.float32).copy(), sm["ball_y"].astype(np.float32).copy()
    predicted = np.zeros(count, dtype=bool)
    for g in filled:
        x[g["frames"]], y[g["frames"]] = g["path"][:, 0], g["path"][:, 1]
        predicted[g["frames"]] = True
    np.savez_compressed(OUT_PATH, x=x, y=y, predicted=predicted, gap_start=np.array([g["frames"][0] for g in filled]),
                        gap_end=np.array([g["frames"][-1] + 1 for g in filled]), gap_fold=np.array([g["fold"] for g in filled]))
    return predicted


def main():
    set_seed(SEED)
    ds = bd.load()
    sm = dict(np.load(os.path.join(CACHE_DIR, "smooth.npz")))
    raw = np.load(os.path.join(CACHE_DIR, "ball.npz"))
    scored = np.load(os.path.join(CACHE_DIR, "scored.npy"))
    results = [dict(np.load(results_path(fold, ALL_TAG))) for fold in range(FOLDS)]
    gaps = runs((sm["ball_state"] == MISSING) & scored)
    filled = fill(load_models(), ds, gaps, raw)

    first, kickoff = second_half_wait(read_tags(), scored)
    stopped = stoppage_mask(len(scored))
    stopped[first:kickoff] = True
    predicted = save(filled, sm, len(scored))
    assert predicted.sum() == ((sm["ball_state"] == MISSING) & scored).sum()
    assert all(np.isfinite(g["path"]).all() for g in filled)

    print("{} real gaps, {:.1f} minutes predicted; gaps per fold {}".format(
        len(filled), predicted.sum() / FPS / 60, np.bincount([g["fold"] for g in filled], minlength=FOLDS).tolist()))
    print("gaps needing overlapping windows: {}".format(sum(
        len(window_starts_for(*np.searchsorted(ds["frames"], [g["frames"][0], g["frames"][-1] + 1]), ds["half"])) > 1 for g in filled)))
    report(filled, sm, stopped, results, raw["x"].astype(np.float64), raw["y"].astype(np.float64))
    print("\nwrote cache/ball_predicted.npz")


if __name__ == "__main__":
    main()
