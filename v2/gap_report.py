"""Part 2, step 4: how well the bench-trained ball model fills gaps it never saw.

    .venv\\Scripts\\python.exe v2/gap_report.py

Needs cache/v2_gap/pred_<match>_fold<k>.npz (v2/gap_train.py --all); folds not done yet are skipped.
Per left-out match, on its real gaps (placed like ours, by cause), against the straight line between
the last and next seen ball (baseline A):
  - error in metres per gap length and per cause (out, line, crowd, none: the seen ball just before);
  - out of the pitch: of the gap frames where the true ball is out, the share the model puts out,
    and how often its out calls are right;
  - who has the ball: mean p_nobody on dead (D) and live gap frames, and on live ones how often the
    larger of p_home / p_away is the team in possession (the weak team clue for Stage 4).
Ours: the 7 models averaged (none has seen our match) -> cache/v2_gap/pred_ours_mean.npz, with
believability totals and how far apart the 7 models are.

Writes cache/v2_gap/report.csv. Prints totals only.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import our_profile as op  # noqa: E402
from bench import MATCHES  # noqa: E402
from gap_dataset import DEGRADED, OUT_DIR  # noqa: E402

# Params
FPS = 25
LENGTH_BINS = (0, 2, 5, 10, 20, 30, np.inf)   # seconds
H, A, D = 0, 1, 2
KEYS = ("x", "y", "p_home", "p_away", "p_nobody")


def pred_path(name, fold):
    return os.path.join(OUT_DIR, "pred_{}_fold{}.npz".format(name, fold))


def line_fill(bx, by, gaps, half):
    """Baseline A on every gap: straight from the last seen ball to the next, held at a half's edge."""
    lx, ly = bx.astype(np.float64).copy(), by.astype(np.float64).copy()
    for s, n, _ in gaps:
        e = s + n
        a = s - 1 if s > 0 and half[s - 1] == half[s] else None
        b = e if e < len(half) and half[e] == half[s] else None
        if a is None and b is None:
            continue
        p0 = np.array([bx[a], by[a]]) if a is not None else np.array([bx[b], by[b]])
        p1 = np.array([bx[b], by[b]]) if b is not None else p0
        t = (np.arange(1, n + 1) / (n + 1))[:, None]
        lx[s:e], ly[s:e] = (p0 + t * (p1 - p0)).T
    return lx, ly


def gap_frames(name):
    """(match dict, gaps [(start, length, cause)], per frame: gap length in s, cause, inside a gap)."""
    m = bench.load(name)
    seen = np.isfinite(m["raw_ball_x"])
    cls = op.ball_classes(m["raw_ball_x"].astype(np.float64), m["raw_ball_y"].astype(np.float64),
                          m["raw_player_x"], m["raw_player_y"], m["pitch"])
    gaps = op.gap_causes(seen, m["half"], cls)
    glen = np.zeros(len(seen))
    cause = np.full(len(seen), -2)
    for s, n, c in gaps:
        glen[s:s + n], cause[s:s + n] = n / FPS, c
    return m, gaps, glen, cause, ~seen & (m["half"] > 0)


def score_match(k, mt):
    m, gaps, glen, cause, gap = gap_frames(mt + "~")
    t = np.load(os.path.join(DEGRADED, mt, "ball_truth.npz"))
    p = np.load(pred_path(mt, k))
    gap = gap & np.isfinite(p["x"])   # fills made before the half-edge fix leave a half's last 1-2 frames empty
    lx, ly = line_fill(m["raw_ball_x"], m["raw_ball_y"], gaps, m["half"])
    em = np.hypot(p["x"] - t["x"], p["y"] - t["y"])
    el = np.hypot(lx - t["x"], ly - t["y"])
    L, W = m["pitch"]
    inside = lambda x, y: np.minimum.reduce([x, L - x, y, W - y])
    true_out = gap & (inside(t["x"], t["y"]) < -op.OUT_MARGIN)
    model_out = gap & (inside(p["x"], p["y"]) < -op.OUT_MARGIN)
    lab = m["labels"]
    live = gap & ((lab == H) | (lab == A))
    rows = []
    for lo, hi in zip(LENGTH_BINS[:-1], LENGTH_BINS[1:]):
        sel = gap & (glen > lo) & (glen <= hi)
        rows.append(dict(match=mt, group="length {}-{} s".format(lo, hi), minutes=sel.sum() / FPS / 60,
                         model_median=np.median(em[sel]) if sel.any() else np.nan,
                         line_median=np.median(el[sel]) if sel.any() else np.nan,
                         model_mean=em[sel].mean() if sel.any() else np.nan, line_mean=el[sel].mean() if sel.any() else np.nan))
    for c, nm in enumerate(op.CAUSES):
        sel = gap & (cause == c)
        rows.append(dict(match=mt, group="cause " + nm, minutes=sel.sum() / FPS / 60,
                         model_median=np.median(em[sel]), line_median=np.median(el[sel]),
                         model_mean=em[sel].mean(), line_mean=el[sel].mean()))
    rows.append(dict(match=mt, group="all gaps", minutes=gap.sum() / FPS / 60,
                     model_median=np.median(em[gap]), line_median=np.median(el[gap]),
                     model_mean=em[gap].mean(), line_mean=el[gap].mean(),
                     out_caught=(model_out & true_out).sum() / max(1, true_out.sum()),
                     out_calls_right=(model_out & true_out).sum() / max(1, model_out.sum()),
                     p_nobody_dead=p["p_nobody"][gap & (lab == D)].mean(), p_nobody_live=p["p_nobody"][live].mean(),
                     team_clue_right=((p["p_away"][live] > p["p_home"][live]) == (lab[live] == A)).mean()))
    return rows


def ours_mean(folds):
    """Average of the folds' fills on our match, and how far apart they are."""
    preds = [np.load(pred_path("ours", k)) for k in folds]
    out = {k: np.mean([p[k] for p in preds], axis=0).astype(np.float32) for k in KEYS}
    out["predicted"] = preds[0]["predicted"]
    spread = np.median([np.hypot(p["x"] - out["x"], p["y"] - out["y"]) for p in preds], axis=0)
    out["spread"] = spread.astype(np.float32)
    np.savez_compressed(os.path.join(OUT_DIR, "pred_ours_mean.npz"), **out)
    m, gaps, glen, cause, gap = gap_frames("ours")
    gap = gap & np.isfinite(out["x"])
    L, W = m["pitch"]
    x, y = out["x"], out["y"]
    near = np.nanmin(np.hypot(m["player_x"] - x[:, None], m["player_y"] - y[:, None]), axis=1)
    print("\nours, {} models averaged, on {:.1f} min of gap:".format(len(folds), gap.sum() / FPS / 60))
    print("  on the pitch {:.1%}; nearest player median {:.1f} m; models apart (median per frame) {:.1f} m, "
          "gaps over 30 s {:.1f} m".format(((x[gap] >= 0) & (x[gap] <= L) & (y[gap] >= 0) & (y[gap] <= W)).mean(),
                                          np.nanmedian(near[gap]), np.median(spread[gap]), np.median(spread[gap & (glen > 30)])))
    print("  p_nobody mean {:.2f}; p_home share of the team shares {:.2f}".format(
        out["p_nobody"][gap].mean(), (out["p_home"] / (out["p_home"] + out["p_away"]))[gap].mean()))


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 2)
    folds = [k for k in range(len(MATCHES)) if os.path.exists(pred_path(MATCHES[k], k))]
    if not folds:
        print("no fold done yet")
        return
    rows = []
    for k in folds:
        rows += score_match(k, MATCHES[k])
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT_DIR, "report.csv"), index=False)
    print("folds done: {} of {}".format(len(folds), len(MATCHES)))
    mean = t.groupby("group", sort=False).mean(numeric_only=True)
    print(mean[["minutes", "model_median", "line_median", "model_mean", "line_mean"]].to_string())
    a = t[t.group == "all gaps"]
    print("\nper left-out match, all gaps:")
    print(a.drop(columns="group").to_string(index=False))
    ours_mean(folds)


if __name__ == "__main__":
    main()
