"""Part 2, step 5: blend the bench-trained ball with the straight line near the gap edges. No training.

    .venv\\Scripts\\python.exe v2/gap_blend.py

Needs cache/v2_gap/pred_<match>_fold<k>.npz and pred_ours_mean.npz (v2/gap_train.py, v2/gap_report.py).
As v1/ball_blend.py: the weight on the line is 1 at the nearest seen ball and fades to 0 at T seconds
from it; an edge with no seen ball behind it (a half's start or end) does not pull.
T is chosen leave one match out, by a rule fixed before the first run: the lowest mean error over
every gap frame of the other 6 matches (each filled by the model that never saw it). Ours takes the
T best on all 7.

Writes cache/v2_gap/blend_<match>.npz (x, y, predicted, t_seconds) for the 7 matches and ours.
Prints totals only.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gap_report as gr  # noqa: E402
from bench import MATCHES  # noqa: E402
from gap_dataset import DEGRADED, OUT_DIR  # noqa: E402

# Params
FPS = 25
T_CHOICES = (0.5, 1, 2, 3, 5, 8)   # seconds


def blend_path(path, path_line, gaps, half, t):
    """Line weight 1 at a seen ball, 0 from t seconds away; per gap, edges with a seen ball only."""
    x, y = path["x"].astype(np.float64).copy(), path["y"].astype(np.float64).copy()
    lx, ly = path_line
    for s, n, _ in gaps:
        e = s + n
        before = s > 0 and half[s - 1] == half[s]
        after = e < len(half) and half[e] == half[s]
        if not (before or after):
            continue
        f = np.arange(s, e)
        dist = np.minimum((f - s + 1) / FPS if before else np.inf, (e - f) / FPS if after else np.inf)
        w = np.clip(1 - dist / t, 0, 1)
        x[s:e] = w * lx[s:e] + (1 - w) * x[s:e]
        y[s:e] = w * ly[s:e] + (1 - w) * y[s:e]
    return x, y


def match_case(k, mt):
    m, gaps, glen, cause, gap = gr.gap_frames(mt + "~")
    p = np.load(gr.pred_path(mt, k))
    t = np.load(os.path.join(DEGRADED, mt, "ball_truth.npz"))
    line = gr.line_fill(m["raw_ball_x"], m["raw_ball_y"], gaps, m["half"])
    return dict(m=m, gaps=gaps, glen=glen, gap=gap & np.isfinite(p["x"]), p=p, truth=t, line=line)


def errors(c, t):
    if t is None:
        x, y = c["p"]["x"], c["p"]["y"]
    elif t == "line":
        x, y = c["line"]
    else:
        x, y = blend_path(c["p"], c["line"], c["gaps"], c["m"]["half"], t)
    return np.hypot(x - c["truth"]["x"], y - c["truth"]["y"])[c["gap"]]


def save(name, path, gaps, half, line, t, predicted):
    x, y = blend_path(path, line, gaps, half, t)
    np.savez_compressed(os.path.join(OUT_DIR, "blend_{}.npz".format(name)), x=x.astype(np.float32),
                        y=y.astype(np.float32), predicted=predicted, t_seconds=t)


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 2)
    cases = [match_case(k, mt) for k, mt in enumerate(MATCHES)]
    rows = []
    for mt, c in zip(MATCHES, cases):
        for t in (None, "line") + T_CHOICES:
            e = errors(c, t)
            short = c["glen"][c["gap"]] <= 5
            rows.append(dict(match=mt, T=str(t if t is not None else "model"), mean=e.mean(), median=np.median(e),
                             short_median=np.median(e[short]), sum=e.sum(), n=len(e)))
    t = pd.DataFrame(rows)
    blends = t[~t["T"].isin(["model", "line"])]
    chosen = {}
    for mt in MATCHES:
        other = blends[blends.match != mt].groupby("T")[["sum", "n"]].sum()
        chosen[mt] = float((other["sum"] / other["n"]).idxmin())
    print("T chosen leaving each match out:", chosen)
    pick = lambda r: r["T"] in ("model", "line") or float(r["T"]) == chosen[r["match"]]
    lo = t[t.apply(pick, axis=1)].copy()
    lo.loc[~lo["T"].isin(["model", "line"]), "T"] = "blend (LOMO T)"
    print("\nevery gap frame of the left-out match; 'short' = gaps up to 5 s")
    print(lo.groupby("T", sort=False)[["mean", "median", "short_median"]].mean().to_string())
    print("\nper T, mean over the 7 matches:")
    print(blends.groupby("T")[["mean", "median", "short_median"]].mean().to_string())

    allsum = blends.groupby("T")[["sum", "n"]].sum()
    t_ours = float((allsum["sum"] / allsum["n"]).idxmin())
    for mt, c in zip(MATCHES, cases):
        save(mt, c["p"], c["gaps"], c["m"]["half"], c["line"], chosen[mt], c["p"]["predicted"])
    m, gaps, _, _, _ = gr.gap_frames("ours")
    p = dict(np.load(os.path.join(OUT_DIR, "pred_ours_mean.npz")))
    line = gr.line_fill(m["raw_ball_x"], m["raw_ball_y"], gaps, m["half"])
    save("ours", p, gaps, m["half"], line, t_ours, p["predicted"])
    print("\nours: T = {:g} s (best on all 7). wrote cache/v2_gap/blend_<match>.npz".format(t_ours))


if __name__ == "__main__":
    main()
