"""Phase 3, step 3, stage 3: how good is the plain ball model on the hidden gaps?

    .venv\\Scripts\\python.exe v1/ball_report.py

Needs cache/ball_results_fold<k>_all.npz for every fold (v1/ball_train.py --all), and
cache/ball_baselines.csv, cache/hidden_gaps.npz, cache/ball.npz, cache/ball_dataset.npz.

Each of the 550 gaps is scored by the one model that never trained on its block. Medians are over
all frames of all gaps, as in v1/ball_baselines.py, with a 95% range that resamples whole gaps.

Writes cache/ball_stage3.csv. Prints totals only.
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ball_dataset as bd  # noqa: E402
from ball_baselines import BOOTSTRAPS, interval  # noqa: E402
from ball_train import ALL_TAG, results_path  # noqa: E402
from smooth import PITCH  # noqa: E402
from hidden_gaps import FOLDS, GAP_SECONDS, SEED  # noqa: E402

# Params
CACHE_DIR = os.path.join(ROOT, "cache")
OUT_PATH = os.path.join(CACHE_DIR, "ball_stage3.csv")
BASELINES_PATH = os.path.join(CACHE_DIR, "ball_baselines.csv")
LINE_BINS_M = (0, 5, 10, 20, np.inf)  # how far the ball is from the nearest line when it disappears
BAR_GAP_S = 10
BAR_M = 5.0


def read_baselines():
    """{(gap seconds, name): row}."""
    with open(BASELINES_PATH, newline="", encoding="utf-8") as handle:
        return {(int(r["gap_s"]), r["baseline"]): r for r in csv.DictReader(handle)}


def gather(results, name, seconds):
    return np.concatenate([r["{}_{}".format(name, seconds)] for r in results])


def lead_interval(e, a, rng):
    """95% range of A's median minus the model's, resampling whole gaps with the same picks for both."""
    picks = rng.integers(0, len(e), (BOOTSTRAPS, len(e)))
    return np.percentile([np.median(a[p]) - np.median(e[p]) for p in picks], [2.5, 97.5])


def main_table(results, baselines, rng):
    """Median and mean per gap length next to A, B, C. Stops if A disagrees with the baselines file."""
    print("\nhidden gaps, each scored by a model that never trained on its block")
    print("gap    n    model median (95% range)  mean      A median  mean     B median  mean     C median  mean")
    rows = []
    for seconds in GAP_SECONDS:
        e, a = gather(results, "model", seconds), gather(results, "line", seconds)
        base = {n: baselines[seconds, n] for n in "ABC"}
        assert len(e) == int(base["A"]["n"]) and abs(np.median(a) - float(base["A"]["median_m"])) < 0.01
        low, high = interval(e, rng)
        print("{:2d} s {:4d}   {:6.1f} m  ({:4.1f} - {:4.1f})  {:6.1f} m   {:6.1f} m {:5.1f} m  {:6.1f} m {:5.1f} m  {:6.1f} m {:5.1f} m".format(
            seconds, len(e), np.median(e), low, high, e.mean(), *[float(base[n][k]) for n in "ABC" for k in ("median_m", "mean_m")]))
        rows.append({"gap_s": seconds, "n": len(e), "model_median_m": round(float(np.median(e)), 2), "model_low": round(float(low), 2),
                     "model_high": round(float(high), 2), "model_mean_m": round(float(e.mean()), 2),
                     **{"{}_median_m".format(n): base[n]["median_m"] for n in "ABC"},
                     **{"{}_mean_m".format(n): base[n]["mean_m"] for n in "ABC"}})
    return rows


def against_line(results, rng):
    """Is the model clearly better than A, gap by gap?"""
    print("\nagainst A on the same gaps")
    print("gap    A median minus model (95% range)   gaps where the model's mean error is lower")
    for seconds in GAP_SECONDS:
        e, a = gather(results, "model", seconds), gather(results, "line", seconds)
        low, high = lead_interval(e, a, rng)
        print("{:2d} s    {:6.1f} m  ({:5.1f} - {:5.1f})            {:.0%}".format(
            seconds, np.median(a) - np.median(e), low, high, (e.mean(axis=1) < a.mean(axis=1)).mean()))


def per_fold(results):
    """The same bar length, model by model: is one fold carrying the rest?"""
    print("\n{} s gaps per fold".format(BAR_GAP_S))
    print("fold    n   model median   A median")
    for fold, r in enumerate(results):
        e, a = r["model_{}".format(BAR_GAP_S)], r["line_{}".format(BAR_GAP_S)]
        print("  {}  {:3d}    {:6.1f} m     {:6.1f} m".format(fold, len(e), np.median(e), np.median(a)))


def distance_table(results, raw):
    """Error against how close to a line the ball was when it disappeared."""
    length, width = PITCH
    print("\nerror against distance from the nearest line, at the last seen frame")
    print("gap       from line      n   model median   A median")
    for seconds in GAP_SECONDS + (None,):
        pick = GAP_SECONDS if seconds is None else (seconds,)
        name = "all" if seconds is None else "{:d} s".format(seconds)
        for low, high in zip(LINE_BINS_M[:-1], LINE_BINS_M[1:]):
            e, a, gaps = [], [], 0
            for s in pick:
                before = gather(results, "gaps", s) - 1
                x, y = raw["x"][before], raw["y"][before]
                distance = np.minimum.reduce([x, length - x, y, width - y])
                inside = (distance >= low) & (distance < high)
                gaps += inside.sum()
                e.append(gather(results, "model", s)[inside].ravel())
                a.append(gather(results, "line", s)[inside].ravel())
            if gaps:
                print("{:7s}  {:3.0f} - {:3.0f} m   {:4d}    {:6.1f} m     {:6.1f} m".format(
                    name, low, min(high, 99), gaps, np.median(np.concatenate(e)), np.median(np.concatenate(a))))
            else:
                print("{:7s}  {:3.0f} - {:3.0f} m      0".format(name, low, min(high, 99)))


def holder_table(results):
    """How often the top guess for who has the ball matches the code-made label, hidden steps."""
    print("\nwho has the ball, hidden steps with a code-made label")
    print("gap    labelled   top guess right")
    for seconds in GAP_SECONDS:
        right = sum(int(r["right_{}".format(seconds)]) for r in results)
        labelled = sum(int(r["labelled_{}".format(seconds)]) for r in results)
        print("{:2d} s   {:6d}     {:.1%}".format(seconds, labelled, right / labelled))


def validation_table(results):
    """The saved validation paths, all folds. Fold 0 alone should repeat v1/ball_checks.py."""
    print("\nvalidation blocks, {} windows, from the saved paths".format(sum(len(r["val_starts"]) for r in results)))
    print("gap    fold 0 median   all folds median   straight line")
    for seconds in GAP_SECONDS:
        pairs = [(np.hypot(*np.moveaxis(r["val_model_{}".format(seconds)] - r["val_truth_{}".format(seconds)], 2, 0)),
                  np.hypot(*np.moveaxis(r["val_line_{}".format(seconds)] - r["val_truth_{}".format(seconds)], 2, 0)),
                  r["val_hidden_{}".format(seconds)]) for r in results]
        e = np.concatenate([m[h] for m, _, h in pairs])
        a = np.concatenate([n[h] for _, n, h in pairs])
        print("{:2d} s     {:6.1f} m        {:6.1f} m         {:6.1f} m".format(
            seconds, np.median(pairs[0][0][pairs[0][2]]), np.median(e), np.median(a)))


def main():
    results = [dict(np.load(results_path(fold, ALL_TAG))) for fold in range(FOLDS)]
    hidden = np.load(os.path.join(CACHE_DIR, "hidden_gaps.npz"))
    raw = np.load(os.path.join(CACHE_DIR, "ball.npz"))
    rng = np.random.default_rng(SEED)
    bd.check_folds(bd.load(), hidden)
    rows = main_table(results, read_baselines(), rng)
    against_line(results, rng)
    per_fold(results)
    distance_table(results, raw)
    holder_table(results)
    validation_table(results)
    bar = next(r for r in rows if r["gap_s"] == BAR_GAP_S)
    print("\nbar: median under {:.0f} m on {} s gaps: {} ({:.1f} m)".format(
        BAR_M, BAR_GAP_S, "MET" if bar["model_median_m"] < BAR_M else "NOT MET", bar["model_median_m"]))
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote cache/ball_stage3.csv")


if __name__ == "__main__":
    main()
