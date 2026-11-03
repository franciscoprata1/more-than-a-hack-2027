"""Phase 3, step 3, after stage 4: blend the plain model with the straight line. No training.

    .venv\\Scripts\\python.exe v1/ball_blend.py

Needs cache/ball_results_fold<k>_all.npz, cache/ball_predicted.npz, cache/ball_dataset.npz.

The weight on the line is 1 at the nearest known ball and fades in a straight line to 0 at T seconds
from it. T is chosen on the validation blocks only, by a rule fixed before the first run:
  1. the gap edge move must be no bigger than the real ball's; if no T manages that, the lowest wins;
  2. of those, the best on the pitch / out score for steps within NEAR_LINE_M of a line;
  3. within TIE_POINTS of the best, the lowest 10 s median error.
The hidden gaps are read after T is fixed and are a second look only: they were already opened, and
they were picked with the ball on the pitch, so they cannot show an out ball being caught.

Writes cache/ball_predicted_blend.npz, same keys as cache/ball_predicted.npz plus t_seconds.
cache/ball_predicted.npz is not changed. Prints totals only.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ball_dataset as bd  # noqa: E402
from ball_baselines import BOOTSTRAPS  # noqa: E402
from ball_fill import REFERENCE_GAPS_S, TURN_FRAMES, hidden_reference, report, row  # noqa: E402
from ball_train import ALL_TAG, centre_hidden, results_path  # noqa: E402
from dead_balls import read_tags  # noqa: E402
from hidden_gaps import FOLDS, GAP_SECONDS, SEED, second_half_wait, stoppage_mask  # noqa: E402
from smooth import FPS, PITCH  # noqa: E402

# Params
CACHE_DIR = os.path.join(ROOT, "cache")
PLAIN_PATH = os.path.join(CACHE_DIR, "ball_predicted.npz")
OUT_PATH = os.path.join(CACHE_DIR, "ball_predicted_blend.npz")
T_CHOICES = (0.5, 1, 2, 3, 5)
NEAR_LINE_M = 5.0        # the steps step 4 cares about
TIE_POINTS = 0.5         # closer than this is noise on a few hundred windows
EDGE_STEPS = bd.HZ * TURN_FRAMES // FPS  # 0.4 s
BAR_GAP_S = 10
L, W = PITCH


def line_weight(distance, t):
    """1 at a known ball, 0 from t seconds away."""
    return np.clip(1 - distance / t, 0, 1)[..., None]


def blend(case, t):
    w = line_weight(case["distance"], t)
    return w * case["line"] + (1 - w) * case["model"]


def inside(xy):
    return (xy[..., 0] >= 0) & (xy[..., 0] <= L) & (xy[..., 1] >= 0) & (xy[..., 1] <= W)


def line_distance(xy):
    x, y = xy[..., 0], xy[..., 1]
    return np.minimum.reduce([np.abs(x), np.abs(L - x), np.abs(y), np.abs(W - y)])


def validation_cases(results, known):
    """One case per gap length, validation windows of every fold together."""
    cases = {}
    for seconds in GAP_SECONDS:
        hide = centre_hidden(seconds)
        first, last = np.flatnonzero(hide)[[0, -1]]
        parts = {k: [] for k in ("model", "line", "truth", "mask", "distance")}
        anchors, probes, truth_ok, groups, base = [], [], [], [], 0
        for r in results:
            for w, s in enumerate(r["val_starts"]):
                shown = known[s:s + bd.WINDOW] & ~hide
                at = np.flatnonzero(shown)
                to_ball = np.abs(np.arange(bd.WINDOW)[:, None] - at[None]).min(axis=1) / bd.HZ if len(at) else np.full(bd.WINDOW, np.inf)
                parts["distance"].append(to_ball)
                before, after = at[at < first], at[at > last]
                ends = []
                if len(before) and before[-1] + EDGE_STEPS < (after[0] if len(after) else bd.WINDOW):
                    ends.append((before[-1], before[-1] + EDGE_STEPS))
                if len(after) and after[0] - EDGE_STEPS > (before[-1] if len(before) else -1):
                    ends.append((after[0], after[0] - EDGE_STEPS))
                for anchor, probe in ends:
                    anchors.append(r["val_truth_{}".format(seconds)][w, anchor])
                    probes.append((base + w, probe))
                    truth_ok.append(known[s + probe])
            for name in ("model", "line", "truth", "hidden"):
                parts["mask" if name == "hidden" else name].append(r["val_{}_{}".format(name, seconds)])
            groups.append(base + np.arange(len(r["val_starts"])))
            base += len(r["val_starts"])
        case = {k: np.concatenate(v) if k != "distance" else np.array(v) for k, v in parts.items()}
        case.update(anchors=np.array(anchors), probes=np.array(probes), truth_ok=np.array(truth_ok))
        cases[seconds] = case
    return cases


def hidden_cases(results, x_true, y_true):
    """One case per gap length: every frame of every hidden gap counts."""
    cases = {}
    for seconds in GAP_SECONDS:
        size = seconds * FPS
        gaps = np.concatenate([r["gaps_{}".format(seconds)] for r in results])
        take = lambda name: np.concatenate([r["path_{}_{}".format(name, seconds)] for r in results]).astype(np.float64)
        rows = np.arange(len(gaps))
        n = TURN_FRAMES
        anchors = np.r_[np.c_[x_true[gaps - 1], y_true[gaps - 1]], np.c_[x_true[gaps + size], y_true[gaps + size]]]
        probes = np.r_[np.c_[rows, np.full(len(gaps), n - 1)], np.c_[rows, np.full(len(gaps), size - n)]]
        to_ball = np.minimum(np.arange(1, size + 1), np.arange(size, 0, -1)) / FPS
        cases[seconds] = {"model": take("model"), "line": take("line"), "truth": take("truth"), "mask": np.ones((len(gaps), size), dtype=bool),
                          "distance": np.broadcast_to(to_ball, (len(gaps), size)), "anchors": anchors, "probes": probes,
                          "truth_ok": np.ones(len(probes), dtype=bool)}
    return cases


def errors(path, case):
    return np.hypot(*np.moveaxis(path - case["truth"], -1, 0))[case["mask"]]


def edge_move(path_of, cases):
    """Median first and last 0.4 s move over the 10 s and 20 s gaps."""
    moves = []
    for seconds in REFERENCE_GAPS_S:
        c = cases[seconds]
        path = path_of(c)
        keep = c["truth_ok"] if path is c["truth"] else np.ones(len(c["probes"]), dtype=bool)
        at = c["probes"][keep]
        moves.append(np.hypot(*(path[at[:, 0], at[:, 1]] - c["anchors"][keep]).T))
    return np.median(np.concatenate(moves))


def sides(path_of, cases):
    """On the pitch / out, every length together."""
    out = {"right": [], "near": [], "pred_in": [], "true_in": []}
    for case in cases.values():
        mask, truth = case["mask"], case["truth"]
        out["pred_in"].append(inside(path_of(case))[mask])
        out["true_in"].append(inside(truth)[mask])
        out["near"].append(line_distance(truth)[mask] <= NEAR_LINE_M)
    pred, true, near = [np.concatenate(out[k]) for k in ("pred_in", "true_in", "near")]
    right = pred == true
    return {"all": right.mean(), "near": right[near].mean(), "near_n": near.sum(), "n": len(right), "out_n": (~true).sum(),
            "caught": (~pred & ~true).sum() / max(1, (~true).sum()), "called": (~pred).sum(),
            "called_right": (~pred & ~true).sum() / max(1, (~pred).sum()),
            "always_all": true.mean(), "always_near": true[near].mean()}


def choose(cases):
    """The rule in the docstring. Prints every T."""
    real_edge = edge_move(lambda c: c["truth"], cases)
    print("validation blocks, every T. Real ball's edge move here: {:.1f} m".format(real_edge))
    print("   T     edge move  " + "  ".join("{:2d} s med".format(s) for s in GAP_SECONDS) + "   on/out all  on/out near  out caught")
    rows = []
    for t in T_CHOICES:
        path_of = lambda c, t=t: blend(c, t)
        s = sides(path_of, cases)
        medians = [np.median(errors(blend(cases[g], t), cases[g])) for g in GAP_SECONDS]
        rows.append((t, edge_move(path_of, cases), medians[GAP_SECONDS.index(BAR_GAP_S)], s["near"]))
        print("{:4.1f} s   {:5.1f} m   ".format(t, rows[-1][1]) + "  ".join("{:5.1f} m ".format(m) for m in medians)
              + "   {:7.1%}     {:7.1%}     {:6.1%}".format(s["all"], s["near"], s["caught"]))
    calm = [r for r in rows if r[1] <= real_edge] or [min(rows, key=lambda r: r[1])]
    best = max(r[3] for r in calm)
    return min([r for r in calm if r[3] >= best - TIE_POINTS / 100], key=lambda r: r[2])[0]


def compare(cases, t, title):
    """A, plain and the blend side by side."""
    methods = (("A straight line", lambda c: c["line"]), ("plain", lambda c: c["model"]),
               ("blend T = {:g} s".format(t), lambda c: blend(c, t)))
    print("\n" + title)
    print("{:18s} ".format("") + "  ".join("{:2d} s med / mean".format(s) for s in GAP_SECONDS) + "   edge move")
    for name, path_of in methods:
        cells = ["{:5.1f} / {:5.1f} m".format(np.median(e), e.mean())
                 for e in (errors(path_of(cases[s]), cases[s]) for s in GAP_SECONDS)]
        print("{:18s} ".format(name) + "  ".join(cells) + "   {:5.1f} m".format(edge_move(path_of, cases)))
    print("real ball's edge move: {:.1f} m".format(edge_move(lambda c: c["truth"], cases)))
    s = sides(methods[0][1], cases)
    print("\non the pitch / out: {} steps, {} within {:.0f} m of a line, {} truly out".format(s["n"], s["near_n"], NEAR_LINE_M, s["out_n"]))
    print("{:18s}  same side, all   within {:.0f} m   out caught   calls out   of those right".format("", NEAR_LINE_M))
    for name, path_of in methods:
        s = sides(path_of, cases)
        print("{:18s}     {:7.1%}      {:7.1%}     {:6.1%}     {:6d}       {:6.1%}".format(
            name, s["all"], s["near"], s["caught"], s["called"], s["called_right"]))
    print("{:18s}     {:7.1%}      {:7.1%}     {:6.1%}     {:6d}".format("always on pitch", s["always_all"], s["always_near"], 0, 0))
    if s["out_n"]:
        out_by_place(cases, methods[2][1], t)


def out_by_place(cases, path_of, t):
    """Where the out balls are caught: near a known ball, where the line helps, or deep in the gap."""
    print("\nblend, truly out steps: are they caught deep in the gap?")
    print("gap     up to T from a known ball     further than T")
    print("        steps    caught               steps    caught")
    for seconds, c in cases.items():
        out = c["mask"] & ~inside(c["truth"])
        caught = out & ~inside(path_of(c))
        close = c["distance"] <= t
        cells = [(out & at).sum() for at in (close, ~close)]
        rates = [(caught & at).sum() / max(1, (out & at).sum()) for at in (close, ~close)]
        print("{:2d} s   {:6d}    {:6.1%}              {:6d}    {:6.1%}".format(seconds, cells[0], rates[0], cells[1], rates[1]))


def near_lead(cases, t, rng):
    """Blend minus plain within NEAR_LINE_M, 95% range, resampling whole validation windows."""
    size = len(cases[GAP_SECONDS[0]]["mask"])
    gained, total = np.zeros(size), np.zeros(size)
    for c in cases.values():
        near = (line_distance(c["truth"]) <= NEAR_LINE_M) & c["mask"]
        true = inside(c["truth"])
        gained += (((inside(blend(c, t)) == true) & near).astype(int) - ((inside(c["model"]) == true) & near).astype(int)).sum(axis=1)
        total += near.sum(axis=1)
    picks = rng.integers(0, size, (BOOTSTRAPS, size))
    low, high = np.percentile(gained[picks].sum(axis=1) / total[picks].sum(axis=1), [2.5, 97.5])
    print("\nblend minus plain within {:.0f} m: {:+.1%} (95% range {:+.1%} to {:+.1%}, resampling windows)".format(
        NEAR_LINE_M, gained.sum() / total.sum(), low, high))


def real_gaps(plain, raw):
    """The Stage 4 gap list, from the saved plain paths."""
    x_raw, y_raw = raw["x"].astype(np.float64), raw["y"].astype(np.float64)
    out = []
    for a, b, fold in zip(plain["gap_start"], plain["gap_end"], plain["gap_fold"]):
        wanted = np.arange(a, b)
        out.append({"frames": wanted, "fold": int(fold), "path": np.c_[plain["x"][wanted], plain["y"][wanted]].astype(np.float64),
                    "before": np.array([x_raw[a - 1], y_raw[a - 1]]), "after": np.array([x_raw[b], y_raw[b]])})
    return out


def blend_real(gap, t):
    """A gap with the blend. An edge with no known ball behind it (half start or end) doesn't pull."""
    frames, a, b = gap["frames"], gap["frames"][0], gap["frames"][-1] + 1
    have_before, have_after = np.isfinite(gap["before"]).all(), np.isfinite(gap["after"]).all()
    if not have_before and not have_after:
        return dict(gap)
    distance = np.minimum((frames - a + 1) / FPS if have_before else np.inf, (b - frames) / FPS if have_after else np.inf)
    share = ((frames - a + 1) / (b - a + 1))[:, None]
    start = gap["before"] if have_before else gap["after"]
    end = gap["after"] if have_after else gap["before"]
    w = line_weight(distance, t)
    return dict(gap, path=w * (start + share * (end - start)) + (1 - w) * gap["path"])


def save(plain, filled, t):
    x, y = plain["x"].copy(), plain["y"].copy()
    for g in filled:
        x[g["frames"]], y[g["frames"]] = g["path"][:, 0], g["path"][:, 1]
    np.savez_compressed(OUT_PATH, x=x, y=y, predicted=plain["predicted"], gap_start=plain["gap_start"],
                        gap_end=plain["gap_end"], gap_fold=plain["gap_fold"], t_seconds=t)


def main():
    rng = np.random.default_rng(SEED)
    ds = bd.load()
    results = [dict(np.load(results_path(fold, ALL_TAG))) for fold in range(FOLDS)]
    raw = np.load(os.path.join(CACHE_DIR, "ball.npz"))
    x_true, y_true = raw["x"].astype(np.float64), raw["y"].astype(np.float64)

    cases = validation_cases(results, ds["known"])
    print("windows with no known ball at all: {}".format(sum(
        int(np.isinf(c["distance"]).all(axis=1).sum()) for c in cases.values())))
    t = choose(cases)
    print("\nchosen on validation blocks: T = {:g} s".format(t))
    compare(cases, t, "VALIDATION BLOCKS")
    near_lead(cases, t, rng)

    compare(hidden_cases(results, x_true, y_true), t,
            "HIDDEN GAPS, SECOND LOOK ONLY: T was fixed before reading these; all were picked with the ball on the pitch")

    sm = dict(np.load(os.path.join(CACHE_DIR, "smooth.npz")))
    plain = dict(np.load(PLAIN_PATH))
    gaps = real_gaps(plain, raw)
    filled = [blend_real(g, t) for g in gaps]
    scored = np.load(os.path.join(CACHE_DIR, "scored.npy"))
    first, kickoff = second_half_wait(read_tags(), scored)
    stopped = stoppage_mask(len(scored))
    stopped[first:kickoff] = True
    print("\nREAL GAPS, PLAIN (should repeat Stage 4)")
    report(gaps, sm, stopped, results, x_true, y_true)
    print("\nREAL GAPS, BLEND T = {:g} s".format(t))
    report(filled, sm, stopped, results, x_true, y_true)
    reference = hidden_reference(results, x_true, y_true, "model")
    row("hidden 10, 20 s: blend, 2nd look", [blend_real(g, t) for g in reference], sm)
    save(plain, filled, t)
    print("\nwrote cache/ball_predicted_blend.npz; cache/ball_predicted.npz not changed")


if __name__ == "__main__":
    main()
