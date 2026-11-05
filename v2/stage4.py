"""Stage 4: the predicted ball inside the gaps (Part 2) as dead ball cues, on top of Stage 3 control.

    .venv\\Scripts\\python.exe v2/stage4.py

Needs cache/v2_gap/pred_<match>_fold<k>.npz, pred_ours_mean.npz (v2/gap_train.py, v2/gap_report.py)
and blend_<match>.npz (v2/gap_blend.py). On a bench match the cues come from the model that never
saw that match; on ours from the 7 models averaged (none saw it).

Only on frames where the ball is missing for more than 1 s (v1 smoothing state 2), per frame
towards D, each with an off value in the grid:
  - nobody has the ball: W_PN x (logit p_nobody - logit P_MID). Above P_MID it leans to D, below
    it leans away from D (live play the model can see), so it can also undo the old lean to D;
  - predicted ball out (blended path past a line by Stage 1's OUT_M): W_OUT;
  - predicted ball near a line (within NEAR_M, not out): W_LINE;
  - predicted ball still (blended path below STILL_SPEED, 1 s mean): W_STILL. Weak by design: the
    bench's dead ball walk is synthetic.
The team shares p_home / p_away are not used: on the bench they pick the right team 50% of the time.

Stage 3 control stays as it is. Its evidence settings are limited to the ones its own LOMO picked
(found by code from its grid), its switch costs stay free, so leaving a match out still chooses
everything on the other 6. Gate (bench.gate): D F1 up and spellF1 not down against Stage 3 control.
Writes cache/v2_stage4_grid.csv, evaluation/results/v2_stage4.csv.
"""

import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import deadball as db  # noqa: E402
import stage1 as s1  # noqa: E402
import stage3 as s3  # noqa: E402
import stage3_control as s3c  # noqa: E402
from bench import D, X, MATCHES, CACHE  # noqa: E402
from gap_dataset import OUT_DIR  # noqa: E402

# Params
FPS = 25
NEAR_M = 3.0             # m: a throw-in or goal kick is taken about here
STILL_SPEED = 1.0        # m/s, as Stage 2's still ball
W_PNS = (0.0, 0.1, 0.3, 1.0)       # nats per frame per unit of log-odds
P_MIDS = (0.2, 0.3, 0.4)           # bench: p_nobody 0.49 dead, 0.11 live
W_OUTS = (0.0, 0.5, 2.0)
W_LINES = (0.0, 0.2)
W_STILLS = (0.0, 0.3)
GRID_PATH = os.path.join(CACHE, "v2_stage4_grid.csv")
S3_GRID = os.path.join(CACHE, "v2_stage3c_grid.csv")


def read_grid(path):
    g = pd.read_csv(path)
    g["param"] = g["param"].map(eval)
    return g


def stage3_settings():
    """The Stage 3 control evidence settings its LOMO picked, and its LOMO table."""
    lo = bench.lomo(read_grid(S3_GRID))
    return sorted({p[:5] for p in lo.param}), lo


def new_grid():
    pn = [(0.0, P_MIDS[0])] + [(w, p) for w in W_PNS[1:] for p in P_MIDS]
    return [a + (b, c, d) for a, b, c, d in itertools.product(pn, W_OUTS, W_LINES, W_STILLS)]


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def files(name, k):
    if name == "ours":
        return os.path.join(OUT_DIR, "pred_ours_mean.npz"), os.path.join(OUT_DIR, "blend_ours.npz")
    return os.path.join(OUT_DIR, "pred_{}_fold{}.npz".format(name, k)), os.path.join(OUT_DIR, "blend_{}.npz".format(name))


class Cues:
    def __init__(self, m, k, name):
        self.c = s3c.Cues(m, k)
        self.n = n = self.c.n
        pp, bp = files(name, k)
        p, b = np.load(pp), np.load(bp)
        self.gap = (m["ball_state"] == 2) & (m["half"] > 0)
        lpn = logit(np.nan_to_num(p["p_nobody"][:n], nan=0.5))
        x, y = b["x"][:n].astype(np.float64), b["y"][:n].astype(np.float64)
        ok = self.gap & np.isfinite(x)
        L, W = m["pitch"]
        inside = np.minimum.reduce([x, L - x, y, W - y])
        speed = np.zeros(n)
        speed[1:] = np.nan_to_num(np.hypot(np.diff(x), np.diff(y)) * FPS, nan=99.0)
        self.lpn = np.where(ok, lpn, 0.0)
        with np.errstate(invalid="ignore"):
            self.out = ok & (inside < -s1.OUT_M)
            self.near = ok & (inside >= -s1.OUT_M) & (inside < NEAR_M)
            self.still = ok & (db.run_mean(speed, FPS) < STILL_SPEED)
        self.ok = ok
        self.base = {}

    def stage3(self, s3p):
        if s3p not in self.base:
            self.base[s3p] = self.c.build(*s3p)
        return self.base[s3p]

    def build(self, s3p, w_pn, p_mid, w_out, w_line, w_still):
        e, turns = self.stage3(s3p)
        extra = np.zeros((self.n, 3))
        lean = w_pn * (self.lpn - logit(p_mid)) + w_out * self.out + w_line * self.near + w_still * self.still
        extra[self.ok, D] = lean[self.ok]
        return e + s1.to_steps(extra), turns


def run_grid(settings):
    costs, disc = s3c.cost_arrays()
    rows = []
    for k, mt in enumerate(MATCHES):
        t0 = time.time()
        m = bench.load(mt + "~")
        c = Cues(m, k, mt)
        for s3p in settings:
            for p in new_grid():
                e, turns = c.build(s3p, *p)
                paths = s3.viterbi_touch(e, costs, disc, c.c.near(turns, len(e)))
                for cg, path in zip(s3c.cost_grid(), paths):
                    pred = bench.finish(m, s1.upsample(path, c.n))
                    rows.append(dict(match=mt, param=s3p + p + cg, **bench.score(m["labels"], pred)))
        print("  {} done in {:.0f} s".format(mt, time.time() - t0), flush=True)
    return pd.DataFrame(rows)


def predict(m, param, k, name):
    s3p, p, (ct, d) = param[:5], param[5:10], param[10:]
    c = Cues(m, k, name)
    _, _, _, _, c_in, c_out = s3c.s2.STAGE1
    e, turns = c.build(s3p, *p)
    path = s3.viterbi_touch(e, s1.cost_matrix(ct, c_in, c_out)[None], np.array([d]), c.c.near(turns, len(e)))[0]
    return bench.finish(m, s1.upsample(path, c.n)), c


def gap_d(m, pred):
    """On frames with the ball missing over 1 s: share called D, share truly D, and D precision there."""
    g = (m["ball_state"] == 2) & (m["half"] > 0) & (m["labels"] != X)
    called, true = pred[g] == D, m["labels"][g] == D
    return {"gap_called_D": called.mean(), "gap_true_D": true.mean(),
            "gap_D_prec": (called & true).sum() / max(1, called.sum()), "gap_D_rec": (called & true).sum() / max(1, true.sum())}


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 3)
    t0 = time.time()
    settings, lo3 = stage3_settings()
    print("Stage 3 control evidence settings its LOMO picked:", settings)
    if os.path.isfile(GRID_PATH):
        g = read_grid(GRID_PATH)
    else:
        g = run_grid(settings)
        g.to_csv(GRID_PATH, index=False)
    lo = bench.lomo(g)

    extra3, extra4 = [], []
    for k, mt in enumerate(MATCHES):
        m = bench.load(mt + "~")
        p3 = s3c.predict(m, lo3[lo3.match == mt].param.iloc[0], k)[0]
        p4 = predict(m, lo[lo.match == mt].param.iloc[0], k, mt)[0]
        extra3.append(gap_d(m, p3))
        extra4.append(gap_d(m, p4))
    gcols = ("gap_called_D", "gap_true_D", "gap_D_prec", "gap_D_rec")
    for df, ex in ((lo3, extra3), (lo, extra4)):
        for col in gcols:
            df[col] = [x[col] for x in ex]
    s3.show("stage 3 control, LOMO", lo3, s3.COLS + gcols)
    s3.show("stage 4, LOMO", lo, s3.COLS + gcols)
    print("stage 4 params (stage 3 evidence x5, w_pn, p_mid, w_out, w_line, w_still, c_team, discount):")
    for mt, p in zip(lo.match, lo.param):
        print("  ", mt, p)
    passed = bench.gate(lo, lo3)
    print("stage 4 vs stage 3 control: spellF1 {:.4f} vs {:.4f}, D F1 {:.3f} vs {:.3f}, macroF1 {:.4f} vs {:.4f} -> {}".format(
        lo.spellF1.mean(), lo3.spellF1.mean(), lo.D_F1.mean(), lo3.D_F1.mean(), lo.macroF1.mean(), lo3.macroF1.mean(),
        "PASS" if passed else "FAIL"))

    # each cue alone switched off (LOMO with the rest free)
    for label, keep in (("no nobody cue (w_pn 0)", lambda p: p[5] == 0),
                        ("no out cue", lambda p: p[7] == 0),
                        ("no near line cue", lambda p: p[8] == 0),
                        ("no still cue", lambda p: p[9] == 0),
                        ("all new cues off (= stage 3, its two settings)", lambda p: p[5] == p[7] == p[8] == p[9] == 0)):
        sub = bench.lomo(g[g.param.map(keep)])
        print("  {:48s} spellF1 {:.4f}  D F1 {:.3f}".format(label, sub.spellF1.mean(), sub.D_F1.mean()))

    if not passed:
        print("\ngate failed: Stage 3 control stays the kept version; no submission written")
        print("took {:.0f} s".format(time.time() - t0))
        return
    final = bench.all_params_best(g)
    print("\nkept: stage 4; final setting (best on all 7 degraded): {}".format(final))
    ours = bench.load("ours")
    pred, c = predict(ours, final, 0, "ours")
    print("ours: ball missing over 1 s on {:.1%} of scored frames; there, predicted out {:.1%}, near a line {:.1%}, "
          "still {:.1%}, p_nobody above P_MID {:.1%}; called D {:.1%}".format(
              c.gap[ours["half"] > 0].mean(), c.out[c.gap].mean(), c.near[c.gap].mean(), c.still[c.gap].mean(),
              (c.lpn[c.gap] > logit(final[6])).mean(), (pred[c.gap] == D).mean()))
    bench.report_ours(pred, "v2_stage4")
    print("took {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
