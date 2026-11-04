"""Stage 2 rebuilt on majority smoothing: the dead ball layer on top of the organisers' baseline.

    .venv\\Scripts\\python.exe v2/stage2_majority.py

Stage 1 failed its gate on bench v2 (spellF1 0.195 against majority smoothing's 0.198), so this puts
Stage 2's stoppages (v2/stage2.py stoppages(), unchanged) on majority smoothing instead of the Stage 1
skeleton: majority smoothing (window from bench.BASE_WINDOWS), then every frame inside a stoppage is D.
There is no Viterbi here, so Stage 2's W_D (how hard a stoppage pulls towards D) has no meaning: a
stoppage is simply D. The window and the stoppage settings are tuned together, LOMO on degraded Sportec;
both modes (full, fallback) as in stage2.py.

Gate (bench v2 rule): D F1 up and spellF1 not down against majority smoothing alone. Also compared with
Stage 2 on the Stage 1 skeleton (cache/v2_stage2_grid.csv).
Writes cache/v2_stage2m_grid.csv, evaluation/results/v2_stage2_majority.csv.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import deadball as db  # noqa: E402
import stage2 as s2  # noqa: E402
from bench import D, MATCHES, CACHE, BASE_WINDOWS  # noqa: E402

# Params
COLS = ("spellF1", "macroF1", "D_prec", "D_rec", "D_F1", "n_pred", "n_gt", "n_stops")


def with_stops(maj, stops):
    out = maj.copy()
    for s, e in stops:
        out[s:e] = D
    return out


def run_grid(mode):
    rows = []
    for k, mt in enumerate(MATCHES):
        t0 = time.time()
        m = bench.load(mt + "~")
        c = db.cues(m)
        seed = s2.seeds(m, k)
        codes = bench.baseline_codes(m)
        majs = {w: bench.majority(codes, w) for w in BASE_WINDOWS}
        for g in s2.grid():
            stops = s2.stoppages(c, seed, *g, mode)
            for w in BASE_WINDOWS:
                pred = bench.finish(m, with_stops(majs[w], stops))
                rows.append(dict(match=mt, param=(w,) + g, mode=mode, n_stops=len(stops),
                                 **bench.score(m["labels"], pred)))
        print("  {} {} done in {:.0f} s".format(mode, mt, time.time() - t0), flush=True)
    return pd.DataFrame(rows)


def predict(m, param, mode, k=0):
    w, *g = param
    stops = s2.stoppages(db.cues(m), s2.seeds(m, k), *g, mode)
    maj = bench.majority(bench.baseline_codes(m), w)
    return bench.finish(m, with_stops(maj, stops)), stops


def show(title, df):
    print("\n== {} ==".format(title))
    print(bench.summary(df, cols=COLS).to_string(index=False))


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)
    t0 = time.time()
    path = os.path.join(CACHE, "v2_stage2m_grid.csv")
    if os.path.isfile(path):
        g = pd.read_csv(path)
        g["param"] = g["param"].map(eval)
    else:
        g = pd.concat([run_grid("full"), run_grid("fallback")], ignore_index=True)
        g.to_csv(path, index=False)

    base = bench.lomo(pd.read_csv(os.path.join(CACHE, "v2_baseline_degraded.csv")))
    show("majority smoothing alone, LOMO window", base)
    old = pd.read_csv(os.path.join(CACHE, "v2_stage2_grid.csv"))
    old["param"] = old["param"].map(eval)
    old = bench.lomo(old[old["mode"] == "full"])
    show("stage 2 on the stage 1 skeleton (full), LOMO", old)

    res = {}
    for mode in ("full", "fallback"):
        lo = bench.lomo(g[g["mode"] == mode])
        res[mode] = lo
        show("stage 2 on majority smoothing, {}, LOMO".format(mode), lo)
        print("params (window, v_slow, t_slow, t_alone, lag):", lo.param.tolist())

    full = res["full"]
    passed = bench.gate(full, base)
    print("\nGATE stage 2 on majority vs majority alone (D F1 up, spellF1 not down): spellF1 {:.4f} vs {:.4f}, "
          "D F1 {:.3f} vs {:.3f} -> {}".format(full.spellF1.mean(), base.spellF1.mean(), full.D_F1.mean(),
                                               base.D_F1.mean(), "PASS" if passed else "FAIL"))
    for name, other in (("stage 2 on skeleton", old), ("fallback on majority", res["fallback"])):
        print("full on majority vs {}: spellF1 {:+.4f}, D F1 {:+.3f}, macroF1 {:+.4f}; gate rule {}".format(
            name, full.spellF1.mean() - other.spellF1.mean(), full.D_F1.mean() - other.D_F1.mean(),
            full.macroF1.mean() - other.macroF1.mean(), "PASS" if bench.gate(full, other) else "FAIL"))
    mode = "full" if passed else "fallback"
    final = bench.all_params_best(g[g["mode"] == mode])
    print("kept mode {}; final setting (best on all 7 degraded):".format(mode), final)

    clean = []
    for k, mt in enumerate(MATCHES):
        m = bench.load(mt)
        pred, stops = predict(m, final, mode, k)
        clean.append(dict(match=mt, n_stops=len(stops), **bench.score(m["labels"], pred)))
    show("same setting on clean Sportec (reference)", pd.DataFrame(clean))

    ours = bench.load("ours")
    pred, stops = predict(ours, final, mode)
    sd = s2.seeds(ours)
    covered = np.mean([any(s - 3 * s2.FPS <= x <= e + 3 * s2.FPS for s, e in stops) for x in sd])
    lens = np.array([e - s for s, e in stops]) / s2.FPS
    print("ours: stoppages {} (median {:.0f} s); Smart Tagging set pieces inside one: {:.0%} of {}".format(
        len(stops), np.median(lens) if len(lens) else 0, covered, len(sd)))
    bench.report_ours(pred, "v2_stage2_majority")
    print("took {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
