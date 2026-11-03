"""Stage 2: a dead ball layer on top of the Stage 1 skeleton.

    .venv\\Scripts\\python.exe v2/stage2.py

A stoppage is a stretch where the players slow down together (v2/deadball.py "crowd" below V_SLOW,
small breaks under 2 s closed), at least T_SLOW long, and backed by something only a stoppage has:
  - a set piece shape from the Laws (thrower past a touchline, taker at a corner flag, kickoff
    formation, ball still in a goal area with no attacker in the box, ball still with no opponent
    near: the free kick is "the rest");
  - the ball seen past a line (our exits);
  - a seed: on our match the Smart Tagging set piece tags (restart = tag start + 3.5 s); on Sportec
    the same thing simulated from its set piece events, keeping each kind at about the share our
    tags cover (SEED_KEEP, football sense: our 26 throw-ins tagged of ~45 expected, and so on).
A slow stretch longer than T_ALONE counts with no backing at all (the slowdown rule).
It starts LAG_S before the players slowed (the whistle comes first) and ends at the restart: the
first ball still-then-kicked within 3 s of the slow stretch's end, else its end.
Frames inside get W_D nats per frame towards D; everything else is Stage 1 unchanged (its costs and
evidence are fixed: one change at a time).

Fallback (plan): seeds plus the slowdown rule only, no shapes, exits or restart kicks.
Tuned on degraded Sportec, LOMO. Gate: better D precision, D recall and spellF1 than Stage 1.
Writes cache/v2_stage2_grid.csv, evaluation/results/v2_stage2.csv.
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
from bench import D, MATCHES, CACHE  # noqa: E402
from smooth import runs  # noqa: E402

# Params
FPS = 25
STAGE1 = (4.0, 0.2, 0.02, 8, 8, 8)   # the Stage 1 setting kept (best on all 7 degraded matches)
CLOSE_S = 2.0                         # slow stretches this close are one
SUPPORT_S = 3.0                       # backing may sit this far outside the slow stretch
RESTART_S = 3.0                       # a restart kick this close to the slow stretch's end ends it
TAG_LEAD = int(3.5 * FPS)
SEED_KEEP = {"ThrowIn": 0.6, "CornerKick": 1.0, "GoalKick": 0.65, "FreeKick": 0.2,
             "KickOff": 1.0, "Penalty": 1.0}
SEED_SEED = 2027
V_SLOW = (1.0, 1.3, 1.6)              # m/s
T_SLOW = (2.0, 4.0, 8.0)              # s
T_ALONE = (8.0, 15.0, 1e9)            # s; 1e9 = never alone
LAG_S = (0.0, 1.0, 2.0)
W_D = (0.2, 1.0, 3.0)                 # nats per frame


def seeds(m, k=0):
    """Seed restart frames: Smart Tagging on our match, simulated from events on Sportec."""
    if m["name"] == "ours":
        t = m["tags"]
        t = t[t["code"] == "SET PIECES"]
        return np.sort(t["start_frame"].astype(int).values + TAG_LEAD)
    rng = np.random.default_rng(SEED_SEED + k)
    ev = m["events"]
    ev = ev[(ev["set_piece"] == 1) & (ev["in_half"] == 1)]
    keep = rng.random(len(ev)) < ev["kind"].map(SEED_KEEP).fillna(0).values
    return np.sort(ev["frame"].values[keep].astype(int))


def near_any(frames, a, b):
    """Is any of the sorted frames inside [a, b)?"""
    i = np.searchsorted(frames, a)
    return i < len(frames) and frames[i] < b


def stoppages(c, seed, v_slow, t_slow, t_alone, lag_s, mode):
    """[(start, end)] of the stoppages found."""
    n = len(c["crowd"])
    slow = np.nan_to_num(c["crowd"], nan=0.0) < v_slow
    # close short breaks
    for s, e in runs(~slow):
        if e - s < CLOSE_S * FPS and s > 0 and e < n:
            slow[s:e] = True
    backing = np.zeros(n, dtype=bool)
    if mode == "full":
        for k in db.SHAPES + ("out",):
            backing |= c[k]
    back_frames = np.flatnonzero(backing)
    restarts = np.flatnonzero(c["restart"]) if mode == "full" else np.array([], dtype=int)
    out = []
    sup, ra = int(SUPPORT_S * FPS), int(RESTART_S * FPS)
    for s, e in runs(slow):
        if e - s < t_slow * FPS:
            continue
        ok = (e - s >= t_alone * FPS or near_any(back_frames, s - sup, e + sup)
              or near_any(seed, s - sup, e + sup))
        if not ok:
            continue
        end = e
        i = np.searchsorted(restarts, e - ra)
        if i < len(restarts) and restarts[i] < e + ra:
            end = restarts[i]
        j = np.searchsorted(seed, e - ra)  # a seed restart near the end is as good as a kick
        if j < len(seed) and seed[j] < e + ra and mode == "fallback":
            end = seed[j]
        out.append((max(0, s - int(lag_s * FPS)), end))
    return out


def evidence_with_dead(e1, stops, w_d, half):
    e = e1.copy()
    for s, t in stops:
        e[s:t, D] += w_d
    e[half == 0] = 0.0
    return e


def viterbi_many(es, cost):
    """Viterbi for several emission sets (G, T, 3) under one cost matrix -> (G, T)."""
    G, T, _ = es.shape
    neg = -es
    delta = neg[:, 0].copy()
    bp = np.empty((T, G, 3), dtype=np.int8)
    for t in range(1, T):
        cand = delta[:, :, None] + cost[None]
        bp[t] = cand.argmin(axis=1)
        delta = cand.min(axis=1) + neg[:, t]
    path = np.empty((G, T), dtype=np.int8)
    s = delta.argmin(axis=1)
    g = np.arange(G)
    for t in range(T - 1, -1, -1):
        path[:, t] = s
        s = bp[t, g, s]
    return path


def stage1_evidence(m):
    scale, q_in, md, *_ = STAGE1
    return s1.evidence(m, s1.distances(m), scale, q_in, md)


def grid():
    return list(itertools.product(V_SLOW, T_SLOW, T_ALONE, LAG_S))


def run_grid(mode):
    cost = s1.cost_matrix(*STAGE1[3:])
    rows = []
    for k, mt in enumerate(MATCHES):
        t0 = time.time()
        m = bench.load(mt + "~")
        c = db.cues(m)
        seed = seeds(m, k)
        e1 = stage1_evidence(m)
        n = len(m["half"])
        for g in grid():
            stops = stoppages(c, seed, *g, mode)
            es = np.stack([s1.to_steps(evidence_with_dead(e1, stops, w, m["half"])) for w in W_D])
            paths = viterbi_many(es, cost)
            for w, path in zip(W_D, paths):
                pred = bench.finish(m, s1.upsample(path, n))
                rows.append(dict(match=mt, param=g + (w,), mode=mode, n_stops=len(stops),
                                 **bench.score(m["labels"], pred)))
        print("  {} {} done in {:.0f} s".format(mode, mt, time.time() - t0), flush=True)
    return pd.DataFrame(rows)


def predict(m, param, mode, k=0):
    *g, w = param
    stops = stoppages(db.cues(m), seeds(m, k), *g, mode)
    e = evidence_with_dead(stage1_evidence(m), stops, w, m["half"])
    path = viterbi_many(s1.to_steps(e)[None], s1.cost_matrix(*STAGE1[3:]))[0]
    return bench.finish(m, s1.upsample(path, len(m["half"]))), stops


def d_f1(df):
    p, r = df["D_prec"], df["D_rec"]
    return 2 * p * r / (p + r)


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)
    t0 = time.time()
    path = os.path.join(CACHE, "v2_stage2_grid.csv")
    if os.path.isfile(path):
        g = pd.read_csv(path)
        g["param"] = g["param"].map(eval)
    else:
        g = pd.concat([run_grid("full"), run_grid("fallback")], ignore_index=True)
        g.to_csv(path, index=False)

    st1 = pd.read_csv(os.path.join(CACHE, "v2_stage1_lomo.csv"))
    print("== stage 1 (kept), degraded Sportec, LOMO ==")
    print(bench.summary(st1).to_string(index=False))

    results = {}
    for mode in ("full", "fallback"):
        lo = bench.lomo(g[g["mode"] == mode])
        results[mode] = lo
        print("\n== stage 2 {}, LOMO ==".format(mode))
        print(bench.summary(lo, cols=("spellF1", "macroF1", "D_prec", "D_rec", "n_pred", "n_gt", "n_stops")).to_string(index=False))
        print("params (v_slow, t_slow, t_alone, lag, w_d):", lo.param.tolist())

    full = results["full"]
    better = {k: full[k].mean() > st1[k].mean() for k in ("spellF1", "D_prec", "D_rec")}
    d1, d2 = d_f1(st1).mean(), d_f1(full).mean()
    print("\nGATE stage 2 vs stage 1 (degraded, LOMO): spellF1 {:.4f} vs {:.4f}, D prec {:.3f} vs {:.3f}, "
          "D rec {:.3f} vs {:.3f}, D F1 {:.3f} vs {:.3f}, macroF1 {:.4f} vs {:.4f}".format(
              full.spellF1.mean(), st1.spellF1.mean(), full.D_prec.mean(), st1.D_prec.mean(),
              full.D_rec.mean(), st1.D_rec.mean(), d2, d1, full.macroF1.mean(), st1.macroF1.mean()))
    print("  improved:", better)
    passed = all(better.values())
    fb = results["fallback"]
    print("fallback vs stage 1: spellF1 {:.4f}, D prec {:.3f}, D rec {:.3f}, D F1 {:.3f}".format(
        fb.spellF1.mean(), fb.D_prec.mean(), fb.D_rec.mean(), d_f1(fb).mean()))

    if passed:
        mode = "full"
    else:
        # The plan's rule: fall back to seeds plus the slowdown rule.
        mode = "fallback"
    print("kept: stage 2 {} ({})".format(mode, "gate passed" if passed else "gate failed, plan fallback"))
    final = bench.all_params_best(g[g["mode"] == mode])
    print("final setting (best on all 7 degraded):", final)

    clean = []
    for k, mt in enumerate(MATCHES):
        m = bench.load(mt)
        pred, stops = predict(m, final, mode, k)
        clean.append(dict(match=mt, n_stops=len(stops), **bench.score(m["labels"], pred)))
    print("\n== same setting on clean Sportec (reference) ==")
    print(bench.summary(pd.DataFrame(clean), cols=("spellF1", "macroF1", "D_prec", "D_rec", "n_pred", "n_gt", "n_stops")).to_string(index=False))

    ours = bench.load("ours")
    pred, stops = predict(ours, final, mode)
    c = db.cues(ours)
    sd = seeds(ours)
    covered = np.mean([any(s - 3 * FPS <= x <= e + 3 * FPS for s, e in stops) for x in sd])
    lens = np.array([e - s for s, e in stops]) / FPS
    print("ours: stoppages {} (median {:.0f} s); Smart Tagging set pieces inside one: {:.0%} of {}".format(
        len(stops), np.median(lens) if len(lens) else 0, covered, len(sd)))
    print("ours: shape frames " + ", ".join("{} {}".format(k, int(c[k].sum())) for k in db.SHAPES + ("out", "restart")))
    bench.report_ours(pred, "v2_stage2")
    print("stage 2 took {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
