"""Stage 1: the layered skeleton. Nearest team as evidence, Viterbi with switch costs, X where unsure.

    .venv\\Scripts\\python.exe v2/stage1.py

1. Evidence, only where the ball is seen or filled across <= 1 s (v1 smoothing), never in a gap:
     p(H vs A) = sigmoid((d_away - d_home) / SCALE), d = distance from the ball to that team's nearest
     player. Nobody within LOOSE_M: no team evidence (a ball in flight belongs to nobody yet).
     p(D) = Q_OUT when the whole ball is past a line, else Q_IN.
   Where the ball is missing there is no evidence (plan); MISSING_D adds a small per-frame lean to D
   there, tested as a variant (0 = the plan as written).
2. Viterbi over H, A, D with three switch costs: H<->A, alive->D, D->alive. Decoded at 5 Hz (evidence
   summed over 5 frames) to keep the grid fast; a 0.2 s step is far inside the 1 s transition tolerance.
3. X: forward-backward posteriors with the same costs. Inside a possession (never its first frame,
   so the scorer's forward fill keeps every possession whole) a frame whose label has posterior below
   X_BELOW becomes X. This only changes macroF1.

All costs and thresholds are tuned on degraded Sportec, leave one match out (LOMO). The final run on
our match uses the setting best on all 7 degraded matches; the 5 minute answer key is report only.
Gate: beat the organisers' majority smoothing on degraded Sportec (spellF1), else keep it.

Writes cache/v2_stage1_grid.csv, cache/v2_stage1_lomo.csv, evaluation/results/v2_stage1.csv.
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
from bench import H, A, D, X, MATCHES, CACHE  # noqa: E402

# Params
STEP = 5                 # frames per decoding step (5 Hz)
LOOSE_M = 5.0            # football sense: nobody within 5 m, nobody controls the ball
OUT_M = 0.5              # ball centre this far past a line: out (a little generous for tracking noise)
Q_OUT = 0.95
FAR = 50.0               # distance used when a team has no tracked player
SCALES = (1.0, 2.0, 4.0)             # metres
Q_INS = (0.05, 0.2)
MISSING_DS = (0.0, 0.005, 0.02)      # nats per frame towards D while the ball is missing
C_TEAM = (2, 4, 8, 16, 32)           # nats
C_IN = (4, 8, 16, 32)
C_OUT = (4, 8, 16, 32)
X_BELOWS = (0.0, 0.5, 0.6, 0.7, 0.8)


def usable(m):
    """Frames with a ball to read: seen, or filled across a gap of at most 1 s."""
    return (m["ball_state"] < 2) & (m["half"] > 0) & ~np.isnan(m["ball_x"])


def distances(m):
    bx, by = m["ball_x"].astype(np.float64), m["ball_y"].astype(np.float64)
    dh, da = bench.nearest_by_team(bx, by, m["player_x"], m["player_y"], m["team"])
    L, W = m["pitch"]
    out = (bx < -OUT_M) | (bx > L + OUT_M) | (by < -OUT_M) | (by > W + OUT_M)
    return np.nan_to_num(dh, nan=FAR), np.nan_to_num(da, nan=FAR), out


def evidence(m, dist, scale, q_in, missing_d):
    """Per-frame log-likelihoods (T, 3) for H, A, D."""
    dh, da, out = dist
    ok = usable(m)
    p = 1 / (1 + np.exp(-(da - dh) / scale))
    p = np.where(np.minimum(dh, da) > LOOSE_M, 0.5, np.clip(p, 1e-3, 1 - 1e-3))
    q = np.where(out, Q_OUT, q_in)
    e = np.stack([np.log((1 - q) * p), np.log((1 - q) * (1 - p)), np.log(q)], axis=1)
    e -= e.max(axis=1, keepdims=True)
    e[~ok] = 0.0
    e[~ok & (m["half"] > 0), D] = missing_d
    return e


def to_steps(e):
    n = len(e)
    pad = (-n) % STEP
    e = np.vstack([e, np.zeros((pad, 3))])
    return e.reshape(-1, STEP, 3).sum(axis=1)


def cost_matrix(c_team, c_in, c_out):
    c = np.zeros((3, 3))
    c[H, A] = c[A, H] = c_team
    c[H, D] = c[A, D] = c_in
    c[D, H] = c[D, A] = c_out
    return c


def viterbi(e, costs):
    """Best paths for G cost matrices at once. e (T, 3) log-lik, costs (G, 3, 3) -> (G, T) int8."""
    T, G = len(e), len(costs)
    neg = -e
    delta = np.repeat(neg[:1], G, axis=0)
    bp = np.empty((T, G, 3), dtype=np.int8)
    for t in range(1, T):
        cand = delta[:, :, None] + costs
        bp[t] = cand.argmin(axis=1)
        delta = cand.min(axis=1) + neg[t]
    path = np.empty((G, T), dtype=np.int8)
    s = delta.argmin(axis=1)
    g = np.arange(G)
    for t in range(T - 1, -1, -1):
        path[:, t] = s
        s = bp[t, g, s]
    return path


def posteriors(e, cost):
    """Forward-backward marginals (T, 3) with transition weights exp(-cost)."""
    T = len(e)
    logt = -cost
    fw = np.empty((T, 3))
    bw = np.zeros((T, 3))
    fw[0] = e[0]
    for t in range(1, T):
        a = fw[t - 1][:, None] + logt
        mx = a.max(axis=0)
        fw[t] = mx + np.log(np.exp(a - mx).sum(axis=0)) + e[t]
    for t in range(T - 2, -1, -1):
        a = logt + (e[t + 1] + bw[t + 1])[None, :]
        mx = a.max(axis=1)
        bw[t] = mx + np.log(np.exp(a - mx[:, None]).sum(axis=1))
    lp = fw + bw
    lp -= lp.max(axis=1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(axis=1, keepdims=True)


def upsample(steps, n):
    return np.repeat(steps, STEP)[:n]


def add_x(labels, post_frames, x_below):
    """X inside a possession where the label's posterior is low; never a possession's first frame."""
    if x_below <= 0:
        return labels
    out = labels.copy()
    conf = post_frames[np.arange(len(labels)), np.minimum(labels, 2)]
    first = np.r_[True, labels[1:] != labels[:-1]]
    out[(labels < D) & (conf < x_below) & ~first] = X
    return out


def emission_grid():
    return list(itertools.product(SCALES, Q_INS, MISSING_DS))


def cost_grid():
    return list(itertools.product(C_TEAM, C_IN, C_OUT))


def run_grid():
    costs = np.array([cost_matrix(*c) for c in cost_grid()])
    rows = []
    for mt in MATCHES:
        t0 = time.time()
        m = bench.load(mt + "~")
        dist = distances(m)
        n = len(m["labels"])
        for scale, q_in, md in emission_grid():
            paths = viterbi(to_steps(evidence(m, dist, scale, q_in, md)), costs)
            for (ct, ci, co), path in zip(cost_grid(), paths):
                pred = bench.finish(m, upsample(path, n))
                s = bench.score(m["labels"], pred)
                rows.append(dict(match=mt, param=(scale, q_in, md, ct, ci, co), **s))
        print("  {} done in {:.0f} s".format(mt, time.time() - t0), flush=True)
    return pd.DataFrame(rows)


def predict(m, param, x_below=0.0):
    scale, q_in, md, ct, ci, co = param
    e = to_steps(evidence(m, distances(m), scale, q_in, md))
    cost = cost_matrix(ct, ci, co)
    n = len(m["half"])
    lab = upsample(viterbi(e, cost[None])[0], n)
    if x_below > 0:
        post = np.repeat(posteriors(e, cost), STEP, axis=0)[:n]
        lab = add_x(lab, post, x_below)
    return bench.finish(m, lab)


def x_table(chosen):
    """macroF1 (and spellF1, which must not move) for each X threshold, per match, at its LOMO param."""
    rows = []
    for mt in MATCHES:
        m = bench.load(mt + "~")
        param = chosen[mt]
        scale, q_in, md, ct, ci, co = param
        e = to_steps(evidence(m, distances(m), scale, q_in, md))
        cost = cost_matrix(ct, ci, co)
        n = len(m["half"])
        lab = upsample(viterbi(e, cost[None])[0], n)
        post = np.repeat(posteriors(e, cost), STEP, axis=0)[:n]
        for xb in X_BELOWS:
            s = bench.score(m["labels"], bench.finish(m, add_x(lab, post, xb)))
            rows.append(dict(match=mt, param=xb, **s))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)
    t0 = time.time()
    grid_path = os.path.join(CACHE, "v2_stage1_grid.csv")
    if os.path.isfile(grid_path):
        grid = pd.read_csv(grid_path)
        grid["param"] = grid["param"].map(eval)
    else:
        grid = run_grid()
        grid.to_csv(grid_path, index=False)

    base = pd.read_csv(os.path.join(CACHE, "v2_baseline_degraded.csv"))
    base_lo = bench.lomo(base)
    print("== majority smoothing, degraded Sportec, LOMO window ==")
    print(bench.summary(base_lo).to_string(index=False))

    plan = grid[grid.param.map(lambda p: p[2] == 0.0)]
    lo_plan = bench.lomo(plan)
    print("\n== stage 1 as planned (no evidence while missing), LOMO ==")
    print(bench.summary(lo_plan).to_string(index=False))
    print("params:", lo_plan.param.tolist())

    lo = bench.lomo(grid)
    print("\n== stage 1 with the missing-ball lean to D in the grid, LOMO ==")
    print(bench.summary(lo).to_string(index=False))
    print("params:", lo.param.tolist())

    # The version kept: the plan as written unless the variant wins by more than the noise band.
    gain = lo.spellF1.mean() - lo_plan.spellF1.mean()
    use = lo if gain > 0.01 else lo_plan
    use_grid = grid if gain > 0.01 else plan
    print("\nkept: {} (variant gain {:+.4f} spellF1)".format("variant" if use is lo else "plan", gain))

    chosen = dict(zip(use.match, use.param))
    xt = x_table(chosen)
    xlo = bench.lomo(xt, key="macroF1")
    print("\n== X threshold, LOMO on macroF1 ==")
    print(xt.groupby("param")[["spellF1", "macroF1"]].mean().round(4).to_string())
    print(bench.summary(xlo).to_string(index=False), " X thresholds:", xlo.param.tolist())
    xlo.to_csv(os.path.join(CACHE, "v2_stage1_lomo.csv"), index=False)

    gate = xlo.spellF1.mean() > base_lo.spellF1.mean()
    print("\nGATE stage 1 vs majority smoothing (degraded, LOMO): spellF1 {:.4f} vs {:.4f}, macroF1 {:.4f} vs {:.4f} -> {}".format(
        xlo.spellF1.mean(), base_lo.spellF1.mean(), xlo.macroF1.mean(), base_lo.macroF1.mean(),
        "PASS" if gate else "FAIL (keep majority smoothing)"))

    final = bench.all_params_best(use_grid)
    final_x = bench.all_params_best(xt, key="macroF1")
    print("final setting (best on all 7 degraded):", final, "X below", final_x)

    clean = []
    for mt in MATCHES:
        m = bench.load(mt)
        clean.append(dict(match=mt, **bench.score(m["labels"], predict(m, final, final_x))))
    print("\n== same setting on clean Sportec (reference) ==")
    print(bench.summary(pd.DataFrame(clean)).to_string(index=False))

    ours = bench.load("ours")
    if gate:
        pred = predict(ours, final, final_x)
    else:
        pred = bench.finish(ours, bench.majority(bench.baseline_codes(ours), int(base_lo.param.mode()[0])))
    bench.report_ours(pred, "v2_stage1")
    print("stage 1 took {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
