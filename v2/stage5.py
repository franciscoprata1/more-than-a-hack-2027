"""Stage 5: a classifier trained on the bench mixes all the clues, in place of hand-set weights.

    .venv\\Scripts\\python.exe v2/stage5.py              # one network
    .venv\\Scripts\\python.exe v2/stage5.py --nets 5     # 5 networks averaged, gated against one

Every clue of Stages 1-4 is kept apart per 5 Hz step (nearest team, owner and control, ball in
flight, last team in control, stoppage, set-piece shapes, seed restarts, ball missing and for how
long, the gap model's p_nobody, out / near a line / still), plus Stage 4's own evidence, plus context:
centred means over CONTEXT_S and who was in control in the PAST_S before and after. A gradient
small neural network (two hidden layers, PyTorch) learns H / A / D per step. (scikit-learn's
trees were the first choice; Windows Smart App Control blocks one of its DLLs, so not used.) Every training step is also used with H and A
swapped, so it cannot learn a favourite team.

Its evidence per step: ALPHA x (log p - log prior), decoded by Stage 3's Viterbi (cheaper switch at a
controlled turnover). Tuned: ALPHA, the H <-> A switch cost, the discount and a scale on the costs in
and out of D.

Nested leave one match out, so no number on match k has seen match k:
  - for each match k, for each other match j: a network trained on the 5 matches that are neither k nor j
    scores j; the settings best on those 6 are the ones match k gets;
  - the network trained on all 6 but k then fills k with them.
Ours: a network trained on all 7, the settings best on the 7 outer fills.
With --nets N, every place a network is trained trains N (seeds NET_SEED, NET_SEED+1, ...) and
averages their probabilities: one network's luck in its random start is smoothed out. Kept with 5:
evaluation/results/v2_stage5_n5.csv is the final submission. Tried and not kept (5 Nov): pitch
control clues (who reaches the ball first), spellF1 +0.0004, no real gain.
Gate (bench.gate): D F1 up and spellF1 not down, one network against Stage 4, N networks against one.
Writes cache/v2_stage5/<match>.npz (features), cache/v2_stage5[_n<N>]_grid.csv,
evaluation/results/v2_stage5[_n<N>].csv.
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import deadball as db  # noqa: E402
import stage1 as s1  # noqa: E402
import stage2 as s2  # noqa: E402
import stage3 as s3  # noqa: E402
import stage3_control as s3c  # noqa: E402
import stage4 as s4  # noqa: E402
from bench import H, A, D, X, MATCHES, CACHE  # noqa: E402

# Params
FPS = 25
STAGE4 = (0.2, 1.0, True, 1.0, True, 0.1, 0.3, 0.5, 0.2, 0.3)   # its kept evidence setting
SEED_S = 3.0             # a seed restart this close to the step
GAP_CAP_S = 60.0
CONTEXT_S = (2, 10)      # centred means
PAST_S = 5               # control before and after
TRAIN_EVERY = 2          # neighbouring steps are near copies: train on every 2nd
HIDDEN = 64
EPOCHS = 8
BATCH = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NET_SEED = 2027
NETS = 1                 # networks averaged; set by --nets
ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0)     # 4 added: the first try picked the edge
C_TEAMS = (2, 4, 8, 16, 32)          # 2 added: the first try picked the edge
DISCOUNTS = (1.0, 0.5, 0.25)
D_SCALES = (0.5, 1.0, 2.0)
OUT_DIR = os.path.join(CACHE, "v2_stage5")
STEP_HZ = FPS // s1.STEP


def step_mean(v):
    """Per-frame array (n,) or (n, k) -> mean over each 5-frame step."""
    v = v.astype(np.float64).reshape(len(v), -1)
    pad = (-len(v)) % s1.STEP
    v = np.vstack([v, np.full((pad, v.shape[1]), np.nan)])
    with np.errstate(invalid="ignore"):
        return np.nanmean(v.reshape(-1, s1.STEP, v.shape[1]), axis=1)


def last_in_control(c, ctl):
    """Per frame: the last team with a controlled owner spell, -1 before any or at a half's start."""
    last = np.full(c.n, -1, dtype=np.int8)
    for (a, _, _, t), ok in zip(c.eps, ctl):
        if ok:
            last[a] = t
    out = np.full(c.n, -1, dtype=np.int8)
    cur = -1
    for i in range(c.n):
        if c.half[i] == 0:
            cur = -1
        elif last[i] >= 0:
            cur = last[i]
        out[i] = cur
    return out


def base_features(m, k, name):
    """Per step: (features, names, kinds). kind 'sym' stays on an H/A swap, 'anti' flips sign,
    'H'/'A' pairs swap with each other (a pair is listed H first, then A)."""
    c4 = s4.Cues(m, k, name)
    c = c4.c
    n = c.n
    e4, turns = c4.build(STAGE4[:5], *STAGE4[5:])
    ctl = c.control(STAGE4[0])
    own_ctl, own_unc = np.zeros((n, 2)), np.zeros((n, 2))
    for (a, b, _, t), ok in zip(c.eps, ctl):
        (own_ctl if ok else own_unc)[a:b, t] = 1
    lt = last_in_control(c, ctl)
    e1 = s2.stage1_evidence(m)
    dh, da, _ = s1.distances(m)
    *g, _ = s3.STAGE2
    cues = db.cues(m)
    stops = s2.stoppages(cues, s2.seeds(m, k), *g, "full")
    in_stop = np.zeros(n)
    for a, b in stops:
        in_stop[a:b] = 1
    seed = np.zeros(n)
    w = int(SEED_S * FPS)
    for f in s2.seeds(m, k):
        seed[max(0, f - w):f + w] = 1
    restart = np.convolve(cues["restart"].astype(float), np.ones(FPS), "same") > 0
    L, W = m["pitch"]
    bx, by = m["ball_x"].astype(np.float64), m["ball_y"].astype(np.float64)
    inside = np.minimum.reduce([bx, L - bx, by, W - by])
    state = m["ball_state"]
    sign = np.array([[c.sign.get((h, t), 0) for t in (0, 1)] for h in (0, 1, 2)])[c.half]
    fwd = c.flight[:, None] & (c.vx[:, None] * sign > 0)

    cols = [
        ("nearest_team", e1[:, H] - e1[:, A], "anti"),
        ("nearest_dead", e1[:, D] - np.maximum(e1[:, H], e1[:, A]), "sym"),
        ("dist_H", np.minimum(dh, 50), "H"), ("dist_A", np.minimum(da, 50), "A"),
        ("own_ctl_H", own_ctl[:, 0], "H"), ("own_ctl_A", own_ctl[:, 1], "A"),
        ("own_unc_H", own_unc[:, 0], "H"), ("own_unc_A", own_unc[:, 1], "A"),
        ("last_ctl_H", lt == 0, "H"), ("last_ctl_A", lt == 1, "A"),
        ("flight_fwd_H", fwd[:, 0], "H"), ("flight_fwd_A", fwd[:, 1], "A"),
        ("opp_near", c.opp_near, "sym"), ("flight", c.flight, "sym"),
        ("in_stop", in_stop, "sym"), ("seed_near", seed, "sym"), ("restart", restart, "sym"),
        ("crowd", np.nan_to_num(cues["crowd"], nan=0.0), "sym"),
        ("still", cues["seen"] & (np.nan_to_num(m["ball_speed"], nan=99.0) < db.STILL_SPEED), "sym"), ("out_seen", cues["out"], "sym"),
    ] + [(s, cues[s], "sym") for s in db.SHAPES] + [
        ("seen", state == 0, "sym"), ("filled", state == 1, "sym"), ("missing", state == 2, "sym"),
        ("gap_s", np.minimum(m["ball_gap"] / FPS, GAP_CAP_S), "sym"),
        ("ball_speed", np.nan_to_num(m["ball_speed"], nan=-1.0), "sym"),
        ("ball_inside_m", np.nan_to_num(np.clip(inside, -10, 40), nan=-20.0), "sym"),
        ("p_nobody_logit", c4.lpn, "sym"), ("gap_out", c4.out, "sym"),
        ("gap_near_line", c4.near, "sym"), ("gap_still", c4.still, "sym"),
    ]
    names = [nm for nm, _, _ in cols]
    kinds = [kd for _, _, kd in cols]
    f = step_mean(np.stack([np.asarray(v, dtype=np.float64) for _, v, _ in cols], axis=1))
    # Stage 4's own evidence per step (already summed over the step)
    f = np.hstack([f, (e4[:, H] - e4[:, A])[:, None], (e4[:, D] - np.maximum(e4[:, H], e4[:, A]))[:, None]])
    names += ["stage4_team", "stage4_dead"]
    kinds += ["anti", "sym"]
    near = c.near(turns, len(e4))
    return np.nan_to_num(f[:len(e4)]), names, kinds, near



CONTEXT_OF = ("nearest_team", "own_ctl_H", "own_ctl_A", "flight", "in_stop", "crowd", "missing",
              "p_nobody_logit", "out_seen", "throw", "corner", "goalkick", "freekick", "kickoff", "stage4_team",
              "stage4_dead")
PAST_OF = ("own_ctl_H", "own_ctl_A", "nearest_team", "stage4_team")


def run_mean_steps(v, w, half):
    """Centred mean over w steps, not across halves."""
    out = np.zeros_like(v)
    for h in (0, 1, 2):
        idx = np.flatnonzero(half == h)
        if len(idx) == 0:
            continue
        seg = v[idx]
        cs = np.vstack([np.zeros((1, seg.shape[1])), np.cumsum(seg, axis=0)])
        i = np.arange(len(seg))
        lo, hi = np.maximum(0, i - w // 2), np.minimum(len(seg), i + w // 2 + 1)
        out[idx] = (cs[hi] - cs[lo]) / (hi - lo)[:, None]
    return out


def shifted_mean(v, w, half, ahead):
    """Mean over the w steps before (ahead=False) or after (True) each step, inside its half."""
    out = np.zeros_like(v)
    for h in (0, 1, 2):
        idx = np.flatnonzero(half == h)
        if len(idx) == 0:
            continue
        seg = v[idx]
        cs = np.vstack([np.zeros((1, seg.shape[1])), np.cumsum(seg, axis=0)])
        i = np.arange(len(seg))
        lo, hi = (i + 1, np.minimum(len(seg), i + 1 + w)) if ahead else (np.maximum(0, i - w), i)
        cnt = np.maximum(hi - lo, 1)
        out[idx] = np.where((hi > lo)[:, None], (cs[np.minimum(hi, len(seg))] - cs[np.minimum(lo, len(seg))]) / cnt[:, None], 0)
    return out


def swap(base, names, kinds):
    """The same steps with H and A swapped."""
    out = base.copy()
    for j, kd in enumerate(kinds):
        if kd == "anti":
            out[:, j] = -base[:, j]
        elif kd == "H":
            out[:, j], out[:, j + 1] = base[:, j + 1], base[:, j]
    return out


def full_features(base, names, half):
    """Base plus context."""
    idx = [names.index(nm) for nm in CONTEXT_OF]
    pidx = [names.index(nm) for nm in PAST_OF]
    parts = [base]
    for s in CONTEXT_S:
        parts.append(run_mean_steps(base[:, idx], s * STEP_HZ, half))
    parts.append(shifted_mean(base[:, pidx], PAST_S * STEP_HZ, half, False))
    parts.append(shifted_mean(base[:, pidx], PAST_S * STEP_HZ, half, True))
    return np.hstack(parts).astype(np.float32)


def step_half(m, n_steps):
    return m["half"][np.minimum(np.arange(n_steps) * s1.STEP + s1.STEP // 2, len(m["half"]) - 1)]


def step_labels(m, n_steps):
    return m["labels"][np.minimum(np.arange(n_steps) * s1.STEP + s1.STEP // 2, len(m["labels"]) - 1)]


def features(name, k):
    """Cached per match: X (steps, features), X with H/A swapped, step labels, step half, near."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name + ".npz")
    if os.path.isfile(path):
        z = np.load(path)
        return {key: z[key] for key in z.files}
    m = bench.load(name if name == "ours" else name + "~")
    base, names, kinds, near = base_features(m, k, name)
    half = step_half(m, len(base))
    out = {"X": full_features(base, names, half), "Xs": full_features(swap(base, names, kinds), names, half),
           "half": half, "near": near, "n_base": np.array(len(names))}
    if name != "ours":
        out["y"] = step_labels(m, len(base))
    np.savez_compressed(path, **out)
    return out


class Net:
    """Two hidden layers on standardised features; predict_proba as the trees would give it."""

    def fit(self, x, y, seed=NET_SEED):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.mu, self.sd = x.mean(axis=0), x.std(axis=0) + 1e-6
        xt = torch.from_numpy(((x - self.mu) / self.sd).astype(np.float32))
        yt = torch.from_numpy(y.astype(np.int64))
        self.net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], HIDDEN), torch.nn.ReLU(),
                                       torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(), torch.nn.Linear(HIDDEN, 3))
        opt = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        for _ in range(EPOCHS):
            order = torch.from_numpy(rng.permutation(len(y)))
            for i in range(0, len(y), BATCH):
                b = order[i:i + BATCH]
                loss = torch.nn.functional.cross_entropy(self.net(xt[b]), yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, x):
        with torch.no_grad():
            z = self.net(torch.from_numpy(((x - self.mu) / self.sd).astype(np.float32)))
            return torch.softmax(z, dim=1).numpy()


class Ensemble:
    """The mean probability of several networks."""

    def __init__(self, nets):
        self.nets = nets

    def predict_proba(self, x):
        return np.mean([n.predict_proba(x) for n in self.nets], axis=0)


def suffix(nets):
    return "" if nets == 1 else "_n{}".format(nets)


def grid_path(nets):
    return os.path.join(CACHE, "v2_stage5{}_grid.csv".format(suffix(nets)))


def train(feats, names):
    """A network trained on these matches, every TRAIN_EVERY-th scored step, H/A swapped copies included."""
    Xs, ys = [], []
    swap_y = np.array([A, H, D, X])
    for nm in names:
        f = feats[nm]
        keep = np.zeros(len(f["y"]), dtype=bool)
        keep[::TRAIN_EVERY] = True
        keep &= (f["half"] > 0) & (f["y"] != X)
        Xs += [f["X"][keep], f["Xs"][keep]]
        ys += [f["y"][keep], swap_y[f["y"][keep]]]
    y = np.concatenate(ys)
    x = np.vstack(Xs)
    clf = Ensemble([Net().fit(x, y, NET_SEED + i) for i in range(NETS)])
    prior = np.array([(y == c).mean() for c in (H, A, D)])
    return clf, prior


def evidence(clf, prior, f):
    """log p - log prior per step (steps, 3); 0 outside the halves."""
    p = np.clip(clf.predict_proba(f["X"]), 1e-4, 1)
    e = np.log(p) - np.log(prior)
    e[f["half"] == 0] = 0
    return e


def cost_grid():
    return list(itertools.product(C_TEAMS, DISCOUNTS, D_SCALES))


def cost_arrays():
    _, _, _, _, c_in, c_out = s2.STAGE1
    return (np.array([s1.cost_matrix(ct, c_in * ds, c_out * ds) for ct, _, ds in cost_grid()]),
            np.array([d for _, d, _ in cost_grid()]))


def score_all(m, e, near):
    """Rows of (alpha, costs) -> scores for one match."""
    costs, disc = cost_arrays()
    n = len(m["half"])
    rows = []
    for a in ALPHAS:
        paths = s3.viterbi_touch(a * e, costs, disc, near)
        for cg, path in zip(cost_grid(), paths):
            pred = bench.finish(m, s1.upsample(path, n))
            rows.append(dict(param=(a,) + cg, **bench.score(m["labels"], pred)))
    return rows


def run_grid(feats, matches):
    """Inner rows (outer, match) and outer evidence per match."""
    rows = []
    outer_e = {}
    for k, mk in enumerate(MATCHES):
        t0 = time.time()
        for j, mj in enumerate(MATCHES):
            if j == k:
                continue
            clf, prior = train(feats, [nm for nm in MATCHES if nm not in (mk, mj)])
            for r in score_all(matches[mj], evidence(clf, prior, feats[mj]), feats[mj]["near"]):
                rows.append(dict(outer=mk, match=mj, **r))
        clf, prior = train(feats, [nm for nm in MATCHES if nm != mk])
        outer_e[mk] = evidence(clf, prior, feats[mk])
        for r in score_all(matches[mk], outer_e[mk], feats[mk]["near"]):
            rows.append(dict(outer="outer", match=mk, **r))
        print("  outer {} done in {:.0f} s".format(mk, time.time() - t0), flush=True)
    return pd.DataFrame(rows)


def nested(g):
    """Nested LOMO: match k takes the setting best on the inner fills of the other 6."""
    outer = g[g.outer == "outer"]
    rows = []
    for mk in MATCHES:
        best = g[g.outer == mk].groupby("param").spellF1.mean().idxmax()
        rows.append(outer[(outer.match == mk) & (outer.param == best)].iloc[0].to_dict())
    return pd.DataFrame(rows), outer


def main():
    global NETS
    parser = argparse.ArgumentParser()
    parser.add_argument("--nets", type=int, default=1)
    NETS = parser.parse_args().nets
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 3)
    t0 = time.time()
    feats = {mt: features(mt, k) for k, mt in enumerate(MATCHES)}
    feats["ours"] = features("ours", 0)
    print("features: {} per step ({} clues + context); built in {:.0f} s".format(
        feats["ours"]["X"].shape[1], int(feats["ours"]["n_base"]), time.time() - t0), flush=True)
    matches = {mt: bench.load(mt + "~") for mt in MATCHES}
    path = grid_path(NETS)
    if not os.path.isfile(path):
        g = run_grid(feats, matches)
        g["param"] = g["param"].map(str)
        g.to_csv(path, index=False)
    lo, outer = nested(s4.read_grid(path))
    name = "stage 5" + ("" if NETS == 1 else ", {} networks".format(NETS))
    lo4 = bench.lomo(s4.read_grid(s4.GRID_PATH))
    extra4, extra5 = [], []
    for k, mt in enumerate(MATCHES):
        m = matches[mt]
        p4 = s4.predict(m, lo4[lo4.match == mt].param.iloc[0], k, mt)[0]
        a, ct, d, ds = lo[lo.match == mt].param.iloc[0]
        _, _, _, _, c_in, c_out = s2.STAGE1
        f = feats[mt]
        clf, prior = train(feats, [nm for nm in MATCHES if nm != mt])
        path = s3.viterbi_touch(a * evidence(clf, prior, f), s1.cost_matrix(ct, c_in * ds, c_out * ds)[None],
                                np.array([d]), f["near"])[0]
        p5 = bench.finish(m, s1.upsample(path, len(m["half"])))
        extra4.append(s4.gap_d(m, p4))
        extra5.append(s4.gap_d(m, p5))
    gcols = ("gap_called_D", "gap_true_D", "gap_D_prec", "gap_D_rec")
    for df, ex in ((lo4, extra4), (lo, extra5)):
        for col in gcols:
            df[col] = [x[col] for x in ex]
    s3.show("stage 4, LOMO", lo4, s3.COLS + gcols)
    if NETS > 1:
        lo1, _ = nested(s4.read_grid(grid_path(1)))
        s3.show("stage 5, one network, nested LOMO", lo1, s3.COLS)
    s3.show(name + ", nested LOMO", lo, s3.COLS + gcols)
    print(name + " settings per match (alpha, c_team, discount, D cost scale):")
    for mt, p in zip(lo.match, lo.param):
        print("  ", mt, p)
    base, base_name = (lo4, "stage 4") if NETS == 1 else (lo1, "stage 5, one network")
    passed = bench.gate(lo, base)
    print("{} vs {}: spellF1 {:.4f} vs {:.4f}, D F1 {:.3f} vs {:.3f}, macroF1 {:.4f} vs {:.4f} -> {}".format(
        name, base_name, lo.spellF1.mean(), base.spellF1.mean(), lo.D_F1.mean(), base.D_F1.mean(), lo.macroF1.mean(),
        base.macroF1.mean(), "PASS" if passed else "FAIL"))
    if NETS > 1:
        print("  spellF1 better on {} of 7 matches".format(int((lo.spellF1.values > lo1.spellF1.values).sum())))
    if not passed:
        print("\ngate failed: {} stays the kept version; no submission written".format(base_name))
        print("took {:.0f} s".format(time.time() - t0))
        return

    final = bench.all_params_best(outer)
    print("\nkept: " + name + "; final setting (best on the 7 outer fills): {}".format(final))
    clf, prior = train(feats, list(MATCHES))
    ours = bench.load("ours")
    a, ct, d, ds = final
    _, _, _, _, c_in, c_out = s2.STAGE1
    f = feats["ours"]
    path = s3.viterbi_touch(a * evidence(clf, prior, f), s1.cost_matrix(ct, c_in * ds, c_out * ds)[None],
                            np.array([d]), f["near"])[0]
    pred = bench.finish(ours, s1.upsample(path, len(ours["half"])))
    gap = (ours["ball_state"] == 2) & (ours["half"] > 0)
    p4 = s4.predict(ours, bench.all_params_best(s4.read_grid(s4.GRID_PATH)), 0, "ours")[0]
    sc = ours["half"] > 0
    print("ours: frames changed against stage 4 {:.1%}; ball missing over 1 s called D {:.1%} (stage 4 {:.1%})".format(
        (pred[sc] != p4[sc]).mean(), (pred[gap] == D).mean(), (p4[gap] == D).mean()))
    if NETS > 1:
        one = pd.read_csv(os.path.join(bench.RESULTS, "v2_stage5.csv"))["label"].values
        print("ours: frames changed against one network {:.1%}".format((bench.LETTERS[pred][sc] != one[sc]).mean()))
    bench.report_ours(pred, "v2_stage5" + suffix(NETS))
    print("took {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
