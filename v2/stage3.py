"""Stage 3: the seen ball. Possession zone, duels, touches and the two-touch rule on the skeleton.

    .venv\\Scripts\\python.exe v2/stage3.py

Starts from Stage 2 on the Stage 1 skeleton (its evidence, dead ball layer and costs, all fixed: one
change at a time) and adds, from v2/touches.py, only where the ball is really seen:
  - zone: ball within 1 m of one team only: W_ZONE nats per frame towards that team (a duel, both
    teams within 1 m, adds nothing: the ball is contested);
  - touches: W_TOUCH nats towards the touching team at each confirmed touch;
  - two-touch rule: a touch not confirmed by a second touch or by holding the ball (a deflection)
    gets W1_FRAC of that weight (0 = the rule on, 1 = the rule off);
  - cheaper switches at a touch: changing possession to a team costs DISCOUNT x C_TEAM within
    TOUCH_WIN of that team's (counted) touch, C_TEAM elsewhere. C_TEAM itself is re-tuned, since
    short possessions need switches that are cheap where a touch backs them.
With all weights 0, discount 1 and C_TEAM 16, this is exactly Stage 2 on the skeleton (checked:
identical labels on J03WPY).

Tuned on degraded Sportec, LOMO. Gate (Stage 3 rule): spellF1 up and D F1 not down against Stage 2
on the skeleton; compared also with Stage 2 on majority smoothing. Also reports scores on seen-ball
frames only (H/A accuracy where play is on, and macroF1), since Stage 3 only changes evidence there.
Writes cache/v2_stage3_grid.csv, evaluation/results/v2_stage3.csv.
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
import stage2 as s2  # noqa: E402
import stage2_majority as s2m  # noqa: E402
import touches as tc  # noqa: E402
from bench import H, A, D, MATCHES, CACHE  # noqa: E402

# Params
FPS = 25
W_ZONES = (0.0, 0.3, 1.0)          # nats per frame
W_TOUCHES = (0.0, 3.0, 8.0)        # nats per confirmed touch
W1_FRACS = (0.0, 1.0)              # unconfirmed touch weight, share of W_TOUCH
C_TEAMS = (4, 8, 16, 32)           # nats, switch H <-> A away from touches
DISCOUNTS = (1.0, 0.5, 0.25)
TOUCH_WIN = 2                      # steps (0.2 s each) either side of a touch
COLS = ("spellF1", "macroF1", "D_prec", "D_rec", "D_F1", "n_pred", "n_gt")
SEEN_COLS = ("seen_HA_acc", "seen_macroF1", "missing_HA_acc", "missing_macroF1")


def stage2_setting():
    g = pd.read_csv(os.path.join(CACHE, "v2_stage2_grid.csv"))
    g["param"] = g["param"].map(eval)
    return bench.all_params_best(g[g["mode"] == "full"])


STAGE2 = stage2_setting()


def stage2_evidence(m, k):
    *g, w_d = STAGE2
    stops = s2.stoppages(db.cues(m), s2.seeds(m, k), *g, "full")
    return s2.evidence_with_dead(s2.stage1_evidence(m), stops, w_d, m["half"])


def evidence_grid():
    return list(itertools.product(tc.A_TOUCHES, W_ZONES, W_TOUCHES, W1_FRACS))


def cost_grid():
    return list(itertools.product(C_TEAMS, DISCOUNTS))


class Cues:
    """Per match: Stage 2 evidence and the touch cues for each threshold."""

    def __init__(self, m, k):
        self.n = len(m["half"])
        self.e2 = stage2_evidence(m, k)
        dist = s1.distances(m)
        self.zone = tc.zone(dist)
        seen = (m["ball_state"] == 0) & (m["half"] > 0)
        self.zone[~seen] = -1
        dv = tc.velocity_change(m)
        self.touch = {a: tc.touches(m, a, dist, dv) for a in tc.A_TOUCHES}

    def evidence(self, a, w_zone, w_touch, w1):
        e = self.e2.copy()
        for t in (H, A):
            e[self.zone == t, t] += w_zone
        f, team, _, conf = self.touch[a]
        w = np.where(conf, w_touch, w_touch * w1)
        np.add.at(e, (f, team.astype(int)), w)
        return s1.to_steps(e)

    def near(self, a, w1, n_steps):
        """(steps, 2) bool: a counted touch by that team within TOUCH_WIN steps."""
        f, team, _, conf = self.touch[a]
        use = conf | (w1 > 0)
        out = np.zeros((n_steps + 2 * TOUCH_WIN + 1, 2), dtype=bool)
        st = f[use] // s1.STEP
        for d in range(-TOUCH_WIN, TOUCH_WIN + 1):
            out[st + d + TOUCH_WIN, team[use]] = True
        return out[TOUCH_WIN:TOUCH_WIN + n_steps]


def viterbi_touch(e, costs, disc, near):
    """Viterbi for one evidence set (T, 3) under G cost matrices (G, 3, 3); a switch into a team
    within reach of its touch costs disc (G,) times the usual H <-> A cost."""
    T, G = len(e), len(costs)
    neg = -e
    delta = np.repeat(neg[:1], G, axis=0)
    bp = np.empty((T, G, 3), dtype=np.int8)
    cut = costs[:, H, A] * (1 - disc)
    for t in range(1, T):
        cand = delta[:, :, None] + costs
        if near[t, A]:
            cand[:, H, A] -= cut
        if near[t, H]:
            cand[:, A, H] -= cut
        bp[t] = cand.argmin(axis=1)
        delta = cand.min(axis=1) + neg[t]
    path = np.empty((G, T), dtype=np.int8)
    s = delta.argmin(axis=1)
    g = np.arange(G)
    for t in range(T - 1, -1, -1):
        path[:, t] = s
        s = bp[t, g, s]
    return path


def cost_arrays():
    _, _, _, _, c_in, c_out = s2.STAGE1
    costs = np.array([s1.cost_matrix(ct, c_in, c_out) for ct, _ in cost_grid()])
    disc = np.array([d for _, d in cost_grid()])
    return costs, disc


def run_grid():
    costs, disc = cost_arrays()
    rows = []
    for k, mt in enumerate(MATCHES):
        t0 = time.time()
        m = bench.load(mt + "~")
        c = Cues(m, k)
        for a, wz, wt, w1 in evidence_grid():
            e = c.evidence(a, wz, wt, w1)
            paths = viterbi_touch(e, costs, disc, c.near(a, w1, len(e)))
            for cg, path in zip(cost_grid(), paths):
                pred = bench.finish(m, s1.upsample(path, c.n))
                rows.append(dict(match=mt, param=(a, wz, wt, w1) + cg, **bench.score(m["labels"], pred)))
        print("  {} done in {:.0f} s".format(mt, time.time() - t0), flush=True)
    return pd.DataFrame(rows)


def predict(m, param, k=0, cues=None):
    a, wz, wt, w1, ct, d = param
    c = cues or Cues(m, k)
    _, _, _, _, c_in, c_out = s2.STAGE1
    e = c.evidence(a, wz, wt, w1)
    path = viterbi_touch(e, s1.cost_matrix(ct, c_in, c_out)[None], np.array([d]), c.near(a, w1, len(e)))[0]
    return bench.finish(m, s1.upsample(path, c.n)), c


def seen_scores(m, pred):
    """Frame scores split by ball seen / missing (> 1 s): H/A accuracy where play is on, macroF1."""
    lab, sc = m["labels"], (m["half"] > 0) & (m["labels"] != bench.X)
    out = {}
    for name, w in (("seen", sc & (m["ball_state"] == 0)), ("missing", sc & (m["ball_state"] == 2))):
        g, p = lab[w], pred[w]
        alive = g < D
        out[name + "_HA_acc"] = (p[alive] == g[alive]).mean()
        f1 = []
        for c in (H, A, D):
            tp = ((p == c) & (g == c)).sum()
            prec, rec = tp / max((p == c).sum(), 1), tp / max((g == c).sum(), 1)
            f1.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        out[name + "_macroF1"] = np.mean(f1)
    return out


def show(title, df, cols=COLS):
    print("\n== {} ==".format(title))
    print(bench.summary(df, cols=cols).to_string(index=False))


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 3)
    t0 = time.time()
    print("stage 2 setting under stage 3:", STAGE2, "stage 1:", s2.STAGE1)
    path = os.path.join(CACHE, "v2_stage3_grid.csv")
    if os.path.isfile(path):
        g = pd.read_csv(path)
        g["param"] = g["param"].map(eval)
    else:
        g = run_grid()
        g.to_csv(path, index=False)

    sk = pd.read_csv(os.path.join(CACHE, "v2_stage2_grid.csv"))
    sk["param"] = sk["param"].map(eval)
    sk = bench.lomo(sk[sk["mode"] == "full"])
    mj = pd.read_csv(os.path.join(CACHE, "v2_stage2m_grid.csv"))
    mj["param"] = mj["param"].map(eval)
    mj = bench.lomo(mj[mj["mode"] == "full"])
    lo = bench.lomo(g)
    # seen / missing frame scores for the three versions, at each match's LOMO setting
    for name, df, fn in (("stage 2 on skeleton", sk, lambda m, p, k: s2.predict(m, p, "full", k)[0]),
                         ("stage 2 on majority", mj, lambda m, p, k: s2m.predict(m, p, "full", k)[0]),
                         ("stage 3", lo, lambda m, p, k: predict(m, p, k)[0])):
        extra = []
        for k, mt in enumerate(MATCHES):
            m = bench.load(mt + "~")
            extra.append(seen_scores(m, fn(m, df[df.match == mt].param.iloc[0], k)))
        for c in SEEN_COLS:
            df[c] = [x[c] for x in extra]
        show(name + ", LOMO", df, COLS + SEEN_COLS)
    print("stage 3 params (a_touch, w_zone, w_touch, w1, c_team, discount):", lo.param.tolist())

    def gate(new, old):
        return new.spellF1.mean() > old.spellF1.mean() and new.D_F1.mean() >= old.D_F1.mean()

    for name, other in (("stage 2 on skeleton (gate)", sk), ("stage 2 on majority", mj)):
        print("stage 3 vs {}: spellF1 {:.4f} vs {:.4f}, D F1 {:.3f} vs {:.3f}, macroF1 {:.4f} vs {:.4f}, "
              "seen H/A acc {:.3f} vs {:.3f} -> {}".format(
                  name, lo.spellF1.mean(), other.spellF1.mean(), lo.D_F1.mean(), other.D_F1.mean(),
                  lo.macroF1.mean(), other.macroF1.mean(), lo.seen_HA_acc.mean(), other.seen_HA_acc.mean(),
                  "PASS" if gate(lo, other) else "FAIL"))
    passed = gate(lo, sk)

    # what each piece adds (LOMO with the others free): zone, touches, two-touch rule, discount
    for label, keep in (("no zone", lambda p: p[1] == 0), ("no touches", lambda p: p[2] == 0),
                        ("rule off (w1 = 1)", lambda p: p[3] == 1.0), ("rule on (w1 = 0)", lambda p: p[3] == 0.0),
                        ("no discount", lambda p: p[5] == 1.0)):
        sub = bench.lomo(g[g.param.map(keep)])
        print("  {:18s} spellF1 {:.4f}  D F1 {:.3f}".format(label, sub.spellF1.mean(), sub.D_F1.mean()))

    final = bench.all_params_best(g)
    print("\nkept: {}; final setting (best on all 7 degraded): {}".format(
        "stage 3" if passed else "stage 2 on skeleton (gate failed)", final))

    clean = []
    for k, mt in enumerate(MATCHES):
        m = bench.load(mt)
        pred, _ = predict(m, final, k)
        clean.append(dict(match=mt, **bench.score(m["labels"], pred), **seen_scores(m, pred)))
    show("same setting on clean Sportec (reference)", pd.DataFrame(clean), COLS + ("seen_HA_acc",))

    ours = bench.load("ours")
    if passed:
        pred, c = predict(ours, final)
        f, team, duel, conf = c.touch[final[0]]
        minutes = (ours["half"] > 0).sum() / FPS / 60
        print("ours: touches {:.1f} per min (H {:.0%}), confirmed {:.0%}, duels {:.0%}".format(
            len(f) / minutes, (team == 0).mean(), conf.mean(), duel.mean()))
    else:
        pred = s2.predict(ours, STAGE2, "full")[0]
    bench.report_ours(pred, "v2_stage3")
    print("stage 3 took {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
