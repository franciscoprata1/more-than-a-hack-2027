"""Stage 3, owner version: who clearly has the seen ball, on the skeleton. Replaces touches.

    .venv\\Scripts\\python.exe v2/stage3_owner.py

Starts from Stage 2 on the Stage 1 skeleton (fixed) and adds, from v2/owner.py, only where the ball is
really seen:
  - owner: a player has the ball when it is within RADIUS of him and the nearest opponent is at
    least MARGIN further away. W_OWN nats per frame towards his team. A crowded spot has no owner
    and no clue (the possession zone and the touches of the first Stage 3 are dropped);
  - turnovers: the owner's team differs from the previous owner's. Changing possession to a team
    costs DISCOUNT x C_TEAM within TURN_WIN of its turnover, C_TEAM elsewhere; a change of owner
    inside a team is a pass and changes nothing.
Our match passed a transfer check before tuning (v2/owner.py: owner on 37.5% of seen frames against
25-35% on the bench, 4.3 turnovers a minute against 2.4-3.7).

Tuned on degraded Sportec, LOMO. Gate (Stage 3 rule): spellF1 up and D F1 not down against Stage 2 on
the skeleton; compared also with Stage 2 on majority smoothing and the touch version of Stage 3.
Writes cache/v2_stage3o_grid.csv, evaluation/results/v2_stage3_owner.csv.
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
import owner as ow  # noqa: E402
import stage1 as s1  # noqa: E402
import stage2 as s2  # noqa: E402
import stage2_majority as s2m  # noqa: E402
import stage3 as s3  # noqa: E402
from bench import MATCHES, CACHE  # noqa: E402

# Params
FPS = 25
W_OWNS = (0.3, 1.0, 3.0)           # nats per frame
C_TEAMS = (4, 8, 16, 32)
DISCOUNTS = (1.0, 0.5, 0.25, 0.1)   # 0.1 added: the first grid picked its edge, 0.25
TURN_WIN = 2                       # steps (0.2 s each) either side of a turnover


def evidence_grid():
    return list(itertools.product(ow.RADII, ow.MARGINS, W_OWNS))


def cost_grid():
    return list(itertools.product(C_TEAMS, DISCOUNTS))


class Cues:
    def __init__(self, m, k):
        self.n = len(m["half"])
        self.e2 = s3.stage2_evidence(m, k)
        d = ow.distances_all(m)
        self.own = {}
        for r, mg in itertools.product(ow.RADII, ow.MARGINS):
            o = ow.owner(d, m["team"], r, mg)
            eps, turn = ow.episodes(o, m["team"], m["half"])
            frames = np.zeros(self.n, dtype=np.int8) - 1     # owner's team per frame, flicker removed
            for a, b, _, t in eps:
                frames[a:b] = t
            self.own[(r, mg)] = (frames, turn)

    def evidence(self, r, mg, w):
        e = self.e2.copy()
        frames, _ = self.own[(r, mg)]
        for t in (0, 1):
            e[frames == t, t] += w
        return s1.to_steps(e)

    def near(self, r, mg, n_steps):
        _, turn = self.own[(r, mg)]
        out = np.zeros((n_steps + 2 * TURN_WIN + 1, 2), dtype=bool)
        for f, t in turn:
            st = f // s1.STEP
            out[st:st + 2 * TURN_WIN + 1, t] = True
        return out[TURN_WIN:TURN_WIN + n_steps]


def cost_arrays():
    _, _, _, _, c_in, c_out = s2.STAGE1
    return (np.array([s1.cost_matrix(ct, c_in, c_out) for ct, _ in cost_grid()]),
            np.array([d for _, d in cost_grid()]))


def run_grid():
    costs, disc = cost_arrays()
    rows = []
    for k, mt in enumerate(MATCHES):
        t0 = time.time()
        m = bench.load(mt + "~")
        c = Cues(m, k)
        for r, mg, w in evidence_grid():
            e = c.evidence(r, mg, w)
            paths = s3.viterbi_touch(e, costs, disc, c.near(r, mg, len(e)))
            for cg, path in zip(cost_grid(), paths):
                pred = bench.finish(m, s1.upsample(path, c.n))
                rows.append(dict(match=mt, param=(r, mg, w) + cg, **bench.score(m["labels"], pred)))
        print("  {} done in {:.0f} s".format(mt, time.time() - t0), flush=True)
    return pd.DataFrame(rows)


def predict(m, param, k=0):
    r, mg, w, ct, d = param
    c = Cues(m, k)
    _, _, _, _, c_in, c_out = s2.STAGE1
    e = c.evidence(r, mg, w)
    path = s3.viterbi_touch(e, s1.cost_matrix(ct, c_in, c_out)[None], np.array([d]), c.near(r, mg, len(e)))[0]
    return bench.finish(m, s1.upsample(path, c.n)), c


def lomo_of(path, keep=None):
    g = pd.read_csv(os.path.join(CACHE, path))
    g["param"] = g["param"].map(eval)
    return bench.lomo(g if keep is None else g[g["mode"] == keep])


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 3)
    t0 = time.time()
    path = os.path.join(CACHE, "v2_stage3o_grid.csv")
    if os.path.isfile(path):
        g = pd.read_csv(path)
        g["param"] = g["param"].map(eval)
    else:
        g = run_grid()
        g.to_csv(path, index=False)

    versions = {
        "stage 2 on skeleton": (lomo_of("v2_stage2_grid.csv", "full"), lambda m, p, k: s2.predict(m, p, "full", k)[0]),
        "stage 2 on majority": (lomo_of("v2_stage2m_grid.csv", "full"), lambda m, p, k: s2m.predict(m, p, "full", k)[0]),
        "stage 3 touches": (lomo_of("v2_stage3_grid.csv"), lambda m, p, k: s3.predict(m, p, k)[0]),
        "stage 3 owner": (bench.lomo(g), lambda m, p, k: predict(m, p, k)[0]),
    }
    for name, (df, fn) in versions.items():
        extra = []
        for k, mt in enumerate(MATCHES):
            m = bench.load(mt + "~")
            extra.append(s3.seen_scores(m, fn(m, df[df.match == mt].param.iloc[0], k)))
        for col in s3.SEEN_COLS:
            df[col] = [x[col] for x in extra]
        s3.show(name + ", LOMO", df, s3.COLS + s3.SEEN_COLS)
    lo = versions["stage 3 owner"][0]
    print("owner params (radius, margin, w_own, c_team, discount):", lo.param.tolist())

    def gate(new, old):
        return new.spellF1.mean() > old.spellF1.mean() and new.D_F1.mean() >= old.D_F1.mean()

    for name in ("stage 2 on skeleton", "stage 2 on majority", "stage 3 touches"):
        other = versions[name][0]
        print("owner vs {}: spellF1 {:.4f} vs {:.4f}, D F1 {:.3f} vs {:.3f}, macroF1 {:.4f} vs {:.4f}, seen H/A acc "
              "{:.3f} vs {:.3f} -> {}{}".format(
                  name, lo.spellF1.mean(), other.spellF1.mean(), lo.D_F1.mean(), other.D_F1.mean(),
                  lo.macroF1.mean(), other.macroF1.mean(), lo.seen_HA_acc.mean(), other.seen_HA_acc.mean(),
                  "PASS" if gate(lo, other) else "FAIL", " (the gate)" if name == "stage 2 on skeleton" else ""))
    passed = gate(lo, versions["stage 2 on skeleton"][0])
    sub = bench.lomo(g[g.param.map(lambda p: p[4] == 1.0)])
    print("  without cheaper switches at turnovers: spellF1 {:.4f}, D F1 {:.3f}".format(sub.spellF1.mean(), sub.D_F1.mean()))

    final = bench.all_params_best(g)
    print("\nkept: {}; final setting (best on all 7 degraded): {}".format(
        "stage 3 owner" if passed else "stage 2 on skeleton (gate failed)", final))
    clean = []
    for k, mt in enumerate(MATCHES):
        m = bench.load(mt)
        pred, _ = predict(m, final, k)
        clean.append(dict(match=mt, **bench.score(m["labels"], pred), **s3.seen_scores(m, pred)))
    s3.show("same setting on clean Sportec (reference)", pd.DataFrame(clean), s3.COLS + ("seen_HA_acc",))

    ours = bench.load("ours")
    if passed:
        pred, c = predict(ours, final)
        frames, turn = c.own[final[:2]]
        sc = (ours["half"] > 0) & (ours["ball_state"] == 0)
        print("ours: owner on {:.1%} of seen frames, turnovers {:.1f} per min".format(
            (frames[sc] >= 0).mean(), len(turn) / ((ours["half"] > 0).sum() / FPS / 60)))
    else:
        pred = s2.predict(ours, s3.STAGE2, "full")[0]
    bench.report_ours(pred, "v2_stage3_owner")
    print("took {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
