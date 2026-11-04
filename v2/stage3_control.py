"""Stage 3, control version: possession changes only when a team shows it controls the ball.

    .venv\\Scripts\\python.exe v2/stage3_control.py

Francisco's football rules on top of the owner version (v2/stage3_owner.py, its owner rule fixed at
radius 1.5 m, margin 0):
  - control: an owner spell is control when the player has the ball clean (no opponent within
    DUEL_M) for at least CONTROL_S, or when the next owner, within PASS_S, is a teammate (a
    successful pass). A spell that is neither (a duel, a block) is not control;
  - duels decide nothing: an owner spell without control gets W_UNC of the owner weight (0 = none)
    and no cheaper switch, so H -> A (duel) -> H stays H throughout;
  - ball in flight (seen, no owner, faster than FLIGHT_SPEED): the nearest team clue of Stage 1 is
    switched off (MUTE), and possession leans W_FLIGHT nats per frame to the last team in control,
    only while the ball travels towards that team's attacking goal if FORWARD_ONLY (a clearance or
    deflection is settled by who controls it next);
  - turnovers: only a controlled spell by the other team counts; switching possession to it costs
    DISCOUNT x C_TEAM within 0.4 s of its start.
Everything else is Stage 2 on the skeleton, fixed. Every new piece has an off value in the grid, so
the bench shows which ones help.

Tuned on degraded Sportec, LOMO. Gate (Stage 3 rule): spellF1 up and D F1 not down against the owner
version; compared with Stage 2 on the skeleton and on majority smoothing.
Writes cache/v2_stage3c_grid.csv, evaluation/results/v2_stage3_control.csv.
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
import owner as ow  # noqa: E402
import stage1 as s1  # noqa: E402
import stage2 as s2  # noqa: E402
import stage2_majority as s2m  # noqa: E402
import stage3 as s3  # noqa: E402
import stage3_owner as s3o  # noqa: E402
from bench import H, A, MATCHES, CACHE  # noqa: E402

# Params
FPS = 25
RADIUS, MARGIN, W_OWN = 1.5, 0.0, 1.0      # the owner version's kept setting
DUEL_M = 1.5             # an opponent this close to the ball: contested, not clean
PASS_S = 3.0             # a teammate becomes the owner within this: a successful pass
FLIGHT_SPEED = 4.0       # m/s: faster than the ball at a walking player's feet
TURN_WIN = 2             # steps either side of a controlled turnover
CONTROL_SS = (0.2, 0.4, 1.0)       # s of clean ball that make control (0.2 added: first grid picked its edge)
W_UNCS = (0.0, 1.0)                # share of W_OWN for a spell without control (1 = owner version)
MUTES = (False, True)              # switch off the nearest-team clue while the ball flies
W_FLIGHTS = (0.0, 0.3, 1.0, 3.0)   # nats per frame to the last team in control, ball in flight (3 added: edge)
FORWARD_ONLYS = (False, True)
C_TEAMS = (4, 8, 16, 32)           # 4 added: first grid picked its edge
DISCOUNTS = (1.0, 0.5, 0.25, 0.1)  # 0.1 added: first grid picked its edge


def evidence_grid():
    g = itertools.product(CONTROL_SS, W_UNCS, MUTES, W_FLIGHTS, FORWARD_ONLYS)
    return [p for p in g if not (p[3] == 0 and p[4])]   # direction means nothing without a lean


def cost_grid():
    return list(itertools.product(C_TEAMS, DISCOUNTS))


def attack_sign(m):
    """+1 if the team attacks towards high x in that half, -1 if low x: per (half, team)."""
    sides = db.own_sides(m)
    return {k: (1 if v == 0 else -1) for k, v in sides.items()}


class Cues:
    def __init__(self, m, k):
        self.n = n = len(m["half"])
        self.half = m["half"]
        self.e2 = s3.stage2_evidence(m, k)
        d = ow.distances_all(m)
        team = m["team"]
        own = ow.owner(d, team, RADIUS, MARGIN)
        self.eps, _ = ow.episodes(own, team, m["half"])
        inf = np.where(np.isnan(d), np.inf, d)
        self.opp_near = np.zeros(n, dtype=bool)                  # opponent of the owner within DUEL_M
        for t in (0, 1):
            close = (inf[:, team == t] < DUEL_M).any(axis=1)
            self.opp_near |= close & (own >= 0) & (team[np.maximum(own, 0)] != t)
        seen = (m["ball_state"] == 0) & (m["half"] > 0)
        self.flight = seen & (own < 0) & (m["ball_speed"] > FLIGHT_SPEED)
        self.vx = np.nan_to_num(m["ball_vx"].astype(np.float64))
        self.sign = attack_sign(m)
        self.cache = {}

    def control(self, control_s):
        """Per owner spell: controlled or not."""
        out = []
        for i, (a, b, slot, t) in enumerate(self.eps):
            clean = (~self.opp_near[a:b]).sum() >= control_s * FPS
            nxt = self.eps[i + 1] if i + 1 < len(self.eps) else None
            passed = (nxt is not None and nxt[3] == t and nxt[2] != slot and nxt[0] - b <= PASS_S * FPS
                      and (self.half[b:nxt[0]] > 0).all())
            out.append(clean or passed)
        return out

    def build(self, control_s, w_unc, mute, w_flight, forward):
        e = self.e2.copy()
        ctl = self.control(control_s)
        last = np.full(self.n, -1, dtype=np.int8)             # last team in control, carried forward
        turns = []
        prev = -1
        for (a, b, _, t), c in zip(self.eps, ctl):
            w = W_OWN if c else W_OWN * w_unc
            e[a:b, t] += w
            if c:
                if prev >= 0 and t != prev:
                    turns.append((a, t))
                prev = t
                last[a] = t
        # carry the last team in control forward, reset where the half changes
        cur = -1
        idx = np.flatnonzero(last >= 0)
        starts = set(idx.tolist())
        lt = np.full(self.n, -1, dtype=np.int8)
        for i in range(self.n):
            if self.half[i] == 0:
                cur = -1
            elif i in starts:
                cur = last[i]
            lt[i] = cur
        fl = self.flight
        if mute:
            tot = np.logaddexp(e[fl, H], e[fl, A]) - np.log(2)
            e[fl, H], e[fl, A] = tot, tot
        if w_flight > 0:
            for t in (0, 1):
                mask = fl & (lt == t)
                if forward:
                    sgn = np.array([self.sign.get((h, t), 0) for h in (0, 1, 2)])[self.half]
                    mask &= self.vx * sgn > 0
                e[mask, t] += w_flight
        return s1.to_steps(e), turns

    def near(self, turns, n_steps):
        out = np.zeros((n_steps + 2 * TURN_WIN + 1, 2), dtype=bool)
        for f, t in turns:
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
        for p in evidence_grid():
            e, turns = c.build(*p)
            paths = s3.viterbi_touch(e, costs, disc, c.near(turns, len(e)))
            for cg, path in zip(cost_grid(), paths):
                pred = bench.finish(m, s1.upsample(path, c.n))
                rows.append(dict(match=mt, param=p + cg, **bench.score(m["labels"], pred)))
        print("  {} done in {:.0f} s".format(mt, time.time() - t0), flush=True)
    return pd.DataFrame(rows)


def predict(m, param, k=0):
    *p, ct, d = param
    c = Cues(m, k)
    _, _, _, _, c_in, c_out = s2.STAGE1
    e, turns = c.build(*p)
    path = s3.viterbi_touch(e, s1.cost_matrix(ct, c_in, c_out)[None], np.array([d]), c.near(turns, len(e)))[0]
    return bench.finish(m, s1.upsample(path, c.n)), c, turns


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 3)
    t0 = time.time()
    path = os.path.join(CACHE, "v2_stage3c_grid.csv")
    if os.path.isfile(path):
        g = pd.read_csv(path)
        g["param"] = g["param"].map(eval)
    else:
        g = run_grid()
        g.to_csv(path, index=False)

    versions = {
        "stage 2 on skeleton": (s3o.lomo_of("v2_stage2_grid.csv", "full"), lambda m, p, k: s2.predict(m, p, "full", k)[0]),
        "stage 2 on majority": (s3o.lomo_of("v2_stage2m_grid.csv", "full"), lambda m, p, k: s2m.predict(m, p, "full", k)[0]),
        "stage 3 owner": (s3o.lomo_of("v2_stage3o_grid.csv"), lambda m, p, k: s3o.predict(m, p, k)[0]),
        "stage 3 control": (bench.lomo(g), lambda m, p, k: predict(m, p, k)[0]),
    }
    for name, (df, fn) in versions.items():
        extra = []
        for k, mt in enumerate(MATCHES):
            m = bench.load(mt + "~")
            extra.append(s3.seen_scores(m, fn(m, df[df.match == mt].param.iloc[0], k)))
        for col in s3.SEEN_COLS:
            df[col] = [x[col] for x in extra]
        s3.show(name + ", LOMO", df, s3.COLS + s3.SEEN_COLS)
    lo = versions["stage 3 control"][0]
    print("control params (control_s, w_unc, mute, w_flight, forward, c_team, discount):", lo.param.tolist())

    def gate(new, old):
        return new.spellF1.mean() > old.spellF1.mean() and new.D_F1.mean() >= old.D_F1.mean()

    for name in ("stage 3 owner", "stage 2 on skeleton", "stage 2 on majority"):
        other = versions[name][0]
        print("control vs {}: spellF1 {:.4f} vs {:.4f}, D F1 {:.3f} vs {:.3f}, macroF1 {:.4f} vs {:.4f}, seen H/A acc "
              "{:.3f} vs {:.3f} -> {}{}".format(
                  name, lo.spellF1.mean(), other.spellF1.mean(), lo.D_F1.mean(), other.D_F1.mean(),
                  lo.macroF1.mean(), other.macroF1.mean(), lo.seen_HA_acc.mean(), other.seen_HA_acc.mean(),
                  "PASS" if gate(lo, other) else "FAIL", " (the gate)" if name == "stage 3 owner" else ""))
    passed = gate(lo, versions["stage 3 owner"][0])

    # each idea alone switched off (LOMO with the rest free)
    for label, keep in (("duels decide nothing OFF (w_unc 1)", lambda p: p[1] == 1.0),
                        ("duels decide nothing ON (w_unc 0)", lambda p: p[1] == 0.0),
                        ("nearest team in flight kept", lambda p: not p[2]),
                        ("nearest team in flight muted", lambda p: p[2]),
                        ("no flight lean", lambda p: p[3] == 0),
                        ("flight lean, any direction", lambda p: p[3] > 0 and not p[4]),
                        ("flight lean, forward only", lambda p: p[3] > 0 and p[4]),
                        ("no cheaper switch", lambda p: p[6] == 1.0)):
        sub = bench.lomo(g[g.param.map(keep)])
        print("  {:36s} spellF1 {:.4f}  D F1 {:.3f}".format(label, sub.spellF1.mean(), sub.D_F1.mean()))

    final = bench.all_params_best(g)
    print("\nkept: {}; final setting (best on all 7 degraded): {}".format(
        "stage 3 control" if passed else "stage 3 owner (gate failed)", final))
    clean = []
    for k, mt in enumerate(MATCHES):
        m = bench.load(mt)
        pred, _, _ = predict(m, final, k)
        clean.append(dict(match=mt, **bench.score(m["labels"], pred), **s3.seen_scores(m, pred)))
    s3.show("same setting on clean Sportec (reference)", pd.DataFrame(clean), s3.COLS + ("seen_HA_acc",))

    ours = bench.load("ours")
    pred, c, turns = predict(ours, final)
    ctl = c.control(final[0])
    minutes = (ours["half"] > 0).sum() / FPS / 60
    print("ours: owner spells {:.1f} per min, controlled {:.0%}, controlled turnovers {:.1f} per min, ball in flight "
          "{:.1%} of seen frames".format(len(c.eps) / minutes, np.mean(ctl), len(turns) / minutes,
                                         c.flight[(ours["half"] > 0) & (ours["ball_state"] == 0)].mean()))
    for k, mt in enumerate(MATCHES[:1]):
        m = bench.load(mt + "~")
        cm = Cues(m, k)
        print("bench {}: owner spells {:.1f} per min, controlled {:.0%}, ball in flight {:.1%} of seen frames".format(
            mt, len(cm.eps) / ((m["half"] > 0).sum() / FPS / 60), np.mean(cm.control(final[0])),
            cm.flight[(m["half"] > 0) & (m["ball_state"] == 0)].mean()))
    bench.report_ours(pred, "v2_stage3_control")
    print("took {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
