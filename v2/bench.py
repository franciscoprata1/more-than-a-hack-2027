"""Stage 0: one test bench for both datasets: load a match, score labels, leave one match out.

    .venv\\Scripts\\python.exe v2/bench.py            # label check + baselines on clean and degraded Sportec

Used as a module by every later stage:
    load(name)                 "J03WPY" clean Sportec, "J03WPY~" degraded, "ours" our match
    score(gt, pred)            spellF1, macroF1, D precision and recall, possession counts
    lomo(table)                leave one match out: pick the grid row that is best on the other six,
                               report it on the held-out match
    report_ours(pred, name)    our match: automatic checks (totals), writes the submission, and the
                               5 minute answer key score as a report only

A match dict has ball_x, ball_y, ball_state (0 seen, 1 filled <= 1 s, 2 missing), ball_vx/vy/speed,
raw_ball_x/y (unsmoothed), player_x/y/vx/vy (frames, slots), team (per slot), half (0 = not scored),
pitch, labels (Sportec only), events (Sportec only), tags (ours only). Smoothing is v1/smooth.py's,
run once per match and kept in <folder>/smooth.npz.

Our match: only totals and scores are ever printed. Its labels are never loaded except the public
5 minutes in report_ours, and that score never chooses anything.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "v1"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from smooth import (fill_ball, fill_track, smooth_and_velocity, runs, BALL_SIGMA,  # noqa: E402
                    PLAYER_SIGMA, MISSING)
from sportec_download import MATCHES  # noqa: E402

# Params
FPS = 25
CACHE = os.path.join(ROOT, "cache")
OUR_PITCH = (105.3, 68.0)
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
RESULTS = os.path.join(ROOT, "evaluation", "results")
H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(list("HADX"))


def folder(name):
    if name == "ours":
        return CACHE
    if name.endswith("~"):
        return os.path.join(CACHE, "sportec_degraded", name[:-1])
    return os.path.join(CACHE, "sportec", name)


def smooth_match(path, ball, players, scored, out):
    """v1 smoothing for one match, saved to out (same keys as cache/smooth.npz, the ones we use)."""
    bx, by, state, gap = fill_ball(ball["x"], ball["y"], scored)
    bx, by, bvx, bvy = smooth_and_velocity(bx, by, runs(state != MISSING), BALL_SIGMA)
    n, k = players["x"].shape
    px = np.full((n, k), np.nan, dtype=np.float32)
    py, pvx, pvy = px.copy(), px.copy(), px.copy()
    for j in range(k):
        fx, fy, _ = fill_track(players["x"][:, j], players["y"][:, j], scored)
        sx, sy, vx, vy = smooth_and_velocity(fx, fy, runs(~np.isnan(fx)), PLAYER_SIGMA)
        px[:, j], py[:, j], pvx[:, j], pvy[:, j] = sx, sy, vx, vy
    np.savez_compressed(out, ball_x=bx.astype(np.float32), ball_y=by.astype(np.float32),
                        ball_vx=bvx.astype(np.float32), ball_vy=bvy.astype(np.float32),
                        ball_speed=np.hypot(bvx, bvy).astype(np.float32), ball_state=state,
                        ball_gap=gap, player_x=px, player_y=py, player_vx=pvx, player_vy=pvy)


def load(name):
    """One match as a dict (see module docstring)."""
    d = folder(name)
    ball = np.load(os.path.join(d, "ball.npz"))
    players = np.load(os.path.join(d, "players.npz"))
    if name == "ours":
        scored = np.load(os.path.join(d, "scored.npy"))
        half = np.zeros(len(scored), dtype=np.int8)
        edges = runs(scored)
        for h, (s, e) in enumerate(edges, start=1):
            half[s:e] = h
        pitch = OUR_PITCH
        sm_path = os.path.join(d, "smooth.npz")
    else:
        half = np.load(os.path.join(d, "half.npy"))
        scored = half > 0
        pitch = tuple(np.load(os.path.join(d, "pitch.npy")))
        sm_path = os.path.join(d, "smooth.npz")
        if not os.path.isfile(sm_path):
            smooth_match(d, ball, players, scored, sm_path)
    sm = np.load(sm_path)
    m = {"name": name, "half": half, "pitch": pitch, "team": players["team"],
         "raw_ball_x": ball["x"], "raw_ball_y": ball["y"],
         "raw_player_x": players["x"], "raw_player_y": players["y"]}
    for k in ("ball_x", "ball_y", "ball_vx", "ball_vy", "ball_speed", "ball_state", "ball_gap",
              "player_x", "player_y", "player_vx", "player_vy"):
        m[k] = sm[k]
    if name == "ours":
        m["tags"] = pd.read_csv(os.path.join(d, "tags.csv"), keep_default_na=False)
    else:
        m["labels"] = np.load(os.path.join(d, "labels.npy"))
        m["events"] = pd.read_csv(os.path.join(d, "events.csv"), keep_default_na=False)
    return m


def nearest_by_team(bx, by, px, py, team):
    """Distance from the ball to the nearest home and nearest away player (NaN if none)."""
    d = np.hypot(px - bx[:, None], py - by[:, None])
    out = []
    for t in (0, 1):
        dt = d[:, team == t]
        ok = ~np.isnan(dt).all(axis=1)
        v = np.full(len(bx), np.nan)
        v[ok] = np.nanmin(dt[ok], axis=1)
        out.append(v)
    return out


def score(gt, pred):
    """The numbers every stage reports."""
    r = pm.evaluate(gt, pred, taus=(25,))
    pc = r["per_class"]["D"]
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"], "D_prec": pc["precision"],
            "D_rec": pc["recall"], "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"]}


def lomo(table, key="spellF1"):
    """table: DataFrame with columns match, param, and scores. For each match, the param best on the
    mean of the other matches, scored on that match. Returns (per-match DataFrame, chosen params)."""
    rows = []
    for m in table["match"].unique():
        other = table[table["match"] != m].groupby("param")[key].mean()
        best = other.idxmax()
        row = table[(table["match"] == m) & (table["param"] == best)].iloc[0].to_dict()
        rows.append(row)
    return pd.DataFrame(rows)


def summary(df, cols=("spellF1", "macroF1", "D_prec", "D_rec", "n_pred", "n_gt")):
    """Per-match rows plus a mean row, as a printable DataFrame."""
    out = df[["match"] + [c for c in cols if c in df]].copy()
    mean = out.drop(columns="match").mean()
    mean["match"] = "mean"
    return pd.concat([out, mean.to_frame().T], ignore_index=True)


def all_params_best(table, key="spellF1"):
    """The single param best on the mean of all matches (used for the final run on our match)."""
    return table.groupby("param")[key].mean().idxmax()


def report_ours(pred, name):
    """Our match: write the submission, print totals and automatic checks, then the 5 minute score
    as a report only. pred is codes over all 199,319 frames."""
    m_scored = np.load(os.path.join(CACHE, "scored.npy"))
    path = os.path.join(RESULTS, "{}.csv".format(name))
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, c) for i, c in enumerate(LETTERS[pred]))
    runs_ = [(s, e) for s, e in runs(m_scored)]
    filled = pm.fill_abstentions(pred, runs_)
    spells = pm.possession_spells(filled, runs_)
    lens = np.array([e - s for s, e, _ in spells]) / FPS
    p = pred[m_scored]
    print("ours {}: H {:.1%} A {:.1%} D {:.1%} X {:.1%} | possessions {} mean {:.1f} s median {:.1f} s".format(
        name, *[(p == k).mean() for k in (H, A, D, X)], len(spells), lens.mean(), np.median(lens)))
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    s = score(gt, pred)
    print("ours {}: 5 minute answer key (report only) spellF1 {:.4f} macroF1 {:.4f} D prec {:.3f} rec {:.3f}".format(
        name, s["spellF1"], s["macroF1"], s["D_prec"], s["D_rec"]))
    return s


def majority(codes, window):
    """The organisers' baseline smoothing (starter/baseline.py), as codes: D where nobody voted."""
    hc = np.concatenate(([0], np.cumsum(codes == H)))
    ac = np.concatenate(([0], np.cumsum(codes == A)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    out = np.full(len(codes), D, dtype=np.int8)
    seen = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], H, A)
    return out


def baseline_codes(m):
    """Nearest player to the raw ball, raw players, nobody (-1) where the ball is missing."""
    dh, da = nearest_by_team(m["raw_ball_x"].astype(np.float64), m["raw_ball_y"].astype(np.float64),
                             m["raw_player_x"], m["raw_player_y"], m["team"])
    c = np.full(len(dh), -1, dtype=np.int8)
    ok = ~np.isnan(dh) | ~np.isnan(da)
    c[ok] = np.where(np.nan_to_num(dh[ok], nan=1e9) <= np.nan_to_num(da[ok], nan=1e9), H, A)
    return c


def finish(m, pred):
    """Unscored frames get X in Sportec (the separator); D for ours (as the baseline does)."""
    pred = pred.copy()
    pred[m["half"] == 0] = X if m["name"] != "ours" else D
    return pred


BASE_WINDOWS = (0, 25, 50, 75, 125, 250)


def baseline_table(degraded=True):
    rows = []
    for mt in MATCHES:
        m = load(mt + ("~" if degraded else ""))
        c = baseline_codes(m)
        for w in BASE_WINDOWS:
            s = score(m["labels"], finish(m, majority(c, w)))
            rows.append(dict(match=mt, param=w, **s))
    return pd.DataFrame(rows)


def label_check():
    """Gate: do the Sportec labels look like football?"""
    rows = []
    for mt in MATCHES:
        lab = np.load(os.path.join(folder(mt), "labels.npy"))
        r = pm.valid_runs(lab)
        sp = pm.possession_spells(lab, r)
        L = np.array([e - s for s, e, _ in sp]) / FPS
        dead = [e - s for s, e, c in pm.segments(lab, r) if c == D]
        sc = lab != X
        rows.append(dict(match=mt, dead_share=(lab[sc] == D).mean(), possessions=len(sp),
                         mean_s=L.mean(), median_s=np.median(L), under_1s=(L < 1).mean(),
                         stoppages=len(dead), stoppage_mean_s=np.mean(dead) / FPS))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)
    print("== label check (gate) ==")
    print(label_check().to_string(index=False))
    for degraded in (False, True):
        t = baseline_table(degraded)
        tag = "degraded" if degraded else "clean"
        t.to_csv(os.path.join(CACHE, "v2_baseline_{}.csv".format(tag)), index=False)
        print("\n== majority smoothing on {} Sportec: window +/-50 (organisers' smooth50) ==".format(tag))
        print(summary(t[t.param == 50]).to_string(index=False))
        print("mean spellF1 by window:", t.groupby("param")["spellF1"].mean().round(4).to_dict())
        print("== majority smoothing, window chosen leave one match out ==")
        lo = lomo(t)
        print(summary(lo).to_string(index=False), " windows:", lo.param.tolist())
