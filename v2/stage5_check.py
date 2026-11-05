"""Stage 5 transfer check: which clues the network leans on, and do they look the same on our match?

    .venv\\Scripts\\python.exe v2/stage5_check.py

Needs cache/v2_stage5/<match>.npz (v2/stage5.py).
  1. Importance: for each left-out match, the network trained on the other 6; each clue (with its
     context columns) is shuffled within that match and the rise in log-loss per step is measured.
  2. For the clues that matter most: their mean on the bench and on ours, on seen-ball and on
     missing-ball steps. A clue far off on ours is one the network may misread there.
  3. What the network says: H / A / D shares and mean confidence, bench (left-out) vs ours.
Prints totals only.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import stage5 as s5  # noqa: E402
from bench import H, A, D, X, MATCHES  # noqa: E402

# Params
TOP = 12
SHUFFLE_SEED = 2027


def base_names():
    """The clue names, in column order (not stored in the feature cache)."""
    m = bench.load(MATCHES[0] + "~")
    return s5.base_features(m, 0, MATCHES[0])[1]


def column_groups(names):
    """Base clue name -> its column indices in the full feature matrix."""
    nb = len(names)
    groups = {nm: [j] for j, nm in enumerate(names)}
    col = nb
    for _ in s5.CONTEXT_S:
        for nm in s5.CONTEXT_OF:
            groups[nm].append(col)
            col += 1
    for _ in (0, 1):
        for nm in s5.PAST_OF:
            groups[nm].append(col)
            col += 1
    return groups, col


def log_loss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-6, 1)).mean()


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)
    feats = {mt: s5.features(mt, k) for k, mt in enumerate(MATCHES)}
    feats["ours"] = s5.features("ours", 0)
    names = base_names()
    groups, ncol = column_groups(names)
    assert ncol == feats["ours"]["X"].shape[1], (ncol, feats["ours"]["X"].shape)
    rng = np.random.default_rng(SHUFFLE_SEED)

    rise = {nm: [] for nm in names}
    shares = []
    for mt in MATCHES:
        clf, _ = s5.train(feats, [nm for nm in MATCHES if nm != mt])
        f = feats[mt]
        ok = (f["half"] > 0) & (f["y"] != X)
        x, y = f["X"][ok], f["y"][ok]
        p = clf.predict_proba(x)
        base = log_loss(p, y)
        shares.append([(p.argmax(1) == c).mean() for c in (H, A, D)] + [p.max(1).mean()])
        for nm, cols in groups.items():
            xs = x.copy()
            perm = rng.permutation(len(xs))
            xs[:, cols] = x[perm][:, cols]
            rise[nm].append(log_loss(clf.predict_proba(xs), y) - base)
        print("  {} done (log-loss {:.3f})".format(mt, base), flush=True)
    imp = pd.Series({nm: np.mean(v) for nm, v in rise.items()}).sort_values(ascending=False)
    print("\nimportance: rise in log-loss when the clue is shuffled (mean over the 7 left-out matches)")
    print(imp.head(TOP).to_string())
    print("  ... the other {} clues: {:.3f} in all".format(len(imp) - TOP, imp.iloc[TOP:].sum()))

    # the top clues: bench vs ours, seen and missing ball
    seen_j, miss_j = names.index("seen"), names.index("missing")
    rows = []
    for nm in imp.index[:TOP]:
        j = names.index(nm)
        row = {"clue": nm}
        for label, key in (("seen", seen_j), ("missing", miss_j)):
            b = np.concatenate([feats[mt]["X"][(feats[mt]["half"] > 0) & (feats[mt]["X"][:, key] > 0.5), j] for mt in MATCHES])
            o = feats["ours"]["X"][(feats["ours"]["half"] > 0) & (feats["ours"]["X"][:, key] > 0.5), j]
            row["bench_" + label], row["ours_" + label] = b.mean(), o.mean()
            row["gap_" + label] = (o.mean() - b.mean()) / (b.std() + 1e-9)
        rows.append(row)
    print("\ntop clues, mean on scored steps (gap = ours minus bench, in bench standard deviations)")
    print(pd.DataFrame(rows).to_string(index=False))
    ob = feats["ours"]
    print("\nsteps with the ball seen / missing: bench {:.1%} / {:.1%}, ours {:.1%} / {:.1%}".format(
        np.mean([(feats[mt]["X"][feats[mt]["half"] > 0, seen_j] > 0.5).mean() for mt in MATCHES]),
        np.mean([(feats[mt]["X"][feats[mt]["half"] > 0, miss_j] > 0.5).mean() for mt in MATCHES]),
        (ob["X"][ob["half"] > 0, seen_j] > 0.5).mean(), (ob["X"][ob["half"] > 0, miss_j] > 0.5).mean()))

    clf, _ = s5.train(feats, list(MATCHES))
    p = clf.predict_proba(ob["X"][ob["half"] > 0])
    sb = np.mean(shares, axis=0)
    print("\nnetwork's own call per step, before decoding: H / A / D share, mean confidence")
    print("  bench (left-out) {:.1%} / {:.1%} / {:.1%}, {:.2f}".format(*sb))
    print("  ours             {:.1%} / {:.1%} / {:.1%}, {:.2f}".format(
        *[(p.argmax(1) == c).mean() for c in (H, A, D)], p.max(1).mean()))


if __name__ == "__main__":
    main()
