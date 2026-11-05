"""Part 2, transfer test: does the bench-trained "nobody has the ball" share read our match right?

    .venv\\Scripts\\python.exe v2/gap_transfer.py

Needs cache/v2_gap/model_fold<k>.pt (v2/gap_train.py) and cache/hidden_gaps.npz (v1/hidden_gaps.py).
Stage 4 leans to D where p_nobody is high. On ours, 91% of the deep gap frames come out above 0.3,
far more than on the bench. Either our gaps really are mostly dead ball, or the models read our
match as "nobody" whatever happens. This test tells the two apart.

Ours: the ball is hidden in v1's open-play test gaps (2, 5, 10, 20 s, picked by code, live play by
construction), one 30 s window centred on each gap, and the 7 models are averaged as Stage 4 does.
Bench control: the same on each left-out match with the model that never saw it, on stretches picked
by code (seed 2027) where the ball is seen throughout and the label is live (H/A) throughout, and on
stretches that are dead (D) throughout. If the models transfer, ours should look like the bench's live.

Each also run with the ball shown, to see how much hiding alone moves p_nobody.
Prints totals only.
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bench  # noqa: E402
import gap_dataset as gd  # noqa: E402
import gap_train as gt  # noqa: E402
from bench import MATCHES, D  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "v1"))
import ball_dataset as bd  # noqa: E402
from ball_model import BallModel  # noqa: E402
from hidden_gaps import SEED  # noqa: E402

# Params
FPS = 25
LENGTHS = (2, 5, 10, 20)       # seconds, as v1's hidden gaps
BENCH_PER_LENGTH = 40          # stretches per match, per length, per kind
P_MID = 0.3                    # Stage 4's final setting
HIDDEN_GAPS = os.path.join(ROOT, "cache", "hidden_gaps.npz")


def load_model(k):
    model = BallModel(on_line=False)
    model.load_state_dict(torch.load(os.path.join(gd.OUT_DIR, "model_fold{}.pt".format(k))))
    model.eval()
    return model


def window(ds, a, b):
    """Start of the 30 s window centred on steps a..b, kept inside the half."""
    idx = np.flatnonzero(ds["half"] == ds["half"][a])
    s = (a + b) // 2 - bd.WINDOW // 2
    return int(np.clip(s, idx[0], idx[-1] + 1 - bd.WINDOW))


def p_nobody(model, ds, stretches, hide):
    """Mean p_nobody per stretch (steps a..b-1), the ball hidden there or shown."""
    data = {"m": ds}
    out = []
    with torch.no_grad():
        for i in range(0, len(stretches), gt.BATCH):
            chunk = stretches[i:i + gt.BATCH]
            starts, hides = [], []
            for a, b in chunk:
                s = window(ds, a, b)
                h = np.zeros(bd.WINDOW, dtype=bool)
                if hide:
                    h[a - s:b - s] = True
                starts.append(s)
                hides.append(h)
            batch = gt.make_batch(data, [("m", s) for s in starts], hides)
            logits, _ = gt.predict(model, batch)
            p = torch.softmax(logits, dim=-1)[..., -1].numpy()
            out += [p[k, a - s:b - s].mean() for k, ((a, b), s) in enumerate(zip(chunk, starts))]
    return np.array(out)


def our_stretches(ds):
    """v1's hidden gaps as step ranges, with their length in seconds."""
    h = np.load(HIDDEN_GAPS)
    out = []
    for s in LENGTHS:
        for g in h["gap_{}".format(s)]:
            a, b = np.searchsorted(ds["frames"], [g, g + s * FPS])
            out.append((a, b, s))
    return out


def bench_stretches(ds, lab, rng):
    """Stretches with the ball seen throughout: live (H/A) or dead (D) throughout. {kind: [(a, b, s)]}."""
    out = {"live": [], "dead": []}
    for s in LENGTHS:
        n = s * bd.HZ
        known = np.convolve(ds["known"], np.ones(n), "valid") == n
        dead = np.convolve(lab == D, np.ones(n), "valid") == n
        live = np.convolve(lab < D, np.ones(n), "valid") == n
        half = np.convolve(ds["half"], np.ones(n), "valid") == n * ds["half"][:len(known)].astype(int)
        for kind, ok in (("live", known & live & half), ("dead", known & dead & half)):
            cand = np.flatnonzero(ok)
            for a in rng.choice(cand, min(BENCH_PER_LENGTH, len(cand)), replace=False):
                out[kind].append((a, a + n, s))
    return out


def show(label, lens, hid, shw):
    lens = np.array(lens)
    parts = ["  {:30s} hidden: mean {:.2f}, above {} {:4.0%} | shown {:.2f}".format(
        label, hid.mean(), P_MID, (hid > P_MID).mean(), shw.mean())]
    parts.append("    by length (hidden mean): " + ", ".join(
        "{} s {:.2f}".format(s, hid[lens == s].mean()) for s in LENGTHS if (lens == s).any()))
    print("\n".join(parts))


def main():
    torch.set_num_threads(8)
    models = [load_model(k) for k in range(len(MATCHES))]
    rng = np.random.default_rng(SEED)

    print("bench control: each left-out match with the model that never saw it")
    allb = {kind: ([], [], []) for kind in ("live", "dead")}
    for k, mt in enumerate(MATCHES):
        ds = gd.load(mt)
        lab = bench.load(mt + "~")["labels"][ds["frames"]]
        for kind, st in bench_stretches(ds, lab, rng).items():
            ab = [(a, b) for a, b, _ in st]
            hid, shw = p_nobody(models[k], ds, ab, True), p_nobody(models[k], ds, ab, False)
            allb[kind][0].extend(s for _, _, s in st)
            allb[kind][1].extend(hid)
            allb[kind][2].extend(shw)
        print("  {} done".format(mt), flush=True)
    for kind in ("live", "dead"):
        lens, hid, shw = allb[kind]
        show("bench, {} ({} stretches)".format(kind, len(hid)), lens, np.array(hid), np.array(shw))

    ds = gd.load("ours")
    st = our_stretches(ds)
    ab = [(a, b) for a, b, _ in st]
    hid = np.mean([p_nobody(m, ds, ab, True) for m in models], axis=0)
    shw = np.mean([p_nobody(m, ds, ab, False) for m in models], axis=0)
    print("\nours: v1's open-play test gaps, 7 models averaged")
    show("ours, open play ({} gaps)".format(len(st)), [s for _, _, s in st], hid, shw)


if __name__ == "__main__":
    main()
