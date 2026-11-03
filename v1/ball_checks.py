"""Phase 3, step 3: look at a trained ball model without opening the hidden gaps.

    .venv\\Scripts\\python.exe v1/ball_checks.py lengths plain          # error per gap length, validation
    .venv\\Scripts\\python.exe v1/ball_checks.py lengths plain line     # two models side by side
    .venv\\Scripts\\python.exe v1/ball_checks.py stoppages line         # stopped play against the rest

A model name is plain (cache/ball_model_fold0_plain.pt), line (cache/ball_model_fold0.pt), or
plain_long, line_long for the same two trained to early stopping.

Everything here uses the validation blocks and the training blocks of fold 0. The hidden gaps are
all open play, so they can't show stopped play, and they stay closed. Error on the training blocks
flatters the model, which has seen those frames: it is reported as a count check, not as a score.

Stopped play is the opposite of open_play in v1/hidden_gaps.py, on steps with the ball tracked: the
players walking, the ball placed or held, the ball over a line, or a stoppage of v1/dead_balls.py.
Prints totals only.
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ball_dataset as bd  # noqa: E402
from ball_model import BallModel  # noqa: E402
from ball_train import BATCH, centre_hidden, hide_stretches, make_batch, predict, set_seed, to_metres  # noqa: E402
from dead_balls import read_tags  # noqa: E402
from hidden_gaps import GAP_SECONDS, SEED, open_play, stoppage_mask  # noqa: E402

# Params
CACHE_DIR = os.path.join(ROOT, "cache")
FOLD = 0
STRIDE_S = 5             # one window every this long, so the same frames aren't counted many times
MODELS = {"plain": ("ball_model_fold0_plain.pt", False), "line": ("ball_model_fold0.pt", True),
          "plain_long": ("ball_model_fold0_plain_long.pt", False), "line_long": ("ball_model_fold0_line_long.pt", True)}


def load_model(name):
    path, on_line = MODELS[name]
    model = BallModel(on_line=on_line)
    model.load_state_dict(torch.load(os.path.join(CACHE_DIR, path)))
    return model.eval()


def hidden_errors(model, ds, picks, hides):
    """Metres per window and step for the model and the line, and which steps are hidden and tracked."""
    out, line, used = [], [], []
    for i in range(0, len(picks), BATCH):
        batch = make_batch(ds, picks[i:i + BATCH], hides[i:i + BATCH])
        with torch.no_grad():
            pred = predict(model, batch)[1]
        out.append((pred - batch["truth"]).square().sum(dim=-1).sqrt().numpy())
        line.append((to_metres(batch["line"]) - batch["truth"]).square().sum(dim=-1).sqrt().numpy())
        used.append(batch["hidden"].numpy() > 0)
    return np.concatenate(out), np.concatenate(line), np.concatenate(used)


def lengths(ds, names):
    """Error per gap length on the validation blocks, next to the straight line."""
    _, val = bd.split(ds, FOLD)
    picks = bd.window_starts(val, ds["half"], ds["known"])[::STRIDE_S * bd.HZ]
    print("validation blocks, {} windows, 10 Hz steps, hidden stretch in the middle of each".format(len(picks)))
    print("gap   steps   line median   " + "   ".join("{:>5s} median".format(n) for n in names)
          + "   line mean   " + "   ".join("{:>5s} mean".format(n) for n in names))
    models = [load_model(n) for n in names]
    for seconds in GAP_SECONDS:
        hides = [centre_hidden(seconds)] * len(picks)
        runs_ = [hidden_errors(m, ds, picks, hides) for m in models]
        a = runs_[0][1][runs_[0][2]]
        e = [r[0][r[2]] for r in runs_]
        print("{:2d} s {:6d}   {:6.1f} m      ".format(seconds, len(a), np.median(a))
              + "   ".join("{:7.1f} m   ".format(np.median(v)) for v in e)
              + " {:6.1f} m   ".format(a.mean()) + "   ".join("{:6.1f} m".format(v.mean()) for v in e))


def stopped_flags(ds):
    """Per step: stopped by the open play rule (tracked steps), a dead_balls stoppage, and players walking."""
    sm = dict(np.load(os.path.join(CACHE_DIR, "smooth.npz")))
    raw = np.load(os.path.join(CACHE_DIR, "ball.npz"))
    scored = np.load(os.path.join(CACHE_DIR, "scored.npy"))
    keep = open_play(sm, raw, scored, read_tags())
    frames = ds["frames"]
    return ~keep[frames] & ds["known"], stoppage_mask(len(scored))[frames]


def group_row(label, e, a, windows):
    """One line of the table."""
    if not len(e):
        print("  {:32s} {:6d}".format(label, 0))
        return
    print("  {:32s} {:6d}  {:4d}   {:6.1f} m  {:6.1f} m    {:6.1f} m  {:6.1f} m".format(
        label, len(e), windows, np.median(e), np.mean(e), np.median(a), np.mean(a)))


def error_table(model, ds, mask, stopped, stoppage, label):
    """Error on hidden tracked steps, stopped play against open play."""
    picks = bd.window_starts(mask, ds["half"], ds["known"])[::STRIDE_S * bd.HZ]
    rng = np.random.default_rng(SEED)
    hides = [hide_stretches(rng) for _ in picks]
    e, a, used = hidden_errors(model, ds, picks, hides)
    grid = picks[:, None] + np.arange(bd.WINDOW)
    groups = (("open play", ~stopped[grid]), ("stopped, all", stopped[grid]),
              ("  of which dead_balls stoppage", stopped[grid] & stoppage[grid]),
              ("  of which other stopped play", stopped[grid] & ~stoppage[grid]))
    print("\n{}: {} windows, ball hidden the same way as in training".format(label, len(picks)))
    print("  {:32s}  steps  wins   model med  model mean   line med  line mean".format(""))
    for name, flag in groups:
        pick = used & flag
        group_row(name, e[pick], a[pick], int(pick.any(axis=1).sum()))


def stoppages(ds, name):
    """How many training steps are stopped play, and the model's error there against the rest."""
    train, val = bd.split(ds, FOLD)
    stopped, stoppage = stopped_flags(ds)
    known = ds["known"]
    n, seen = train.sum(), (train & known).sum()
    print("fold {} training steps: {}".format(FOLD, n))
    print("  ball tracked (the only steps with a position loss): {} ({:.1%})".format(seen, seen / n))
    print("  stopped by the open play rule, of the tracked:      {} ({:.1%})".format(
        (train & stopped).sum(), (train & stopped).sum() / seen))
    print("    inside a dead_balls stoppage:                     {}".format((train & stopped & stoppage).sum()))
    print("    other stopped play:                               {}".format((train & stopped & ~stoppage).sum()))
    print("  dead_balls stoppage, ball tracked or not:           {} ({:.1%} of all steps)".format(
        (train & stoppage).sum(), (train & stoppage).sum() / n))
    print("  ball missing: the rule cannot judge these:          {} ({:.1%})".format(n - seen, (n - seen) / n))
    model = load_model(name)
    error_table(model, ds, val, stopped, stoppage, "validation blocks (model never trained here)")
    error_table(model, ds, train, stopped, stoppage, "training blocks (model has seen these, so flattered)")


def main():
    what, names = sys.argv[1], sys.argv[2:]
    set_seed(SEED)
    ds = bd.load()
    if what == "lengths":
        lengths(ds, names)
    else:
        stoppages(ds, names[0])


if __name__ == "__main__":
    main()
