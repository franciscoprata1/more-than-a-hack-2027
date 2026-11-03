"""Phase 3, step 3: train the ball model and score it on the hidden gaps.

    .venv\\Scripts\\python.exe v1/ball_train.py --check      # model and reproducibility checks
    .venv\\Scripts\\python.exe v1/ball_train.py --smoke      # fold 0, at most 10 epochs
    .venv\\Scripts\\python.exe v1/ball_train.py --fold 3     # one full fold
    .venv\\Scripts\\python.exe v1/ball_train.py --fold 0 --no-test --plain --tag _plain_long
    .venv\\Scripts\\python.exe v1/ball_train.py --all        # every fold, plain; run it again to resume

Needs cache/ball_dataset.npz (v1/ball_dataset.py), cache/hidden_gaps.npz, cache/ball.npz.
After --all, v1/ball_report.py prints the Stage 3 report.

Training hides stretches of the real ball so the model learns to fill gaps from both ends. Early
stopping looks at validation blocks with stretches hidden the same way every epoch. The hidden gaps
are only opened after training, by the model that never trained on their block, and are never
used to choose anything.

Writes cache/ball_model_fold<k>.pt and cache/ball_results_fold<k>.npz. --all adds _all to both names,
and its results also hold the ball paths on the hidden gaps and on the validation blocks. It saves
cache/ball_checkpoint_fold<k>_all.pt after every epoch, so a stopped run carries on exactly where it
was. Prints totals only.
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ball_dataset as bd  # noqa: E402
from ball_baselines import straight_line  # noqa: E402
from ball_model import BallModel, CE_WEIGHT, REALITY_WEIGHT, position_loss, reality_loss  # noqa: E402
from smooth import FPS, PITCH  # noqa: E402
from hidden_gaps import FOLDS, GAP_SECONDS, SEED  # noqa: E402

# Params
CACHE_DIR = os.path.join(ROOT, "cache")
THREADS = 8              # the Ryzen 7's cores. Changing this can change the results in the last digits
BATCH = 16
LEARNING_RATE = 5e-4     # the paper's
CLIP = 1.0
TRAIN_STRIDE_S = 3       # windows a pass; the start is shifted each epoch, so they don't repeat
VAL_STRIDE_S = 15
MAX_EPOCHS = 60
SMOKE_EPOCHS = 10
PATIENCE = 8             # epochs without a better validation score before stopping
HIDE_S = (2, 20)         # shortest and longest stretch hidden in training
HIDE_COUNT = (1, 3)      # stretches per window
EDGE_S = 1               # a hidden stretch keeps this much of the window each side
VAL_GAP_S = (5, 10, 20)
SAVE_STRIDE_S = 5        # validation windows kept for later: the ones v1/ball_checks.py reports
ALL_TAG = "_all"
L, W = PITCH


def set_seed(seed):
    """Same seed, same thread count, same run."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(THREADS)


def hide_stretches(rng):
    """Random hidden stretches of one window: a bool per step. Lengths are even on a log scale."""
    hidden = np.zeros(bd.WINDOW, dtype=bool)
    edge = EDGE_S * bd.HZ
    for _ in range(rng.integers(HIDE_COUNT[0], HIDE_COUNT[1] + 1)):
        size = int(np.exp(rng.uniform(np.log(HIDE_S[0]), np.log(HIDE_S[1]))) * bd.HZ)
        start = rng.integers(edge, bd.WINDOW - edge - size + 1)
        hidden[start:start + size] = True
    return hidden


def centre_hidden(seconds):
    """One hidden stretch in the middle of the window."""
    hidden = np.zeros(bd.WINDOW, dtype=bool)
    start = (bd.WINDOW - seconds * bd.HZ) // 2
    hidden[start:start + seconds * bd.HZ] = True
    return hidden


def make_batch(ds, starts, hides, rng=None):
    """Tensors for windows at `starts`, ball hidden on `hides`. With rng: slots shuffled and pitch flipped."""
    b = len(starts)
    lists = [np.flatnonzero(ds["present"][s:s + bd.WINDOW].any(axis=0)) for s in starts]
    if rng is not None:
        lists = [rng.permutation(slots) for slots in lists]
    n = max(len(slots) for slots in lists)
    players = np.zeros((b, bd.WINDOW, n, ds["player"].shape[2]), dtype=np.float32)
    present = np.zeros((b, bd.WINDOW, n), dtype=bool)
    ball = np.zeros((b, bd.WINDOW, 3), dtype=np.float32)
    line = np.zeros((b, bd.WINDOW, 2), dtype=np.float32)
    truth = np.zeros((b, bd.WINDOW, 2), dtype=np.float32)
    labels = np.full((b, bd.WINDOW), -100, dtype=np.int64)
    known = np.zeros((b, bd.WINDOW), dtype=bool)
    for i, (s, slots, hidden) in enumerate(zip(starts, lists, hides)):
        e = s + bd.WINDOW
        k = len(slots)
        players[i, :, :k] = ds["player"][s:e][:, slots]
        present[i, :, :k] = ds["present"][s:e][:, slots]
        known[i] = ds["known"][s:e]
        x, y = np.nan_to_num(ds["ball_x"][s:e]), np.nan_to_num(ds["ball_y"][s:e])
        if rng is not None and rng.random() < 0.5:
            players[i, :, :k, 0] *= -1
            players[i, :, :k, 2] *= -1
            x = L - x
        if rng is not None and rng.random() < 0.5:
            players[i, :, :k, 1] *= -1
            players[i, :, :k, 3] *= -1
            y = W - y
        truth[i, :, 0], truth[i, :, 1] = x, y
        shown = known[i] & ~hidden
        ball[i, :, 0] = np.where(shown, (x - L / 2) / (L / 2), 0)
        ball[i, :, 1] = np.where(shown, (y - W / 2) / (W / 2), 0)
        ball[i, :, 2] = shown
        if shown.any():
            at = np.flatnonzero(shown)
            line[i, :, 0] = np.interp(np.arange(bd.WINDOW), at, ball[i, at, 0])
            line[i, :, 1] = np.interp(np.arange(bd.WINDOW), at, ball[i, at, 1])
        lookup = np.full(ds["present"].shape[1], -100, dtype=np.int64)
        lookup[slots] = np.arange(k)
        label = ds["label"][s:e]
        labels[i] = np.where(label >= 0, lookup[np.maximum(label, 0)], np.where(label == bd.NOBODY, n, -100))
    hidden = np.array(hides)
    return {"players": torch.from_numpy(players), "present": torch.from_numpy(present),
            "ball": torch.from_numpy(ball), "line": torch.from_numpy(line), "truth": torch.from_numpy(truth),
            "labels": torch.from_numpy(labels),
            "hidden": torch.from_numpy(known & hidden).float(), "shown": torch.from_numpy(known & ~hidden).float()}


def to_metres(xy):
    return torch.stack([xy[..., 0] * L / 2 + L / 2, xy[..., 1] * W / 2 + W / 2], dim=-1)


def predict(model, batch):
    """(who-has-the-ball logits, ball in metres)."""
    logits, xy = model(batch["players"], batch["present"], batch["ball"], batch["line"])
    return logits, to_metres(xy)


def losses(model, batch):
    """(total, position, who has the ball, reality)."""
    logits, pred = predict(model, batch)
    pos = position_loss(pred, batch["truth"], batch["hidden"], batch["shown"])
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1), ignore_index=-100)
    real = reality_loss(pred, to_metres(batch["players"][..., :2]), batch["present"])
    return pos + CE_WEIGHT * ce + REALITY_WEIGHT * real, pos, ce, real


def validation_set(ds, starts):
    """Fixed windows and fixed hidden stretches, the same every epoch. Lengths take turns."""
    picks = starts[::VAL_STRIDE_S * bd.HZ]
    return picks, [centre_hidden(VAL_GAP_S[i % len(VAL_GAP_S)]) for i in range(len(picks))]


def validate(model, ds, picks, hides):
    """Mean metres on hidden tracked steps."""
    model.eval()
    total, count = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(picks), BATCH):
            batch = make_batch(ds, picks[i:i + BATCH], hides[i:i + BATCH])
            e = (predict(model, batch)[1] - batch["truth"]).square().sum(dim=-1).sqrt()
            total += (e * batch["hidden"]).sum().item()
            count += batch["hidden"].sum().item()
    return total / count


def train_epoch(model, optimiser, ds, starts, rng, limit=None):
    """One pass over shuffled windows. Returns mean (position, who has the ball, reality)."""
    model.train()
    offset = rng.integers(0, TRAIN_STRIDE_S * bd.HZ)
    picks = rng.permutation(starts[offset::TRAIN_STRIDE_S * bd.HZ])
    if limit:
        picks = picks[:limit * BATCH]
    logs = []
    for i in range(0, len(picks), BATCH):
        chunk = picks[i:i + BATCH]
        batch = make_batch(ds, chunk, [hide_stretches(rng) for _ in chunk], rng)
        total, pos, ce, real = losses(model, batch)
        optimiser.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimiser.step()
        logs.append((pos.item(), ce.item(), real.item()))
    return np.mean(logs, axis=0)


def save_checkpoint(path, state):
    """Write then swap, so a run stopped mid-save leaves the last good checkpoint in place."""
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def fit(ds, fold, epochs, on_line=True, log=print, checkpoint=None, limit=None):
    """Train one fold, carrying on from `checkpoint` if it exists. Returns the model at its best validation score."""
    set_seed(SEED + fold)
    rng = np.random.default_rng(SEED + fold)
    train, val = bd.split(ds, fold)
    starts = bd.window_starts(train, ds["half"], ds["known"])
    picks, hides = validation_set(ds, bd.window_starts(val, ds["half"], ds["known"]))
    model = BallModel(on_line=on_line)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best, best_state, best_epoch, times, first, stop = np.inf, None, 0, [], 1, False
    log("fold {}: {} parameters, {} training window starts, {} validation windows, device cpu, {} threads".format(
        fold, sum(p.numel() for p in model.parameters()), len(starts), len(picks), torch.get_num_threads()))
    if checkpoint and os.path.exists(checkpoint):
        saved = torch.load(checkpoint)
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        best, best_state, best_epoch, times = saved["best"], saved["best_state"], saved["best_epoch"], saved["times"]
        first, stop = saved["epoch"] + 1, saved["stop"]
        log("  resumed after epoch {}".format(saved["epoch"]))
    for epoch in range(first, first if stop else epochs + 1):
        t0 = time.perf_counter()
        pos, ce, real = train_epoch(model, optimiser, ds, starts, rng, limit)
        t1 = time.perf_counter()
        score = validate(model, ds, picks, hides)
        times.append((t1 - t0, time.perf_counter() - t1))
        log("  epoch {:2d}: train position {:7.1f} m2, who has it {:.3f}, reality {:.2f}; validation {:.2f} m; {:.0f} s + {:.0f} s".format(
            epoch, pos, ce, real, score, *times[-1]))
        if score < best:
            best, best_epoch = score, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        stop = epoch - best_epoch >= PATIENCE
        if checkpoint:
            save_checkpoint(checkpoint, {"epoch": epoch, "stop": stop, "model": model.state_dict(),
                                         "optimiser": optimiser.state_dict(), "numpy_rng": rng.bit_generator.state,
                                         "torch_rng": torch.get_rng_state(), "best": best, "best_state": best_state,
                                         "best_epoch": best_epoch, "times": times})
        if stop:
            log("  stopped early: no better for {} epochs".format(PATIENCE))
            break
    model.load_state_dict(best_state)
    log("  best validation {:.2f} m at epoch {}; {:.0f} s in all; train {:.0f} s and validate {:.0f} s per epoch".format(
        best, best_epoch, np.sum(times), *np.mean(times, axis=0)))
    return model


def fill_gap(ds, start, pred, hidden, frames_wanted):
    """Ball path at 25 fps from 10 Hz steps: the model on hidden steps, the tracked ball where shown."""
    e = start + bd.WINDOW
    steps = ds["frames"][start:e]
    use = ds["known"][start:e] & ~hidden
    x = np.where(use, ds["ball_x"][start:e], pred[:, 0])
    y = np.where(use, ds["ball_y"][start:e], pred[:, 1])
    return np.c_[np.interp(frames_wanted, steps, x), np.interp(frames_wanted, steps, y)]


def score_gaps(model, ds, gaps, seconds, x_true, y_true):
    """Paths at 25 fps per gap for the model and baseline A, the true path, and who-has-the-ball agreement."""
    length = seconds * FPS
    model.eval()
    paths, line, truths, right, labelled = [], [], [], 0, 0
    frames, half = ds["frames"], ds["half"]
    for i in range(0, len(gaps), BATCH):
        chunk = gaps[i:i + BATCH]
        spans = [np.searchsorted(frames, [a, a + length]) for a in chunk]
        starts = [bd.window_around(p, q - 1, half) for p, q in spans]
        hides = [(frames[s:s + bd.WINDOW] >= a) & (frames[s:s + bd.WINDOW] < a + length) for s, a in zip(starts, chunk)]
        batch = make_batch(ds, starts, hides)
        with torch.no_grad():
            logits, pred = predict(model, batch)
        pred = pred.numpy()
        guess = logits.argmax(dim=-1).numpy()
        labels = batch["labels"].numpy()
        for k, (a, s) in enumerate(zip(chunk, starts)):
            wanted = np.arange(a, a + length)
            truths.append(np.c_[x_true[wanted], y_true[wanted]])
            paths.append(fill_gap(ds, s, pred[k], hides[k], wanted))
            last, nxt = np.array([x_true[a - 1], y_true[a - 1]]), np.array([x_true[a + length], y_true[a + length]])
            line.append(straight_line(last, nxt, length))
            mine = hides[k] & (labels[k] >= 0)
            right += (guess[k][mine] == labels[k][mine]).sum()
            labelled += mine.sum()
    shape = (len(gaps), length, 2)
    return [np.array(v).reshape(shape) for v in (paths, line, truths)] + [right, labelled]


def validation_paths(model, ds, fold):
    """Ball on this fold's validation blocks, one hidden stretch in the middle of each window, per gap length.
    Whole windows at 10 Hz in metres. val_hidden marks the steps to score: hidden and tracked."""
    _, val = bd.split(ds, fold)
    picks = bd.window_starts(val, ds["half"], ds["known"])[::SAVE_STRIDE_S * bd.HZ]
    model.eval()
    out = {"val_starts": picks}
    for seconds in GAP_SECONDS:
        hides = [centre_hidden(seconds)] * len(picks)
        parts = []
        for i in range(0, len(picks), BATCH):
            batch = make_batch(ds, picks[i:i + BATCH], hides[i:i + BATCH])
            with torch.no_grad():
                pred = predict(model, batch)[1]
            parts.append((pred.numpy(), to_metres(batch["line"]).numpy(), batch["truth"].numpy(), batch["hidden"].numpy() > 0))
        for name, part in zip(("model", "line", "truth", "hidden"), zip(*parts)):
            out["val_{}_{}".format(name, seconds)] = np.concatenate(part)
    return out


def results_path(fold, tag):
    return os.path.join(CACHE_DIR, "ball_results_fold{}{}.npz".format(fold, tag))


def test_fold(model, ds, fold, hidden, x_true, y_true, tag=""):
    """Open this fold's hidden gaps. Saves errors and paths, and returns them by length."""
    out = {}
    for seconds in GAP_SECONDS:
        gaps = hidden["gap_{}".format(seconds)]
        mine = gaps[hidden["fold_{}".format(seconds)] == fold]
        path, line, truth, right, labelled = score_gaps(model, ds, mine, seconds, x_true, y_true)
        out.update({"model_{}".format(seconds): np.hypot(*np.moveaxis(path - truth, 2, 0)),
                    "line_{}".format(seconds): np.hypot(*np.moveaxis(line - truth, 2, 0)),
                    "gaps_{}".format(seconds): mine, "right_{}".format(seconds): right, "labelled_{}".format(seconds): labelled,
                    "path_model_{}".format(seconds): path.astype(np.float32), "path_line_{}".format(seconds): line.astype(np.float32),
                    "path_truth_{}".format(seconds): truth.astype(np.float32)})
    out.update(validation_paths(model, ds, fold))
    temporary = results_path(fold, tag) + ".tmp.npz"
    np.savez_compressed(temporary, **out)
    os.replace(temporary, results_path(fold, tag))  # a results file that exists is always a whole one
    return out


def print_fold(results):
    """Median and mean per gap length, next to baseline A on the same gaps."""
    print("\ngap    n    model median  A median   model mean  A mean   who has it right")
    for seconds in GAP_SECONDS:
        e, a = results["model_{}".format(seconds)], results["line_{}".format(seconds)]
        print("{:2d} s {:4d}    {:6.1f} m    {:6.1f} m    {:6.1f} m  {:6.1f} m   {:.0%}".format(
            seconds, len(e), np.median(e), np.median(a), e.mean(), a.mean(),
            results["right_{}".format(seconds)] / max(1, results["labelled_{}".format(seconds)])))


def run_fold(ds, fold, epochs, hidden, x_true, y_true, on_line=True, tag="", resume=False):
    checkpoint = os.path.join(CACHE_DIR, "ball_checkpoint_fold{}{}.pt".format(fold, tag)) if resume else None
    model = fit(ds, fold, epochs, on_line, checkpoint=checkpoint)
    torch.save(model.state_dict(), os.path.join(CACHE_DIR, "ball_model_fold{}{}.pt".format(fold, tag)))
    if hidden is None:
        return None
    return test_fold(model, ds, fold, hidden, x_true, y_true, tag)


def fingerprint(model):
    """Short hash of every weight: two runs agree only if they match to the last bit."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()[:16]


def check(ds):
    """Does player order not matter, and does a short run repeat exactly?"""
    set_seed(SEED)
    model = BallModel().eval()
    train, _ = bd.split(ds, 0)
    starts = bd.window_starts(train, ds["half"], ds["known"])[::5000][:4]
    batch = make_batch(ds, starts, [centre_hidden(10)] * len(starts))
    order = torch.randperm(batch["players"].shape[2])
    with torch.no_grad():
        a_logits, a_xy = model(batch["players"], batch["present"], batch["ball"], batch["line"])
        b_logits, b_xy = model(batch["players"][:, :, order], batch["present"][:, :, order], batch["ball"], batch["line"])
    print("order check: ball moves {:.2e}, who has it moves {:.2e}".format(
        (a_xy - b_xy).abs().max().item(), (a_logits[..., :-1][:, :, order] - b_logits[..., :-1]).abs().max().item()))

    sums = []
    for _ in range(2):
        set_seed(SEED)
        rng = np.random.default_rng(SEED)
        model = BallModel()
        optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        t0 = time.perf_counter()
        train_epoch(model, optimiser, ds, bd.window_starts(train, ds["half"], ds["known"]), rng, limit=3)
        sums.append(fingerprint(model))
        print("3 batches: fingerprint {}, {:.1f} s per batch".format(sums[-1], (time.perf_counter() - t0) / 3))
    print("same run twice in this process:", "IDENTICAL" if sums[0] == sums[1] else "DIFFERENT")

    # Two epochs straight against one, stop, reload from disk, one more.
    paths = [os.path.join(CACHE_DIR, "ball_check_{}.pt".format(name)) for name in ("straight", "resumed")]
    for path in paths:
        if os.path.exists(path):
            os.remove(path)
    quiet = lambda _: None
    fit(ds, 0, 2, False, quiet, paths[0], limit=3)
    fit(ds, 0, 1, False, quiet, paths[1], limit=3)
    fit(ds, 0, 2, False, quiet, paths[1], limit=3)
    a, b = [torch.load(path) for path in paths]
    same = all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    print("two epochs against stopped and resumed:", "IDENTICAL" if same and a["epoch"] == b["epoch"] == 2 else "DIFFERENT")
    for path in paths:
        os.remove(path)
    print("validation windows saved per fold:", [
        len(bd.window_starts(bd.split(ds, fold)[1], ds["half"], ds["known"])[::SAVE_STRIDE_S * bd.HZ]) for fold in range(FOLDS)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-test", action="store_true", help="train only, leave the hidden gaps closed")
    parser.add_argument("--plain", action="store_true", help="predict the ball itself, not a correction to the line")
    parser.add_argument("--tag", default="", help="added to the saved model's name")
    args = parser.parse_args()

    ds = bd.load()
    if args.check:
        check(ds)
        return
    hidden = np.load(os.path.join(CACHE_DIR, "hidden_gaps.npz"))
    raw = np.load(os.path.join(CACHE_DIR, "ball.npz"))
    x_true, y_true = raw["x"].astype(np.float64), raw["y"].astype(np.float64)
    if args.smoke and args.no_test:
        run_fold(ds, 0, SMOKE_EPOCHS, None, x_true, y_true)
    elif args.smoke:
        print_fold(run_fold(ds, 0, SMOKE_EPOCHS, hidden, x_true, y_true))
    elif args.fold is not None and args.no_test:
        run_fold(ds, args.fold, MAX_EPOCHS, None, x_true, y_true, not args.plain, args.tag)
    elif args.fold is not None:
        print_fold(run_fold(ds, args.fold, MAX_EPOCHS, hidden, x_true, y_true, not args.plain, args.tag))
    elif args.all:
        for fold in range(FOLDS):
            if os.path.exists(results_path(fold, ALL_TAG)):
                print("fold {}: already done".format(fold))
                continue
            print_fold(run_fold(ds, fold, MAX_EPOCHS, hidden, x_true, y_true, False, ALL_TAG, True))


if __name__ == "__main__":
    main()
