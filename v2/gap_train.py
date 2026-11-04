"""Part 2, step 3: train the v1 ball model on the degraded bench, one match left out per fold.

    .venv\\Scripts\\python.exe v2/gap_train.py --smoke     # fold 0, 2 epochs: measures the epoch time
    .venv\\Scripts\\python.exe v2/gap_train.py --fold 3    # one fold
    .venv\\Scripts\\python.exe v2/gap_train.py --all       # every fold; run it again to resume

Needs cache/v2_gap/<match>.npz (v2/gap_dataset.py). The model is v1/ball_model.py, plain (no line).

Fold k leaves match k out: the model trains on the other 6 and fills match k's gaps, which it has
never seen. Validation is each training match's validation blocks (v2/gap_dataset.py), kept
BUFFER_S from anything trained on, scored on the real gaps only (as placed by the degradation).
Every fold's model also fills our match; none of them has seen it.

What the loss sees: the true ball on every step that is not shown to the model. That is the real
gaps, plus 0-2 extra stretches hidden on purpose (open-play gaps, as v1 trained on).

Each epoch takes one window every TRAIN_STRIDE_S per match, from a new random offset, so the
epoch costs about v1's while the run still covers every match. A fold stops at PATIENCE epochs
without a better validation score, MAX_EPOCHS, or FOLD_CAP_MIN minutes of training.

Writes cache/v2_gap/model_fold<k>.pt, checkpoint_fold<k>.pt (every epoch, so --all resumes exactly),
pred_<match>_fold<k>.npz for the left-out match and for ours: per frame x, y, predicted, and the
who-has-the-ball shares p_home, p_away, p_nobody. Prints totals only.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bench  # noqa: E402
import gap_dataset as gd  # noqa: E402
from bench import MATCHES  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "v1"))
import ball_dataset as bd  # noqa: E402
from ball_model import BallModel, CE_WEIGHT, REALITY_WEIGHT, position_loss, reality_loss  # noqa: E402
from ball_train import BATCH, CLIP, EDGE_S, HIDE_S, LEARNING_RATE, PATIENCE, save_checkpoint, set_seed  # noqa: E402
from hidden_gaps import SEED  # noqa: E402

# Params
OUT_DIR = gd.OUT_DIR
TRAIN_STRIDE_S = 15      # v1 used 3 s on one match; 6 matches at 15 s is about the same epoch
VAL_STRIDE_S = 10
PRED_STRIDE_S = 15       # windows overlap by half; each step takes the window it is most central in
HIDE_COUNT = (0, 2)      # extra stretches per window, on top of the real gaps
MAX_EPOCHS = 60
SMOKE_EPOCHS = 2
FOLD_CAP_MIN = 100       # keeps 7 folds inside one night
BUFFER_S = bd.BUFFER_S


def load_all():
    return {name: gd.load(name) for name in gd.NAMES}


def split(ds):
    """(train, validation) masks of a match that trains."""
    val = ds["val"]
    return ~bd.dilate(val, BUFFER_S * bd.HZ), val


def hide_extra(rng):
    hidden = np.zeros(bd.WINDOW, dtype=bool)
    edge = EDGE_S * bd.HZ
    for _ in range(rng.integers(HIDE_COUNT[0], HIDE_COUNT[1] + 1)):
        size = int(np.exp(rng.uniform(np.log(HIDE_S[0]), np.log(HIDE_S[1]))) * bd.HZ)
        start = rng.integers(edge, bd.WINDOW - edge - size + 1)
        hidden[start:start + size] = True
    return hidden


def make_batch(data, items, hides, rng=None):
    """Tensors for windows items = [(match, start)], extra stretches hidden on `hides`.
    With rng: slots shuffled and pitch flipped. 'hidden' = the loss steps, 'shown' = the ball input."""
    b = len(items)
    lists = [np.flatnonzero(data[nm]["present"][s:s + bd.WINDOW].any(axis=0)) for nm, s in items]
    if rng is not None:
        lists = [rng.permutation(slots) for slots in lists]
    n = max(len(slots) for slots in lists)
    players = np.zeros((b, bd.WINDOW, n, 7), dtype=np.float32)
    present = np.zeros((b, bd.WINDOW, n), dtype=bool)
    ball = np.zeros((b, bd.WINDOW, 3), dtype=np.float32)
    truth = np.zeros((b, bd.WINDOW, 2), dtype=np.float32)
    labels = np.full((b, bd.WINDOW), -100, dtype=np.int64)
    hidden = np.zeros((b, bd.WINDOW), dtype=bool)
    shown = np.zeros((b, bd.WINDOW), dtype=bool)
    pitch = np.zeros((b, 2), dtype=np.float32)
    for i, ((nm, s), slots, extra) in enumerate(zip(items, lists, hides)):
        ds, e, k = data[nm], s + bd.WINDOW, len(slots)
        L, W = ds["pitch"]
        pitch[i] = L, W
        players[i, :, :k] = ds["player"][s:e][:, slots]
        present[i, :, :k] = ds["present"][s:e][:, slots]
        x, y = np.nan_to_num(ds["truth_x"][s:e]), np.nan_to_num(ds["truth_y"][s:e])
        if rng is not None and rng.random() < 0.5:
            players[i, :, :k, 0] *= -1
            players[i, :, :k, 2] *= -1
            x = L - x
        if rng is not None and rng.random() < 0.5:
            players[i, :, :k, 1] *= -1
            players[i, :, :k, 3] *= -1
            y = W - y
        truth[i, :, 0], truth[i, :, 1] = x, y
        shown[i] = ds["known"][s:e] & ~extra
        hidden[i] = ds["truth_known"][s:e] & ~shown[i]
        ball[i, :, 0] = np.where(shown[i], (x - L / 2) / (L / 2), 0)
        ball[i, :, 1] = np.where(shown[i], (y - W / 2) / (W / 2), 0)
        ball[i, :, 2] = shown[i]
        lookup = np.full(ds["present"].shape[1], -100, dtype=np.int64)
        lookup[slots] = np.arange(k)
        label = ds["label"][s:e]
        labels[i] = np.where(label >= 0, lookup[np.maximum(label, 0)], np.where(label == bd.NOBODY, n, -100))
    return {"players": torch.from_numpy(players), "present": torch.from_numpy(present),
            "ball": torch.from_numpy(ball), "line": torch.zeros(b, bd.WINDOW, 2), "truth": torch.from_numpy(truth),
            "labels": torch.from_numpy(labels), "pitch": torch.from_numpy(pitch),
            "hidden": torch.from_numpy(hidden).float(), "shown": torch.from_numpy(shown).float()}


def to_metres(xy, pitch):
    """Scaled -1..1 to metres, each window by its own pitch."""
    shape = (pitch.shape[0],) + (1,) * (xy.dim() - 2)
    L, W = pitch[:, 0].reshape(shape), pitch[:, 1].reshape(shape)
    return torch.stack([xy[..., 0] * L / 2 + L / 2, xy[..., 1] * W / 2 + W / 2], dim=-1)


def predict(model, batch):
    logits, xy = model(batch["players"], batch["present"], batch["ball"], batch["line"])
    return logits, to_metres(xy, batch["pitch"])


def losses(model, batch):
    logits, pred = predict(model, batch)
    pos = position_loss(pred, batch["truth"], batch["hidden"], batch["shown"])
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1), ignore_index=-100)
    real = reality_loss(pred, to_metres(batch["players"][..., :2], batch["pitch"]), batch["present"])
    return pos + CE_WEIGHT * ce + REALITY_WEIGHT * real, pos, ce, real


def starts_of(data, names, which):
    """{match: window starts} inside the train (0) or validation (1) mask, with enough seen ball."""
    out = {}
    for nm in names:
        ds = data[nm]
        out[nm] = bd.window_starts(split(ds)[which], ds["half"], ds["known"])
    return out


def validate(model, data, items):
    """Mean metres on the real gaps of fixed validation windows (nothing extra hidden)."""
    model.eval()
    total, count = 0.0, 0.0
    none = [np.zeros(bd.WINDOW, dtype=bool)] * BATCH
    with torch.no_grad():
        for i in range(0, len(items), BATCH):
            batch = make_batch(data, items[i:i + BATCH], none)
            e = (predict(model, batch)[1] - batch["truth"]).square().sum(dim=-1).sqrt()
            total += (e * batch["hidden"]).sum().item()
            count += batch["hidden"].sum().item()
    return total / count


def train_epoch(model, optimiser, data, starts, rng):
    model.train()
    items = []
    for nm, st in starts.items():
        offset = rng.integers(0, TRAIN_STRIDE_S * bd.HZ)
        items += [(nm, s) for s in st[offset::TRAIN_STRIDE_S * bd.HZ]]
    items = [items[i] for i in rng.permutation(len(items))]
    logs = []
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        batch = make_batch(data, chunk, [hide_extra(rng) for _ in chunk], rng)
        total, pos, ce, real = losses(model, batch)
        optimiser.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimiser.step()
        logs.append((pos.item(), ce.item(), real.item()))
    return len(items), np.mean(logs, axis=0)


def fit(data, fold, epochs, log=print, checkpoint=None):
    """Train fold `fold` (its match left out), carrying on from `checkpoint`. Returns the best model."""
    set_seed(SEED + fold)
    rng = np.random.default_rng(SEED + fold)
    names = [nm for k, nm in enumerate(MATCHES) if k != fold]
    starts = starts_of(data, names, 0)
    vs = starts_of(data, names, 1)
    val_items = [(nm, s) for nm in names for s in vs[nm][::VAL_STRIDE_S * bd.HZ]]
    model = BallModel(on_line=False)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best, best_state, best_epoch, times, first, stop = np.inf, None, 0, [], 1, False
    log("fold {} (leaves {} out): {} training window starts in {} matches, {} validation windows, {} threads".format(
        fold, MATCHES[fold], sum(len(v) for v in starts.values()), len(names), len(val_items), torch.get_num_threads()))
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
        n, (pos, ce, real) = train_epoch(model, optimiser, data, starts, rng)
        t1 = time.perf_counter()
        score = validate(model, data, val_items)
        times.append((t1 - t0, time.perf_counter() - t1))
        log("  epoch {:2d}: {} windows, train position {:7.1f} m2, who has it {:.3f}, reality {:.2f}; "
            "validation {:.2f} m on real gaps; {:.0f} s + {:.0f} s".format(epoch, n, pos, ce, real, score, *times[-1]))
        if score < best:
            best, best_epoch = score, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        stop = epoch - best_epoch >= PATIENCE
        capped = np.sum(times) > FOLD_CAP_MIN * 60
        if checkpoint:
            save_checkpoint(checkpoint, {"epoch": epoch, "stop": stop or capped, "model": model.state_dict(),
                                         "optimiser": optimiser.state_dict(), "numpy_rng": rng.bit_generator.state,
                                         "torch_rng": torch.get_rng_state(), "best": best, "best_state": best_state,
                                         "best_epoch": best_epoch, "times": times})
        if stop:
            log("  stopped early: no better for {} epochs".format(PATIENCE))
            break
        if capped:
            log("  stopped at the {} min cap".format(FOLD_CAP_MIN))
            break
    model.load_state_dict(best_state)
    log("  best validation {:.2f} m at epoch {}; {:.0f} min in all; {:.0f} s train + {:.0f} s validate per epoch".format(
        best, best_epoch, np.sum(times) / 60, *np.mean(times, axis=0)))
    return model


def fill_match(model, ds):
    """Every step of the match from overlapping windows, each step from the window it is most central in.
    Returns per frame (x, y, predicted, p_home, p_away, p_nobody); the seen ball is kept as seen."""
    model.eval()
    steps = len(ds["frames"])
    out = np.full((steps, 5), np.nan, dtype=np.float32)    # x, y, p_home, p_away, p_nobody
    dist = np.full(steps, np.inf)
    items = []
    for h in (0, 1):
        idx = np.flatnonzero(ds["half"] == h)
        a, b = idx[0], idx[-1] + 1 - bd.WINDOW
        items += sorted(set(range(a, b + 1, PRED_STRIDE_S * bd.HZ)) | {b})
    centre = np.abs(np.arange(bd.WINDOW) - (bd.WINDOW - 1) / 2)
    none = [np.zeros(bd.WINDOW, dtype=bool)] * BATCH
    data = {"m": ds}
    with torch.no_grad():
        for i in range(0, len(items), BATCH):
            chunk = items[i:i + BATCH]
            batch = make_batch(data, [("m", s) for s in chunk], none)
            logits, xy = predict(model, batch)
            p = torch.softmax(logits, dim=-1)
            team = batch["players"][..., 6]
            pres = batch["present"].float()
            ph = (p[..., :-1] * pres * (team == 0)).sum(-1)
            pa = (p[..., :-1] * pres * (team == 1)).sum(-1)
            vals = torch.cat([xy, ph[..., None], pa[..., None], p[..., -1:]], dim=-1).numpy()
            for k, s in enumerate(chunk):
                sl = slice(s, s + bd.WINDOW)
                better = centre < dist[sl]
                out[sl][better] = vals[k][better]
                dist[sl] = np.where(better, centre, dist[sl])
    frames = ds["frames"]
    n = frames[-1] + 3
    res = {k: np.full(n, np.nan, dtype=np.float32) for k in ("x", "y", "p_home", "p_away", "p_nobody")}
    for h in (0, 1):
        sel = ds["half"] == h
        f = frames[sel]
        want = np.arange(f[0], f[-1] + 1)
        for j, k in enumerate(("x", "y", "p_home", "p_away", "p_nobody")):
            res[k][want] = np.interp(want, f, out[sel, j])
    return res


def save_fill(model, data, name, fold, tag=""):
    """The model's ball and shares for a whole match; the seen raw ball is kept where seen."""
    res = fill_match(model, data[name])
    m = bench.load(name if name == "ours" else name + "~")
    n = len(m["half"])
    for k in res:
        v = np.full(n, np.nan, dtype=np.float32)
        v[:min(n, len(res[k]))] = res[k][:n]
        res[k] = v
    seen = np.isfinite(m["raw_ball_x"])
    res["predicted"] = ~seen & (m["half"] > 0)
    res["x"] = np.where(seen, m["raw_ball_x"], res["x"]).astype(np.float32)
    res["y"] = np.where(seen, m["raw_ball_y"], res["y"]).astype(np.float32)
    path = os.path.join(OUT_DIR, "pred_{}_fold{}{}.npz".format(name, fold, tag))
    np.savez_compressed(path + ".tmp.npz", **res)
    os.replace(path + ".tmp.npz", path)


def pred_done(fold):
    return all(os.path.exists(os.path.join(OUT_DIR, "pred_{}_fold{}.npz".format(nm, fold))) for nm in (MATCHES[fold], "ours"))


def run_fold(data, fold, epochs=MAX_EPOCHS, smoke=False):
    tag = "_smoke" if smoke else ""
    checkpoint = None if smoke else os.path.join(OUT_DIR, "checkpoint_fold{}.pt".format(fold))
    t0 = time.time()
    model = fit(data, fold, epochs, checkpoint=checkpoint)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "model_fold{}{}.pt".format(fold, tag)))
    for name in (MATCHES[fold], "ours"):
        save_fill(model, data, name, fold, tag)
    print("  fold {} filled {} and ours; {:.0f} min with filling".format(fold, MATCHES[fold], (time.time() - t0) / 60), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    data = load_all()
    if args.smoke:
        t0 = time.time()
        run_fold(data, 0, SMOKE_EPOCHS, smoke=True)
        print("smoke: {:.0f} s".format(time.time() - t0))
    elif args.fold is not None:
        run_fold(data, args.fold)
    elif args.all:
        for fold in range(len(MATCHES)):
            if pred_done(fold):
                print("fold {}: already done".format(fold))
                continue
            run_fold(data, fold)


if __name__ == "__main__":
    main()
