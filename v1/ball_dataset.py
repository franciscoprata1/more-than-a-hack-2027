"""Phase 3, step 3, stage 1: turn the match into 10 Hz training material for the ball model.

    .venv\\Scripts\\python.exe v1/ball_dataset.py

Needs cache/smooth.npz, cache/ball.npz, cache/scored.npy, cache/hidden_gaps.npz, cache/tags.csv.

  - 10 Hz: one step every 2.5 frames, inside each half. A step is a real frame, never an average;
  - per player: position, velocity, speed, acceleration, team. A slot is absent where it has no
    track or moves faster than a human (a slot jump). Absent slots are masked, not guessed;
  - ball: the tracked position and a known flag. Raw ball only, so a test gap's frames cannot leak
    in through the smoothing or the short-gap filling;
  - who has the ball, made by code from the tracked ball. The nearest player within HAS_BALL_M has
    it. While it travels, the next player to receive it has it. Otherwise nobody / out. Where the
    ball is missing there is nothing to check a label against, so those frames are not trained on.

Folds are the 5-minute blocks of v1/hidden_gaps.py. A test fold's blocks and a validation block
are kept BUFFER_S away from anything trained on: a gap can start in one block and run into the
next, which belongs to another fold.

Writes cache/ball_dataset.npz. Prints totals only.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from smooth import FPS, PITCH, MISSING, SEEN, runs  # noqa: E402
from dead_balls import off_pitch, read_tags  # noqa: E402
from hidden_gaps import FOLD_BLOCK, FOLDS, GAP_SECONDS, HUMAN_SPEED, second_half_wait  # noqa: E402

# Params
CACHE_DIR = os.path.join(ROOT, "cache")
OUT_PATH = os.path.join(CACHE_DIR, "ball_dataset.npz")
HZ = 10
STEP_FRAMES = FPS / HZ       # 2.5: steps fall on frames 0, 2, 5, 7, ...
WINDOW_S = 30                # a 20 s gap in the middle keeps 5 s of real ball each side
WINDOW = WINDOW_S * HZ
BUFFER_S = 20                # the longest test gap: nothing within this of a held-out block is trained on
HAS_BALL_M = 1.5             # about a stride: within this the player can play the ball
MIN_KNOWN = 0.3              # a training window needs this share of tracked ball, or it teaches little
SPEED_SCALE = 10.0           # m/s; keeps the inputs near the range -1 to 1
ACCEL_SCALE = 10.0           # m/s2
ACCEL_LIMIT = 2.0            # after scaling: 20 m/s2 is beyond any real start, so it is a tracking blip
PLAYER_FEATURES = ("x", "y", "vx", "vy", "speed", "accel", "team")

# Who has the ball, besides a slot number.
NOBODY, UNKNOWN = -1, -2


def nearest_player(sm, scored):
    """Slot within HAS_BALL_M of the seen ball per frame, -1 if none. Slot jumps cannot have it."""
    near = np.full(len(scored), -1, dtype=np.int16)
    frames = np.flatnonzero(scored & (sm["ball_state"] == SEEN))
    for chunk in np.array_split(frames, 200):
        d = np.hypot(sm["player_x"][chunk] - sm["ball_x"][chunk, None],
                     sm["player_y"][chunk] - sm["ball_y"][chunk, None])
        d[~(sm["player_speed"][chunk] <= HUMAN_SPEED)] = np.inf
        best = d.argmin(axis=1)
        close = d[np.arange(len(chunk)), best] <= HAS_BALL_M
        near[chunk[close]] = best[close]
    return near


def who_has_ball(sm, scored):
    """Label per frame: a slot, NOBODY or UNKNOWN. Walks backwards, to know the next receiver."""
    near = nearest_player(sm, scored)
    out = off_pitch(sm["ball_x"], sm["ball_y"])
    state = sm["ball_state"]
    labels = np.full(len(scored), UNKNOWN, dtype=np.int16)
    receiver = NOBODY
    for t in range(len(scored) - 1, -1, -1):
        if not scored[t] or state[t] == MISSING:
            receiver = NOBODY    # whoever got it next is out of sight
        elif out[t]:
            receiver = NOBODY
            labels[t] = NOBODY
        else:
            if near[t] >= 0:
                receiver = near[t]
            labels[t] = receiver
    return labels, near


def step_frames(scored):
    """(frames, half) of the 10 Hz steps, half = 0 or 1. Each half starts on its own first frame."""
    frames, half = [], []
    for h, (start, end) in enumerate(runs(scored)):
        f = start + np.floor(np.arange(int((end - start) / STEP_FRAMES)) * STEP_FRAMES).astype(np.int64)
        frames.append(f)
        half.append(np.full(len(f), h, dtype=np.int8))
    return np.concatenate(frames), np.concatenate(half)


def player_inputs(sm, frames):
    """(steps, slots, features) scaled to about -1..1, and which slots are present at each step."""
    length, width = PITCH
    speed = sm["player_speed"][frames]
    x = (sm["player_x"][frames] - length / 2) / (length / 2)
    y = (sm["player_y"][frames] - width / 2) / (width / 2)
    vx, vy = sm["player_vx"][frames] / SPEED_SCALE, sm["player_vy"][frames] / SPEED_SCALE
    present = np.isfinite(x) & np.isfinite(y) & np.isfinite(vx) & np.isfinite(vy) & (speed <= HUMAN_SPEED)
    accel = np.clip(np.nan_to_num(sm["player_accel"][frames]) / ACCEL_SCALE, 0, ACCEL_LIMIT)
    team = np.broadcast_to(sm["team"].astype(np.float32), x.shape)
    feats = np.stack([x, y, vx, vy, speed / SPEED_SCALE, accel, team], axis=-1).astype(np.float32)
    feats[~present] = 0
    return feats, present


def build():
    sm = dict(np.load(os.path.join(CACHE_DIR, "smooth.npz")))
    raw = np.load(os.path.join(CACHE_DIR, "ball.npz"))
    scored = np.load(os.path.join(CACHE_DIR, "scored.npy"))
    frames, half = step_frames(scored)
    labels_25, near_25 = who_has_ball(sm, scored)

    feats, present = player_inputs(sm, frames)
    labels = labels_25[frames].copy()
    steps = np.arange(len(frames))
    # A receiver's slot may not exist yet at this step. The target would then be unreachable.
    absent = (labels >= 0) & ~present[steps, np.maximum(labels, 0)]
    dropped = int(absent.sum())
    labels[absent] = UNKNOWN

    elapsed = np.cumsum(scored) - 1
    block = (elapsed[frames] // FOLD_BLOCK).astype(np.int16)
    first, kickoff = second_half_wait(read_tags(), scored)
    ds = {"frames": frames, "half": half, "block": block, "label": labels.astype(np.int8),
          "ball_x": raw["x"][frames].astype(np.float32), "ball_y": raw["y"][frames].astype(np.float32),
          "known": np.isfinite(raw["x"][frames]), "waiting": (frames >= first) & (frames < kickoff),
          "player": feats, "present": present}
    stats = {"labels_25": labels_25, "near_25": near_25, "scored": scored, "state": sm["ball_state"], "dropped": dropped}
    return ds, stats


def dilate(mask, width):
    """True within `width` steps of a True."""
    return np.convolve(mask.astype(np.int32), np.ones(2 * width + 1, dtype=np.int32), "same") > 0


def held_out_blocks(ds, test_fold):
    """(test blocks, validation blocks). Validation: the next fold's first block in each half."""
    block, half = ds["block"], ds["half"]
    test = sorted(set(block[block % FOLDS == test_fold].tolist()))
    next_fold = sorted(set(block[block % FOLDS == (test_fold + 1) % FOLDS].tolist()))
    first_half_of = {b: half[np.argmax(block == b)] for b in next_fold}
    val = [next(b for b in next_fold if first_half_of[b] == h) for h in (0, 1)]
    return test, val


def split(ds, test_fold):
    """(train, validation) masks over steps. Nothing near a held-out block is trained on."""
    test, val = held_out_blocks(ds, test_fold)
    is_test, is_val = np.isin(ds["block"], test), np.isin(ds["block"], val)
    buffer = BUFFER_S * HZ
    train = ~dilate(is_test | is_val, buffer) & ~ds["waiting"]
    validation = is_val & ~dilate(is_test, buffer) & ~ds["waiting"]
    return train, validation


def window_starts(mask, half, known):
    """First step of each window lying fully in `mask`, in one half, with enough tracked ball."""
    ok = np.lib.stride_tricks.sliding_window_view(mask, WINDOW).all(axis=1)
    ok &= half[:len(ok)] == half[WINDOW - 1:]
    seen = np.convolve(known.astype(np.int32), np.ones(WINDOW, dtype=np.int32), "valid")
    return np.flatnonzero(ok & (seen >= MIN_KNOWN * WINDOW))


def window_around(first_step, last_step, half):
    """Start of the window centred on steps first_step..last_step, moved to stay inside its half."""
    start = (first_step + last_step + 1 - WINDOW) // 2
    in_half = np.flatnonzero(half == half[first_step])
    return int(np.clip(start, in_half[0], in_half[-1] + 1 - WINDOW))


def load():
    """The dataset as a dict of arrays."""
    return dict(np.load(OUT_PATH))


def check_folds(ds, hidden):
    """Stop if any step of a test gap could be trained on or validated on. Prints the split."""
    print("\nfold  test blocks  validation  train steps  windows  test gaps")
    for fold in range(FOLDS):
        train, val = split(ds, fold)
        count = 0
        for seconds in GAP_SECONDS:
            gaps = hidden["gap_{}".format(seconds)]
            mine = gaps[hidden["fold_{}".format(seconds)] == fold]
            count += len(mine)
            for a in mine:
                inside = (ds["frames"] >= a) & (ds["frames"] < a + seconds * FPS)
                assert inside.any() and not train[inside].any() and not val[inside].any()
        test, blocks = held_out_blocks(ds, fold)
        print("  {}       {}        {:>2d},{:>2d}      {:6d}    {:6d}     {:3d}".format(
            fold, len(test), blocks[0], blocks[1], train.sum(), len(window_starts(train, ds["half"], ds["known"])), count))
    print("checked: no step of any test gap is trained on or validated on by the model that tests it")


def check_windows(ds, hidden):
    """Does a centred window hold real ball on both sides of every test gap?"""
    frames, known, half = ds["frames"], ds["known"], ds["half"]
    print("\ngap    window has real ball      known share of the steps around the gap")
    for seconds in GAP_SECONDS:
        both, share = 0, []
        for a in hidden["gap_{}".format(seconds)]:
            i, j = np.searchsorted(frames, [a, a + seconds * FPS])
            s = window_around(i, j - 1, half)
            before, after = known[s:i], known[j:s + WINDOW]
            both += before.any() and after.any()
            share.append(np.r_[before, after].mean())
        print("{:2d} s   {:3d} of {:3d} on both sides    {:.0%}".format(
            seconds, both, len(hidden["gap_{}".format(seconds)]), np.mean(share)))


def report(ds, stats):
    """Totals only."""
    labels, near, scored, state = stats["labels_25"], stats["near_25"], stats["scored"], stats["state"]
    print("steps: {} at {} Hz, 30 s windows are {} steps".format(len(ds["frames"]), HZ, WINDOW))
    print("ball known on {:.1%} of steps; slots present per step: mean {:.1f}, max {}".format(
        ds["known"].mean(), ds["present"].sum(axis=1).mean(), ds["present"].sum(axis=1).max()))
    s = scored
    has = near[s] >= 0
    travelling = (labels[s] >= 0) & ~has
    print("\nwho has the ball, share of scored frames (25 fps):")
    print("  player within {} m       {:.1%}".format(HAS_BALL_M, has.mean()))
    print("  travelling to a receiver {:.1%}".format(travelling.mean()))
    print("  nobody / out             {:.1%}".format((labels[s] == NOBODY).mean()))
    print("  no label, ball missing   {:.1%}".format((labels[s] == UNKNOWN).mean()))
    print("  (ball missing or filled  {:.1%})".format((state[s] != SEEN).mean()))
    known = ds["label"] >= 0
    changes = (np.diff(ds["label"][known]) != 0).sum()
    print("holder changes at 10 Hz: {} (slot numbers, so a slot swap counts); "
          "{} steps lost a label because the receiver's slot was absent".format(changes, stats["dropped"]))


def main():
    ds, stats = build()
    np.savez_compressed(OUT_PATH, **ds)
    hidden = np.load(os.path.join(CACHE_DIR, "hidden_gaps.npz"))
    report(ds, stats)
    check_folds(ds, hidden)
    check_windows(ds, hidden)
    print("\nwrote cache/ball_dataset.npz ({:.0f} MB)".format(os.path.getsize(OUT_PATH) / 1e6))


if __name__ == "__main__":
    main()
