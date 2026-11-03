"""Phase 3, step 1: choose the hidden gap test, stretches where we know the ball and pretend not to.

    .venv\\Scripts\\python.exe v1/hidden_gaps.py

Needs cache/smooth.npz (v1/smooth.py) and cache/dead_events.csv (v1/dead_balls.py). Gaps of 2, 5,
10 and 20 s are drawn at random, with a fixed seed, from open play. Open play is decided by code,
from football sense, with no frame numbers:
  - the ball is seen and on the pitch, with CONTEXT seen before and after the gap;
  - the players are moving: their mean speed over a few seconds is OPEN_PLAYERS or more;
  - the ball is moving: not placed or held, QUIET_BALL or more over a few seconds;
  - not inside a stoppage from v1/dead_balls.py, and not before the 2nd-half kickoff (from its tag).

Each gap also gets a fold, from 5-minute blocks of match time. Neighbouring frames are almost
copies, so a model tested on a gap must not train on that block (step 3).

The answer key and the tags are checks here, never inputs. Writes cache/hidden_gaps.npz.
Prints totals only.
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from smooth import FPS, PITCH, SEEN, MISSING, runs  # noqa: E402
from dead_balls import TAG_LEAD, TAG_MATCH, off_pitch, read_labels, read_tags  # noqa: E402

# Params
CACHE_DIR = os.path.join(ROOT, "cache")
OUT_PATH = os.path.join(CACHE_DIR, "hidden_gaps.npz")
SEED = 2027
GAP_SECONDS = (2, 5, 10, 20)
MAX_PER_LENGTH = 200     # enough for a steady median; a gap of one length never overlaps another of it
CONTEXT = 1 * FPS        # seen ball kept on each side of a gap, also the spacing between gaps
OPEN_WINDOW = 3 * FPS    # the speeds below are means over this long
OPEN_PLAYERS = 1.5       # m/s: above a walk. Stopped play sits around or below 1.2 m/s
QUIET_BALL = 1.0         # m/s: slower than this the ball is being placed or held, not played
FOLD_BLOCK = 5 * 60 * FPS
FOLDS = 5
HUMAN_SPEED = 12.0       # m/s: nobody runs faster. Above it a track slot jumped to another player
NEAR_M = 5.0


def rolling_mean(v, width):
    """Centred mean of v over `width` frames, ignoring NaN."""
    ok = np.isfinite(v)
    w = np.ones(width)
    return np.convolve(np.where(ok, v, 0), w, "same") / np.maximum(np.convolve(ok, w, "same"), 1)


def second_half_wait(tags, scored):
    """(first, kickoff): the scored frames before the 2nd-half kickoff, found from its tag."""
    kickoff = next(t["start_frame"] for t in tags
                   if t["type"] == "Kick Off (Start)" and t["half"] == "2nd Half") + TAG_LEAD
    first = kickoff
    while scored[first - 1]:
        first -= 1
    return first, kickoff


def stoppage_mask(n):
    """True inside the stoppages of v1/dead_balls.py."""
    mask = np.zeros(n, dtype=bool)
    with open(os.path.join(CACHE_DIR, "dead_events.csv"), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "stoppage":
                mask[int(row["start"]):int(row["end"])] = True
    return mask


def open_play(sm, raw, scored, tags):
    """True on frames that can sit inside a test gap."""
    seen = sm["ball_state"] == SEEN
    players = rolling_mean(np.nanmean(sm["player_speed"], axis=1), OPEN_WINDOW)
    ball = rolling_mean(np.where(seen, sm["ball_speed"], np.nan), OPEN_WINDOW)
    keep = scored & seen & ~off_pitch(raw["x"], raw["y"]) & (players >= OPEN_PLAYERS) & (ball >= QUIET_BALL)
    keep &= ~stoppage_mask(len(scored))
    first, kickoff = second_half_wait(tags, scored)
    keep[first:kickoff] = False
    return keep


def choose(keep, length, rng):
    """Random gap starts of `length` frames, none overlapping, all with CONTEXT each side."""
    starts = [np.arange(s + CONTEXT, e - length - CONTEXT + 1) for s, e in runs(keep) if e - s >= length + 2 * CONTEXT]
    if not starts:
        return np.zeros(0, dtype=np.int64)
    taken = np.zeros(len(keep), dtype=bool)
    out = []
    for a in rng.permutation(np.concatenate(starts)):
        if not taken[a - CONTEXT:a + length + CONTEXT].any():
            taken[a:a + length] = True
            out.append(a)
            if len(out) == MAX_PER_LENGTH:
                break
    return np.sort(np.array(out, dtype=np.int64))


def fold_of(frames, scored):
    """Fold per frame, from 5-minute blocks of scored time."""
    elapsed = np.cumsum(scored) - 1
    return (elapsed[frames] // FOLD_BLOCK) % FOLDS


def moment(sm, raw, frame):
    """What the ball's surroundings look like at `frame`: nearest player m, players within NEAR_M, m to a line."""
    d = np.hypot(sm["player_x"][frame] - raw["x"][frame], sm["player_y"][frame] - raw["y"][frame])
    length, width = PITCH
    x, y = raw["x"][frame], raw["y"][frame]
    return np.nanmin(d), (d < NEAR_M).sum(), min(x, length - x, y, width - y)


def compare_with_real(sm, raw, scored, chosen):
    """Are the test gaps like the real ones? Compare the moment the ball disappears."""
    state = sm["ball_state"]
    real = [s - 1 for s, _ in runs((state == MISSING) & scored) if scored[s - 1] and state[s - 1] == SEEN]
    test = np.concatenate(list(chosen.values())) - 1
    rows = [("real gaps", real), ("test gaps", test)]
    print("\nwhen the ball disappears     nearest player m  players within 5 m  m to a line")
    for label, frames in rows:
        m = np.array([moment(sm, raw, f) for f in frames])
        print("  {:9s} ({:3d})              {:6.1f}          {:6.1f}         {:6.1f}   (medians)".format(
            label, len(frames), *np.median(m, axis=0)))


def checks(chosen, labels, tags):
    """Do the test gaps avoid stopped play? Checks only: nothing here feeds the choice."""
    restarts = np.array([t["start_frame"] + TAG_LEAD for t in tags if t["code"] == "SET PIECES"])
    in_key = labels != "X"
    print("\nchecks (not used to choose):")
    for seconds, starts in chosen.items():
        length = seconds * FPS
        hit_tag = sum(((restarts >= a) & (restarts < a + length)).any() for a in starts)
        key = [a for a in starts if in_key[a:a + length].all()]
        dead = sum((labels[a:a + length] == "D").mean() for a in key)
        print("  {:2d} s: {:3d} gaps, {:3d} hold a tagged restart; {:2d} inside the 5 minutes, {:.0%} of their frames are true D".format(
            seconds, len(starts), hit_tag, len(key), dead / max(1, len(key))))


def main():
    sm = dict(np.load(os.path.join(CACHE_DIR, "smooth.npz")))
    raw = np.load(os.path.join(CACHE_DIR, "ball.npz"))
    scored = np.load(os.path.join(CACHE_DIR, "scored.npy"))
    tags = read_tags()

    keep = open_play(sm, raw, scored, tags)
    rng = np.random.default_rng(SEED)
    chosen = {s: choose(keep, s * FPS, rng) for s in GAP_SECONDS}
    np.savez_compressed(OUT_PATH, **{"gap_{}".format(s): a for s, a in chosen.items()},
                        **{"fold_{}".format(s): fold_of(a, scored) for s, a in chosen.items()})

    # Totals only.
    print("open play frames that can hold a test gap: {:.1%} of scored".format(keep.sum() / scored.sum()))
    speed = sm["player_speed"][scored]
    print("track slot jumps: {:.3%} of player frames above {:.0f} m/s (baselines ignore these)".format(
        np.nanmean(speed[np.isfinite(speed)] > HUMAN_SPEED), HUMAN_SPEED))
    spans = np.array([e - s for s, e in runs(keep)]) / FPS
    print("unbroken open play stretches: {}, median {:.1f} s, longest {:.0f} s".format(len(spans), np.median(spans), spans.max()))
    for seconds, starts in chosen.items():
        folds = np.bincount(fold_of(starts, scored), minlength=FOLDS) if len(starts) else np.zeros(FOLDS, dtype=int)
        print("  {:2d} s gaps: {:3d}  (per fold: {})".format(seconds, len(starts), " ".join(str(f) for f in folds)))
    compare_with_real(sm, raw, scored, chosen)
    checks(chosen, read_labels(), tags)
    print("\nwrote cache/hidden_gaps.npz")


if __name__ == "__main__":
    main()
