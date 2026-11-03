"""Phase 3, step 2: find dead balls (label D) from ball and player positions.

    .venv\\Scripts\\python.exe v1/dead_balls.py
    .venv\\Scripts\\python.exe v1/dead_balls.py --filled     # also use the ball model's paths in the gaps
    .venv\\Scripts\\python.exe v1/dead_balls.py --cross-start  # as --filled, Start = predicted crossing frame
    .venv\\Scripts\\python.exe v1/dead_balls.py --cross-seen   # as --cross-start, also for balls that came back out

Needs cache/smooth.npz (v1/smooth.py). Every stoppage has a Start (ball goes out) and an End (ball
back in play), and every frame in between is D. Only the four types with fixed boundaries for now:
throw-in, corner, goal kick, goal. Fouls and offsides are step 3 (action plan, Phase 3, step 2).

Start:
  - ball seen past a line: Start = first frame out, confirmed if the ball restarts somewhere that fits.
    If it came back already out, it left during the gap: Start = last frame seen before the gap;
  - ball goes missing near a line: possible exit, Start = last seen frame. It needs the End check too.
    A ball that goes missing infield is left for step 3.
  - ball goes missing while moving fast: extend its last direction (trajectory). If that line hits
    a line of the pitch soon enough, it is a possible exit there, Start = last seen frame. The ball
    is usually lost well before the line (median 5.9 m), so the 0.5 m rule alone misses most of them.

End = first frame the ball moves in the expected direction from the expected spot:
    throw-in   sideline, near the exit, into the field
    corner     a flag on that byline, towards the box
    goal kick  inside the 6 yard box at that end, into the field
    goal       centre spot, outwards
The ball counts if it sat slow at the spot first, if it is first seen already moving from there
(we never saw it still: a little lag, the same lag every time), or if it comes in from outside the
lines (off the thrower's hands). An exit with no fitting restart never left: tracking error, not a
stoppage. Player shape is recorded as evidence for every type and required for goal kicks only: a
ball that reappears in the 6 yard box with no lone taker is play near the byline, not a goal kick.
It is also required for any exit found by trajectory: that is weaker evidence than a ball seen leaving.
Numbers come from football sense. The tags and the answer key are checks only.

The 2nd half's scored frames before its kickoff are D (FAQ.md).

--filled also reads the ball model's paths in the gaps (cache/ball_predicted_blend.npz), for location only:
  - which line: if the predicted ball goes past a line during a gap, that is the line it left by and the
    exit point. A gap that starts infield with a predicted exit is a new possible exit, "predicted", and
    needs player shape like a trajectory exit. Start stays the last seen frame, as for every unseen exit
    (--cross-start: the first frame the predicted ball is past the line instead; writes ..._filled_cross;
    --cross-seen: the same, and also for a ball that came back already out; writes ..._filled_cross_seen);
  - where it reappears: a ball first seen moving the right way but too far from the spot still counts
    if the predicted ball was at the spot in the last FILLED_LOOKBACK before it was seen ("filled spot").
Standing still, speed and direction stay on real frames only. A stoppage that leans on a gap over
LONG_GAP_S is flagged: the model is clearly worse there. Writes cache/dead_filled.npy and
cache/dead_events_filled.csv instead, so the plain run (which v1/hidden_gaps.py reads) is not changed.

Writes cache/dead.npy (bool per frame) and cache/dead_events.csv. Prints totals only.
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from smooth import FPS, PITCH, SEEN, MISSING, runs  # noqa: E402

# Params
CACHE_DIR = os.path.join(ROOT, "cache")
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
SECOND_HALF_WAIT = (110808, 113742)  # scored frames before the 2nd-half kickoff, all D (FAQ.md)

# Pitch, metres. Spots are a metre or so generous: the tracking is not that precise.
OUT_M = 0.11             # ball centre this far past the line = the whole ball is over it
NEAR_LINE_M = 0.5        # ball last seen this close to a line may have gone out unseen
TRAJ_WINDOW = FPS        # heading and speed = travel over the last second seen. The very last frame is
                         # no guide: the ball is often nearly still there (median 1 m/s)
TRAJ_MIN_SPEED = 6.0     # m/s: a pass or clearance. A dribble (3-5 m/s) has no useful direction
TRAJ_MAX_S = 4.0         # s of flight at the last speed: a ball in the air for longer was shot or hoofed
TRAJ_MAX_M = 35.0        # m: no ball keeps its pace and heading further than this
GOAL_HALF_M = 3.66       # half the goal's width
SIX_YARD = (6.5, 10.16)  # 5.5 x 9.16 m plus 1 m: depth from the byline, half-width from the centre line
PENALTY_BOX = (16.5, 20.16)
PENALTY_SPOT_M = 11.0
CORNER_M = 3.0           # from the flag
THROW_LINE_M = 2.0       # from the sideline, either side: the thrower stands behind it
THROW_ALONG_M = 10.0     # along the sideline from where the ball went out
CENTRE_M = 2.0           # from the centre spot
SIGHT_EXTRA_M = 5.0      # a ball first seen already moving has left its spot by up to this much

# Ball movement.
STILL_SPEED = 2.5        # m/s: walking pace or slower = placed, held or carried
STILL_FOR = FPS // 2     # slow at the spot for this long, then ...
KICK_SPEED = 5.0         # m/s: ... this fast means the restart was taken
LOOK = 6                 # frames ahead used to judge direction and speed
REAL_KICK = 8.0          # m/s: slower than this between two stoppages is no real kick

# Time.
SEARCH = 60 * FPS        # look this long for a restart after an exit
GOAL_SEARCH = 180 * FPS  # goals take longer: celebrations, VAR
MIN_STOP = 3 * FPS       # a shorter "stoppage" is the tracking touching a line, not a real exit
SHAPE_REQUIRED = ("goal kick",)
MIN_STOP_BY_KIND = {"throw-in": 4 * FPS, "goal kick": 6 * FPS}  # football sense: the ball is fetched, the taker
                         # walks up. Anything quicker is play carried along a line, not a restart
MERGE_GAP = 3 * FPS      # two stoppages this close with no real kick between them are one

# Filled ball (--filled).
FILLED_PATH = os.path.join(CACHE_DIR, "ball_predicted_blend.npz")
FILLED_LOOKBACK = 2 * FPS  # a restart taken unseen is usually spotted within a couple of seconds (one long throw)
LONG_GAP_S = 30          # past the longest gap the model was tested on (20 s) and then some: low confidence

TAG_LEAD = int(3.5 * FPS)
TAG_MATCH = 10 * FPS     # a tag agrees if our restart falls within this after the tag opens
TAG_TYPES = {"Throw-in": "throw-in", "Corner Kick": "corner", "Goal Kick": "goal kick",
             "Kick Off (After Goal)": "goal"}
KINDS = ("throw-in", "corner", "goal kick", "goal")


def read_tags():
    """Smart Tagging rows as dicts, frames as ints."""
    with open(os.path.join(CACHE_DIR, "tags.csv"), newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["start_frame"], row["end_frame"] = int(row["start_frame"]), int(row["end_frame"])
    return rows


def read_labels(path=GT_PATH):
    """The label column of a frame,label file."""
    with open(path, newline="", encoding="utf-8") as handle:
        return np.array([row["label"] for row in csv.DictReader(handle)])


def off_pitch(x, y):
    """Is the whole ball over a line?"""
    length, width = PITCH
    return (x < -OUT_M) | (x > length + OUT_M) | (y < -OUT_M) | (y > width + OUT_M)


def line_crossed(x, y):
    """Which line a position is past or nearest to: (axis, value), axis "x" = byline, "y" = sideline."""
    length, width = PITCH
    nearest_x, nearest_y = (0.0 if x < length / 2 else length), (0.0 if y < width / 2 else width)
    return ("x", nearest_x) if abs(x - nearest_x) < abs(y - nearest_y) else ("y", nearest_y)


def trajectory_exit(x, y, vx, vy, gap_frames):
    """Where the ball would cross a line if it kept its last heading: (axis, value, cx, cy) or None."""
    length, width = PITCH
    speed = np.hypot(vx, vy)
    if not np.isfinite(speed) or speed < TRAJ_MIN_SPEED:
        return None
    hits = []
    if vx != 0:
        line = length if vx > 0 else 0.0
        t = (line - x) / vx
        hits.append((t, "x", line, line, y + vy * t))
    if vy != 0:
        line = width if vy > 0 else 0.0
        t = (line - y) / vy
        hits.append((t, "y", line, x + vx * t, line))
    t, axis, value, cx, cy = min(hits)
    inside = -1.0 <= cx <= length + 1.0 and -1.0 <= cy <= width + 1.0
    if t <= 0 or t > TRAJ_MAX_S or speed * t > TRAJ_MAX_M or not inside or gap_frames / FPS < 0.5 * t:
        return None
    return axis, value, cx, cy


def predicted_exit(start, end, filled):
    """First predicted ball past a line in the gap [start, end): (line, x, y, frame) or None."""
    fx, fy = filled["x"][start:end], filled["y"][start:end]
    out = np.flatnonzero(off_pitch(fx, fy))
    if not len(out):
        return None
    k = out[0]
    return line_crossed(fx[k], fy[k]), fx[k], fy[k], start + k


def exit_candidates(x, y, state, scored, filled=None):
    """Possible exits in time order: {start, how, line, x, y}, plus {filled_line, gap_s} with --filled."""
    seen = state == SEEN
    past = seen & off_pitch(x, y)
    out = []
    for s, _ in runs(past & scored):
        start, extra = s, {}
        if state[s - 1] == MISSING:  # came back already out: it left during the gap
            while state[start - 1] == MISSING:
                start -= 1
            start -= 1
            guess = predicted_exit(start + 1, s, filled) if filled is not None and filled["cross_seen"] else None
            if guess is not None:  # the line stays the one it was seen past; only Start moves
                start, extra = guess[3], {"filled_start": True, "gap_s": (s - start - 1) / FPS}  # gap_s from the old start
        out.append({"start": start, "how": "seen", "line": line_crossed(x[s], y[s]), "x": x[s], "y": y[s], **extra})
    for start, end in runs((state == MISSING) & scored):
        last = start - 1
        if not (scored[last] and seen[last]) or past[last]:
            continue
        guess = predicted_exit(start, end, filled) if filled is not None else None
        extra = {"filled_line": True, "gap_s": (end - start) / FPS} if guess is not None else {}
        at = guess[3] if guess is not None and filled["cross_start"] else last
        line = line_crossed(x[last], y[last])
        if abs((x[last] if line[0] == "x" else y[last]) - line[1]) <= NEAR_LINE_M:
            if guess is not None:
                out.append({"start": at, "how": "missing", "line": guess[0], "x": guess[1], "y": guess[2], **extra})
            else:
                out.append({"start": last, "how": "missing", "line": line, "x": x[last], "y": y[last]})
            continue
        first = last - TRAJ_WINDOW
        hit = None
        if first >= 0 and (seen[first:last + 1]).all():
            span = TRAJ_WINDOW / FPS
            hit = trajectory_exit(x[last], y[last], (x[last] - x[first]) / span, (y[last] - y[first]) / span, end - start)
        if hit is not None and guess is not None:
            out.append({"start": at, "how": "trajectory", "line": guess[0], "x": guess[1], "y": guess[2], **extra})
        elif hit is not None:
            out.append({"start": last, "how": "trajectory", "line": hit[:2], "x": hit[2], "y": hit[3]})
        elif guess is not None:
            out.append({"start": at, "how": "predicted", "line": guess[0], "x": guess[1], "y": guess[2], **extra})
    return sorted(out, key=lambda c: c["start"])


def spot_checks(kind, cand):
    """Two functions for one restart type: at_spot(x, y, extra) and fits(x0, y0, x1, y1)."""
    length, width = PITCH
    value = cand["line"][1]
    if kind == "throw-in":
        inward = 1 if value == 0 else -1
        return (lambda x, y, extra: abs(y - value) <= THROW_LINE_M + extra and abs(x - cand["x"]) <= THROW_ALONG_M + extra,
                lambda x0, y0, x1, y1: (y1 - y0) * inward > 0)
    if kind == "corner":
        flags = [(value, 0.0), (value, width)]
        inward = 1 if value == 0 else -1
        spot = (value + inward * PENALTY_SPOT_M, width / 2)

        def at_flag(x, y, extra):
            return min(np.hypot(x - fx, y - fy) for fx, fy in flags) <= CORNER_M + extra

        def towards_box(x0, y0, x1, y1):
            fx, fy = min(flags, key=lambda f: np.hypot(x0 - f[0], y0 - f[1]))
            return (x1 - x0) * (spot[0] - fx) + (y1 - y0) * (spot[1] - fy) > 0
        return at_flag, towards_box
    if kind == "goal kick":
        inward = 1 if value == 0 else -1
        return (lambda x, y, extra: abs(x - value) <= SIX_YARD[0] + extra and abs(y - width / 2) <= SIX_YARD[1] + extra,
                lambda x0, y0, x1, y1: (x1 - x0) * inward > 0)
    centre = (length / 2, width / 2)  # goal: kickoff
    return (lambda x, y, extra: np.hypot(x - centre[0], y - centre[1]) <= CENTRE_M + extra,
            lambda x0, y0, x1, y1: np.hypot(x1 - centre[0], y1 - centre[1]) > np.hypot(x0 - centre[0], y0 - centre[1]))


def was_at_spot(at_spot, i, filled):
    """Was the predicted ball at the spot in the FILLED_LOOKBACK before frame i?"""
    for k in range(i - 1, max(0, i - FILLED_LOOKBACK) - 1, -1):
        if not filled["predicted"][k]:
            return False
        if at_spot(filled["x"][k], filled["y"][k], 1.0):
            return True
    return False


def find_end(cand, kinds, search, x, y, state, scored, filled=None):
    """First frame the ball restarts for one of `kinds`. Returns (frame, kind, how) or None."""
    checks = {k: spot_checks(k, cand) for k in kinds}
    still = {k: 0 for k in kinds}
    speed_needed = KICK_SPEED * LOOK / FPS
    stop = min(cand["start"] + search, len(x) - LOOK)
    for i in range(cand["start"] + 1, stop):
        if not scored[i]:
            return None
        if state[i] != SEEN:
            still = {k: 0 if still[k] < STILL_FOR else still[k] for k in kinds}  # placed stays placed
            continue
        first_sight = state[i - 1] != SEEN
        j = i + LOOK
        ahead = (state[i:j + 1] == SEEN).all()
        moved = np.hypot(x[j] - x[i], y[j] - y[i]) if ahead else 0.0
        for kind in kinds:
            at_spot, fits = checks[kind]
            direction_ok = moved >= speed_needed and fits(x[i], y[i], x[j], y[j])
            if direction_ok and still[kind] >= STILL_FOR and at_spot(x[i], y[i], 1.0):
                return i, kind, "placed"
            if direction_ok and first_sight and at_spot(x[i], y[i], SIGHT_EXTRA_M):
                return i, kind, "first sight"
            if direction_ok and off_pitch(x[i], y[i]) and at_spot(x[i], y[i], 1.0) and i - cand["start"] >= MIN_STOP:
                return i, kind, "from outside"
            if direction_ok and first_sight and filled is not None and was_at_spot(at_spot, i, filled):
                return i, kind, "filled spot"
            if i + 1 < len(x) and state[i + 1] == SEEN and at_spot(x[i], y[i], 0.0) \
                    and np.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) * FPS < STILL_SPEED:
                still[kind] += 1
            elif still[kind] < STILL_FOR:
                still[kind] = 0
    return None


def restart_of(cand, x, y, state, scored, filled=None):
    """Try the restarts that fit where the ball went out. A ball out through the goal is a goal
    only if play restarts at the centre spot; otherwise it went over the bar or was tracked wrong."""
    axis, _ = cand["line"]
    if axis == "y":
        return find_end(cand, ("throw-in",), SEARCH, x, y, state, scored, filled)
    if abs(cand["y"] - PITCH[1] / 2) <= GOAL_HALF_M:
        goal = find_end(cand, ("goal",), GOAL_SEARCH, x, y, state, scored, filled)
        if goal is not None:
            return goal
    return find_end(cand, ("corner", "goal kick"), SEARCH, x, y, state, scored, filled)


def on_the_ball(frame, sm):
    """Team letter of the player nearest the ball at `frame`."""
    d = np.hypot(sm["player_x"][frame] - sm["ball_x"][frame], sm["player_y"][frame] - sm["ball_y"][frame])
    return "HA"[sm["team"][np.nanargmin(d)]] if np.isfinite(d).any() else "?"


def shape_fits(kind, frame, cand, sm):
    """Do the players stand as this restart expects? Evidence only."""
    length, width = PITCH
    px, py, team = sm["player_x"][frame], sm["player_y"][frame], sm["team"]
    ok = np.isfinite(px)
    px, py, team = px[ok], py[ok], team[ok]
    if not ok.any():
        return False
    d = np.sort(np.hypot(px - sm["ball_x"][frame], py - sm["ball_y"][frame]))
    if kind == "corner":
        in_box = (np.abs(px - cand["line"][1]) < PENALTY_BOX[0]) & (np.abs(py - width / 2) < PENALTY_BOX[1])
        return bool(in_box.sum() >= 6)
    if kind == "goal kick":
        return bool(d[0] < 3 and (len(d) < 2 or d[1] > 5))
    if kind == "throw-in":
        return bool(d[0] < 3 and (d < 15).sum() >= 3)
    left = [(px[team == t] < length / 2).mean() for t in (0, 1) if (team == t).any()]  # goal: kickoff
    return len(left) == 2 and min(max(left), 1 - min(left)) >= 0.8


def find_stoppages(sm, raw, scored, filled=None):
    """Walk through the exits in order. Returns one row per exit looked at."""
    x, y = raw["x"].astype(np.float64), raw["y"].astype(np.float64)
    state = sm["ball_state"]
    rows, busy_until = [], 0
    for cand in exit_candidates(x, y, state, scored, filled):
        if cand["start"] < busy_until or SECOND_HALF_WAIT[0] <= cand["start"] < SECOND_HALF_WAIT[1]:
            continue
        row = {"start": cand["start"], "how": cand["how"], "seen_leaving": cand["how"] == "seen", "side": cand["line"][0],
               "status": "no restart", "kind": "", "end": "", "end_how": "", "team": "", "shape_ok": ""}
        if filled is not None:
            row.update(filled_line=cand.get("filled_line", False), filled_end=False, gap_s=cand.get("gap_s", 0.0))
            if filled["cross_seen"]:
                row["filled_start"] = cand.get("filled_start", False)
        found = restart_of(cand, x, y, state, scored, filled)
        if found is not None:
            end, kind, how = found
            row.update(status="too short", kind=kind, end=end, end_how=how)
            if how == "filled spot":
                row.update(filled_end=True, gap_s=max(row["gap_s"], filled["gap_s"][end - 1]))
            row["shape_ok"] = shape_fits(kind, end, cand, sm)
            needs_shape = kind in SHAPE_REQUIRED or cand["how"] in ("trajectory", "predicted")
            if end - cand["start"] >= MIN_STOP_BY_KIND.get(kind, MIN_STOP) and (row["shape_ok"] or not needs_shape):
                row.update(status="stoppage", team=on_the_ball(end, sm))
                busy_until = end
            elif end - cand["start"] >= MIN_STOP_BY_KIND.get(kind, MIN_STOP):
                row["status"] = "wrong shape"
        if filled is not None:
            row["long_gap"] = row["gap_s"] > LONG_GAP_S
        rows.append(row)
    return rows


def dead_mask(rows, sm, scored):
    """D frames from the stoppages, joining two with no real kick between them."""
    dead = np.zeros(len(scored), dtype=bool)
    stops = [r for r in rows if r["status"] == "stoppage"]
    for r in stops:
        dead[r["start"]:r["end"]] = True
    joined = 0
    for a, b in zip(stops, stops[1:]):
        speed = sm["ball_speed"][a["end"]:b["start"] + 1]
        kicked = np.isfinite(speed).any() and np.nanmax(speed) >= REAL_KICK
        if b["start"] - a["end"] <= MERGE_GAP and not kicked:
            dead[a["end"]:b["start"]] = True
            joined += 1
    dead[SECOND_HALF_WAIT[0]:SECOND_HALF_WAIT[1]] = True
    return dead & scored, joined


def check_tags(rows, tags):
    """Agreement per type: same type of ours restarting within TAG_MATCH after the tag opens."""
    stops = [r for r in rows if r["status"] == "stoppage"]
    print("\ncross check with the tags (tags are a hint, not the truth):")
    for tag_type, kind in TAG_TYPES.items():
        ours = np.array([r["end"] for r in stops if r["kind"] == kind])
        sp = [t for t in tags if t["code"] == "SET PIECES" and t["type"] == tag_type]
        agree = [t for t in sp if ((ours >= t["start_frame"]) & (ours <= t["start_frame"] + TAG_MATCH)).any()]
        either = np.array([r["end"] for r in stops])
        any_kind = sum(((either >= t["start_frame"]) & (either <= t["start_frame"] + TAG_MATCH)).any() for t in sp)
        print("  {:10s} tags {:3d}, ours {:3d}: same type {:3d} ({:.0%}), any type {:3d}".format(
            kind, len(sp), len(ours), len(agree), len(agree) / max(1, len(sp)), any_kind))
    same = total = 0
    for r in stops:
        for t in tags:
            if t["code"] == "SET PIECES" and t["team"] in ("H", "A") and TAG_TYPES.get(t["type"]) == r["kind"] \
                    and t["start_frame"] <= r["end"] <= t["start_frame"] + TAG_MATCH:
                total += 1
                same += r["team"] == t["team"]
                break
    print("  restarting team agrees with the tag: {} of {}".format(same, total))


def check_public(dead, rows):
    """Compare with the 5-minute answer key. A direction check: never tune against it."""
    labels = read_labels()
    window = labels != "X"
    truth = (labels == "D") & window
    ours = dead & window
    hit = (truth & ours).sum()
    print("\npublic 5 minutes: {} true D frames, we mark {}, overlap {}".format(truth.sum(), ours.sum(), hit))
    print("  share of true D we catch: {:.0%}   share of our D that is right: {:.0%}".format(
        hit / truth.sum(), hit / max(1, ours.sum())))
    for number, (first, last) in enumerate(runs(truth), 1):
        here = [r for r in rows if r["status"] == "stoppage" and r["start"] < last and r["end"] > first]
        what = "; ours: {} ({:+.1f} s at start, {:+.1f} s at end)".format(
            "+".join(r["kind"] for r in here), (here[0]["start"] - first) / FPS, (here[-1]["end"] - last) / FPS) if here else ""
        print("  true stoppage {} ({:.1f} s): {:.0%} covered{}".format(number, (last - first) / FPS, dead[first:last].mean(), what))
    invented = [(s, e) for s, e in runs(ours) if not truth[s:e].any()]
    print("  stoppages we invent: {} ({:.1f} s in total)".format(len(invented), sum(e - s for s, e in invented) / FPS))
    in_key = [r for r in rows if r["status"] == "stoppage" and window[r["end"]] and labels[r["end"] + FPS] in ("H", "A")]
    print("  restarting team right, 1 s after our End: {} of {}".format(
        sum(r["team"] == labels[r["end"] + FPS] for r in in_key), len(in_key)))


def load_filled(cross_start=False, cross_seen=False):
    """The blended ball paths, plus the length in seconds of the gap each predicted frame sits in."""
    blend = np.load(FILLED_PATH)
    gap_s = np.zeros(len(blend["x"]))
    for a, b in zip(blend["gap_start"], blend["gap_end"]):
        gap_s[a:b] = (b - a) / FPS
    return {"x": blend["x"].astype(np.float64), "y": blend["y"].astype(np.float64),
            "predicted": blend["predicted"], "gap_s": gap_s, "cross_start": cross_start,
            "cross_seen": cross_seen}


def main():
    sm = dict(np.load(os.path.join(CACHE_DIR, "smooth.npz")))
    raw = np.load(os.path.join(CACHE_DIR, "ball.npz"))
    scored = np.load(os.path.join(CACHE_DIR, "scored.npy"))
    cross_seen = "--cross-seen" in sys.argv
    cross = "--cross-start" in sys.argv or cross_seen
    filled = load_filled(cross, cross_seen) if "--filled" in sys.argv or cross else None
    suffix = "_filled_cross_seen" if cross_seen else "_filled_cross" if cross else "_filled" if filled is not None else ""

    rows = find_stoppages(sm, raw, scored, filled)
    dead, joined = dead_mask(rows, sm, scored)
    np.save(os.path.join(CACHE_DIR, "dead{}.npy".format(suffix)), dead)
    with open(os.path.join(CACHE_DIR, "dead_events{}.csv".format(suffix)), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Totals only.
    stops = [r for r in rows if r["status"] == "stoppage"]
    for how, label in (("seen", "seen leaving"), ("missing", "gone missing near a line"), ("trajectory", "gone missing, trajectory"),
                       ("predicted", "gone missing, predicted ball out")):
        some = [r for r in rows if r["how"] == how]
        print("exits {}: {} looked at -> {} stoppages, {} never left (no fitting restart), {} too short or wrong shape".format(
            label, len(some), sum(r["status"] == "stoppage" for r in some),
            sum(r["status"] == "no restart" for r in some), sum(r["status"] in ("too short", "wrong shape") for r in some)))
    print("\ntype        count  median s  shape fits  End placed/first sight/from outside/filled spot")
    for kind in KINDS:
        some = [r for r in stops if r["kind"] == kind]
        if some:
            print("{:10s}  {:5d}  {:8.1f}  {:5d}/{:<4d}  {}/{}/{}/{}".format(
                kind, len(some), np.median([r["end"] - r["start"] for r in some]) / FPS,
                sum(r["shape_ok"] for r in some), len(some),
                *[sum(r["end_how"] == h for r in some) for h in ("placed", "first sight", "from outside", "filled spot")]))
    if filled is not None:
        uses = [r["filled_line"] or r["filled_end"] or r.get("filled_start", False) for r in stops]
        print("\nstoppages using the filled ball: line {}, restart spot {}, start {}; leaning on a gap over {} s: {}".format(
            sum(r["filled_line"] for r in stops), sum(r["filled_end"] for r in stops),
            sum(r.get("filled_start", False) for r in stops), LONG_GAP_S,
            sum(r["long_gap"] and used for r, used in zip(stops, uses))))
    print("\nstoppages joined: {}; dead time {:.1f} min = {:.0%} of scored frames (includes the 2nd-half wait)".format(
        joined, dead.sum() / FPS / 60, dead.sum() / scored.sum()))
    check_tags(rows, read_tags())
    check_public(dead, rows)


if __name__ == "__main__":
    main()
