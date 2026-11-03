"""Stage 0: turn the 7 IDSSE matches into our cache format, plus labels H/A/D per frame.

    .venv\\Scripts\\python.exe v2/sportec_cache.py [MATCH ...]

Needs data/sportec/ (v2/sportec_download.py). Reads the XML directly with lxml: kloppy's loader
does the same work much slower and keeps BallStatus behind its own names.

Writes cache/sportec/<match>/ (gitignored):
  ball.npz      x, y, z, speed, visible          metres, our orientation (see below)
  players.npz   x, y, speed (frames, slots), team (0 home, 1 away), person
  scored.npy    True on every frame except the one-frame separator between halves
  half.npy      1, 2 (0 on the separator)
  labels.npy    0 H, 1 A, 2 D, 3 X (the scorer's codes). BallStatus 0 = D, else BallPossession 1 = H, 2 = A
  pitch.npy     (length, width) in metres
  events.csv    every event: kind, sub, team (H/A/blank), frame, half, x, y, set piece flag

Orientation: our cache is Metrica's, origin top-left, x along the pitch, y down. Sportec tracking
has its origin at the centre with y up, its events the origin bottom-left with y up. Both are moved
to ours. Row layout: 1st half frames, one separator frame (label X, not scored), 2nd half frames.
The separator makes the scorer treat the halves as two runs, as the halftime hole does for ours.
"""

import csv
import datetime as dt
import os
import sys

import numpy as np
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from sportec_download import MATCHES, path_of  # noqa: E402

# Params
FPS = 25
OUT_ROOT = os.path.join(ROOT, "cache", "sportec")
SECTIONS = {"firstHalf": 1, "secondHalf": 2}
SET_PIECES = ("KickOff", "ThrowIn", "CornerKick", "GoalKick", "FreeKick", "Penalty")
H, A, D, X = 0, 1, 2, 3


def parse_time(text):
    return dt.datetime.fromisoformat(text).timestamp()


def read_meta(match):
    """(pitch (L, W), home team id, away team id)."""
    root = etree.parse(path_of(match, "meta")).getroot()
    env = root.find(".//Environment")
    pitch = (float(env.get("PitchX")), float(env.get("PitchY")))
    teams = {t.get("Role"): t.get("TeamId") for t in root.iter("Team")}
    return pitch, teams["home"], teams["guest"]


def read_tracking(match):
    """{half: {person: record}}; the ball record also has Z, possession, status and its first time."""
    out = {1: {}, 2: {}}
    for _, fs in etree.iterparse(path_of(match, "tracking"), tag="FrameSet"):
        half = SECTIONS.get(fs.get("GameSection"))
        if half is None:  # extra time: none in this dataset, but never mix it in
            fs.clear()
            continue
        frames = fs.findall("Frame")

        def get(k, f=float):
            return np.array([f(fr.get(k, "nan")) for fr in frames])

        rec = {"team": fs.get("TeamId"), "N": get("N", int), "X": get("X"), "Y": get("Y")}
        if rec["team"] == "BALL":
            rec.update(Z=get("Z"), poss=get("BallPossession"), status=get("BallStatus"),
                       T0=parse_time(frames[0].get("T")))
        out[half][fs.get("PersonId")] = rec
        fs.clear()
    return out


def speed(x, y):
    v = np.full(x.shape, np.nan, dtype=np.float32)
    v[1:] = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)) * FPS
    return v


def build(match):
    pitch, home, away = read_meta(match)
    L, W = pitch
    track = read_tracking(match)
    team_of = {}
    for h in track.values():
        for p, r in h.items():
            if r["team"] in (home, away):  # referees are tracked too: leave them out
                team_of[p] = 0 if r["team"] == home else 1
    persons = sorted(team_of)
    slot = {p: i for i, p in enumerate(persons)}

    # Frame index per half from the ball's frame numbers; one separator frame between halves.
    balls = {h: next(r for r in track[h].values() if r["team"] == "BALL") for h in (1, 2)}
    first = {h: int(balls[h]["N"].min()) for h in (1, 2)}
    length = {h: int(balls[h]["N"].max()) - first[h] + 1 for h in (1, 2)}
    start = {1: 0, 2: length[1] + 1}
    n = length[1] + 1 + length[2]

    bx, by, bz = (np.full(n, np.nan) for _ in range(3))
    labels = np.full(n, X, dtype=np.int8)
    half = np.zeros(n, dtype=np.int8)
    px = np.full((n, len(persons)), np.nan, dtype=np.float32)
    py = px.copy()
    t0 = {}
    for h in (1, 2):
        half[start[h]:start[h] + length[h]] = h
        b = balls[h]
        i = start[h] + b["N"] - first[h]
        bx[i], by[i], bz[i] = b["X"] + L / 2, W / 2 - b["Y"], b["Z"]
        labels[i] = np.where(b["status"] == 0, D, np.where(b["poss"] == 1, H, np.where(b["poss"] == 2, A, X)))
        t0[h] = b["T0"]
        for p, r in track[h].items():
            if p not in slot:
                continue
            j = start[h] + r["N"] - first[h]
            ok = (j >= start[h]) & (j < start[h] + length[h])
            px[j[ok], slot[p]] = r["X"][ok] + L / 2
            py[j[ok], slot[p]] = W / 2 - r["Y"][ok]

    scored = half > 0
    out = os.path.join(OUT_ROOT, match)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "ball.npz"), x=bx.astype(np.float32), y=by.astype(np.float32),
                        z=bz.astype(np.float32), speed=speed(bx, by), visible=~np.isnan(bx))
    np.savez_compressed(os.path.join(out, "players.npz"), x=px, y=py, speed=speed(px, py),
                        team=np.array([team_of[p] for p in persons], dtype=np.int8),
                        person=np.array(persons))
    np.save(os.path.join(out, "scored.npy"), scored)
    np.save(os.path.join(out, "half.npy"), half)
    np.save(os.path.join(out, "labels.npy"), labels)
    np.save(os.path.join(out, "pitch.npy"), np.array(pitch))
    n_events = write_events(match, out, home, away, pitch, t0, start, length)

    lab = labels[scored]
    print("{} pitch {}x{}  frames {}  slots {}  players/frame {:.1f}  ball seen {:.1%}  "
          "H {:.1%} A {:.1%} D {:.1%} X {:.2%}  events {}".format(
              match, L, W, n, len(persons), (~np.isnan(px[scored])).sum(1).mean(),
              (~np.isnan(bx[scored])).mean(), *[(lab == k).mean() for k in (H, A, D, X)], n_events),
          flush=True)


def write_events(match, out, home, away, pitch, t0, start, length):
    """Every event with its tracking frame (by wall clock), team and position in our orientation."""
    L, W = pitch
    rows = []
    for _, ev in etree.iterparse(path_of(match, "event"), tag="Event"):
        if ev.get("EventTime") is None or len(ev) == 0:
            ev.clear()
            continue
        child = ev[0]
        t = parse_time(ev.get("EventTime"))
        team = child.get("Team") or child.get("WinnerTeam") or next(
            (c.get("Team") for c in child.iter() if c.get("Team")), None)
        # The half whose clock window holds the event (events carry no section, set pieces do).
        section = child.get("GameSection")
        h = SECTIONS.get(section) or next(
            (h for h in (2, 1) if (t - t0[h]) * FPS >= -60 * FPS), None)
        if h is None:
            ev.clear()
            continue
        f = start[h] + int(round((t - t0[h]) * FPS))
        x, y = ev.get("X-Position"), ev.get("Y-Position")
        rows.append({"kind": child.tag, "sub": next((c.tag for c in child), ""),
                     "team": "H" if team == home else "A" if team == away else "",
                     "frame": f, "half": h, "in_half": int(start[h] <= f < start[h] + length[h]),
                     "x": float(x) if x else "", "y": W - float(y) if y else "",
                     "set_piece": int(child.tag in SET_PIECES)})
        ev.clear()
    rows.sort(key=lambda r: r["frame"])
    with open(os.path.join(out, "events.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    for m in (sys.argv[1:] or MATCHES):
        build(m)
