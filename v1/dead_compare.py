"""Compare the dead ball run without and with the filled ball (v1/dead_balls.py and --filled).

    .venv\\Scripts\\python.exe v1/dead_compare.py
    .venv\\Scripts\\python.exe v1/dead_compare.py _filled _filled_cross   # any two runs, by file suffix

Needs cache/dead.npy, cache/dead_events.csv, cache/dead_filled.npy, cache/dead_events_filled.csv.
Prints a before / after table and every stoppage that appeared, vanished or changed: Start and End as
match clock (from each half's Kick Off (Start) tag, so within a few frames; the 2nd half counts on from
45:00, and the 1st half runs past 45:00 in added time) and as time into BAR-ATH_tactical.mp4, then type,
length and how it was found. Totals and our own events only.
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from dead_balls import KINDS, TAG_LEAD, TAG_MATCH, TAG_TYPES, read_labels, read_tags  # noqa: E402
from smooth import FPS  # noqa: E402

# Params
CACHE_DIR = os.path.join(ROOT, "cache")
RUNS = (("before", ""), ("after", "_filled"))  # or pass two file suffixes, e.g. _filled _filled_cross


def load(suffix):
    """Stoppage rows (frames as ints) and the D mask of one run."""
    with open(os.path.join(CACHE_DIR, "dead_events{}.csv".format(suffix)), newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["status"] == "stoppage"]
    for r in rows:
        r["start"], r["end"] = int(r["start"]), int(r["end"])
    return rows, np.load(os.path.join(CACHE_DIR, "dead{}.npy".format(suffix)))


def kickoffs(tags):
    """[(kickoff frame, clock minutes at kickoff)] per half, from the Kick Off (Start) tags."""
    starts = sorted(t["start_frame"] + TAG_LEAD for t in tags if t["type"] == "Kick Off (Start)")
    return [(frame, 45 * half) for half, frame in enumerate(starts)]


def mmss(seconds):
    seconds = int(seconds)
    return "{}:{:02d}".format(seconds // 60, seconds % 60)


def clock(frame, kicks):
    """Match clock (as on the broadcast) and time into the video."""
    kick, minutes = [k for k in kicks if frame >= k[0]][-1] if frame >= kicks[0][0] else kicks[0]
    played, full = (frame - kick) / FPS, 45 * 60
    match = mmss(minutes * 60 + played) if played < full else "{}+{}".format(minutes + 45, mmss(played - full))
    return "{:>7s} (video {})".format(match, mmss(frame / FPS))


def numbers(rows, dead, labels, tags):
    """One column of the table."""
    window = labels != "X"
    truth = (labels == "D") & window
    hit = (truth & dead).sum()
    out = {"answer key: true D caught": "{:.0%}".format(hit / truth.sum()),
           "answer key: our D right": "{:.0%}".format(hit / max(1, (dead & window).sum()))}
    for tag_type, kind in TAG_TYPES.items():
        ours = np.array([r["end"] for r in rows if r["kind"] == kind])
        sp = [t for t in tags if t["code"] == "SET PIECES" and t["type"] == tag_type]
        agree = sum(((ours >= t["start_frame"]) & (ours <= t["start_frame"] + TAG_MATCH)).any() for t in sp)
        out["tags matched: " + kind] = "{} of {}".format(agree, len(sp))
    for kind in KINDS:
        out["stoppages: " + kind] = str(sum(r["kind"] == kind for r in rows))
    out["stoppages: total"] = str(len(rows))
    out["dead time, min"] = "{:.1f}".format(dead.sum() / FPS / 60)
    return out


def describe(r, kicks):
    filled = [w for w, k in (("filled line", "filled_line"), ("filled spot", "filled_end"), ("filled start", "filled_start"))
              if r.get(k) == "True"]
    long_gap = " LONG GAP {:.0f} s".format(float(r["gap_s"])) if r.get("long_gap") == "True" else ""
    return "{} to {}  {:9s} {:5.1f} s, exit {}, end {}{}{}".format(
        clock(r["start"], kicks), clock(r["end"], kicks), r["kind"], (r["end"] - r["start"]) / FPS, r["how"], r["end_how"],
        " (" + ", ".join(filled) + ")" if filled else "", long_gap)


def changes(before, after, kicks):
    """Stoppages that appear, vanish or change between the runs, matched by overlap."""
    print("\nchanges:")
    seen_after = set()
    for b in before:
        match = [i for i, a in enumerate(after) if a["start"] < b["end"] and a["end"] > b["start"]]
        seen_after.update(match)
        if not match:
            print("  gone     " + describe(b, kicks))
            continue
        a = after[match[0]]
        if (a["start"], a["end"], a["kind"]) != (b["start"], b["end"], b["kind"]):
            print("  changed  " + describe(b, kicks) + "\n        -> " + describe(a, kicks))
    for i, a in enumerate(after):
        if i not in seen_after:
            print("  new      " + describe(a, kicks))


def main():
    labels, tags = read_labels(), read_tags()
    pairs = tuple(zip(("before", "after"), sys.argv[1:3])) if len(sys.argv) == 3 else RUNS
    runs_ = {name: load(suffix) for name, suffix in pairs}
    columns = {name: numbers(rows, dead, labels, tags) for name, (rows, dead) in runs_.items()}
    print("{:34s} {:>10s} {:>10s}".format("", *columns))
    for key in columns["before"]:
        print("{:34s} {:>10s} {:>10s}".format(key, columns["before"][key], columns["after"][key]))
    changes(runs_["before"][0], runs_["after"][0], kickoffs(tags))


if __name__ == "__main__":
    main()
