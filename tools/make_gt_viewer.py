"""Build a local HTML player: the tactical video with the public ground truth drawn on top.

    python tools/make_gt_viewer.py
    # then open tools/gt_viewer.html in Chrome or Edge

What you get, for the 5 minutes that have a public ground truth (frames 11185 .. 18684):
  - the tactical video, seeked to the right frame
  - the ground-truth label for the current frame, and a timeline of every possession
  - a top-down minimap of the tracking (players coloured by team, the ball)
  - an optional second strip with a baseline submission, to see where the nearest-player rule
    disagrees with the ground truth
  - our dead balls from v1/dead_balls.py against the ground truth, with a jump list of our stoppages

The page points at the video by relative path, so it must stay in tools/ next to ../data. The
generated HTML holds tracking positions, which come from the gated data bundle, so it is gitignored
and must not be committed or shared.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require  # noqa: E402
from baseline import team_of_each_track, team_names  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
COMPARE_PATH = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")
DEAD_PATH = os.path.join(ROOT, "cache", "dead.npy")
EVENTS_PATH = os.path.join(ROOT, "cache", "dead_events.csv")
OUT_PATH = os.path.join(HERE, "gt_viewer.html")
VIDEO_REL = "../data/tactical/BAR-ATH_tactical.mp4"


def read_labels(path):
    """The label column of a frame,label file."""
    with open(path, newline="", encoding="utf-8") as handle:
        return np.array([row["label"] for row in csv.DictReader(handle)])


def our_dead_balls(first, last):
    """Our D frames as a string (D or -) and our stoppages touching the window, or (None, [])."""
    if not (os.path.isfile(DEAD_PATH) and os.path.isfile(EVENTS_PATH)):
        return None, []
    dead = np.load(DEAD_PATH)[first:last + 1]
    with open(EVENTS_PATH, newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["status"] == "stoppage"]
    stops = [{"s": int(r["start"]), "e": int(r["end"]), "kind": r["kind"], "team": r["team"], "how": r["end_how"]}
             for r in rows if int(r["start"]) <= last and int(r["end"]) >= first]
    return "".join("D" if d else "-" for d in dead), stops


def covered_window(labels):
    """(first, last) video frame where the ground truth is not X."""
    graded = np.flatnonzero(labels != "X")
    return int(graded[0]), int(graded[-1])


def read_tracking(raw_path, team_of_track, first, last):
    """One compact string per video frame: 'ballx,bally|x,y,team;x,y,team;...' (empty = missing).

    The ATD counts frames from 1 and a submission from 0, hence the minus one.
    """
    out = [None] * (last - first + 1)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if frame < first:
                continue
            if frame > last:
                break
            ball = parts[2].strip()
            ball_text = "" if ball[0] == "N" else ",".join(ball.split(",")[:2])
            players = []
            for slot, group in enumerate(parts[1].split(";")):
                if group[0] == "N":
                    continue
                x, y = group.split(",")[:2]
                players.append("{:.3f},{:.3f},{}".format(float(x), float(y), team_of_track[slot]))
            out[frame - first] = ball_text + "|" + ";".join(players)
    return ["|" if item is None else item for item in out]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compare", default=COMPARE_PATH,
                        help="a second submission to show as a strip (default: baseline_smooth50)")
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    gt = read_labels(GT_PATH)
    first, last = covered_window(gt)
    compare = read_labels(args.compare) if os.path.isfile(args.compare) else None
    ours, stops = our_dead_balls(first, last)

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    teams = team_names(meta_path)
    print("reading tracking for frames {}..{} ...".format(first, last))
    tracking = read_tracking(raw_path, team_of_each_track(meta_path), first, last)

    payload = {
        "video": VIDEO_REL,
        "fps": GAMES[GAME]["fps"],
        "pitch": GAMES[GAME]["pitch"],
        "first": first,
        "last": last,
        "teams": list(teams),
        "gt": "".join(gt[first:last + 1]),
        "compare": "".join(compare[first:last + 1]) if compare is not None else None,
        "compare_name": os.path.splitext(os.path.basename(args.compare))[0],
        "ours": ours,
        "stops": stops,
        "tracking": tracking,
    }
    with open(os.path.join(HERE, "gt_viewer_template.html"), encoding="utf-8") as handle:
        template = handle.read()
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(template.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":"))))
    print("wrote {}  ({:.1f} MB)".format(os.path.relpath(args.out, ROOT),
                                       os.path.getsize(args.out) / 1e6))
    print("open it in Chrome or Edge; keep it inside tools/ so the video path resolves")


if __name__ == "__main__":
    main()
