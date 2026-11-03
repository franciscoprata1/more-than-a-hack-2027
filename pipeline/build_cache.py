"""Phase 2: read the raw tracking + Smart Tagging once and save a small, fast copy (the cache).

    .venv\Scripts\python.exe pipeline/build_cache.py

Writes cache/ (gitignored, never published): ball.npz, players.npz, tags.csv, scored.npy.
Positions are in metres; missing values stay NaN. Row i is video frame i (ATD frame_id - 1).
Prints only totals, never raw rows.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
from baseline import team_of_each_track, team_names, FIELD_SEP, GROUP_SEP  # noqa: E402
import load_smart_tagging  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")
TEAM_CODE = {"home": "H", "away": "A"}


def parse_group(group, length, width):
    """'x,y,speed' -> (x_m, y_m, speed). NaN if missing."""
    x, y, speed = group.split(",")
    return float(x) * length, float(y) * width, float(speed)


def read_tracking(raw_path, n_frames, n_slots, pitch):
    length, width = pitch
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    seen = np.zeros(n_frames, dtype=bool)
    scale = np.array([length, width, 1.0], dtype=np.float64)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            seen[i] = True
            ball[i] = np.array(parts[2].split(","), dtype=np.float64) * scale
            groups = parts[1].split(GROUP_SEP)
            players[i, :len(groups)] = (
                np.array([g.split(",") for g in groups], dtype=np.float64) * scale)
    return ball, players, seen


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    meta, raw = require(["atd_meta", "atd_raw"], GAME)
    n_frames = GAMES[GAME]["n_video_frames"]
    pitch = GAMES[GAME]["pitch"]
    team_of_track = team_of_each_track(meta)

    ball, players, seen = read_tracking(raw, n_frames, len(team_of_track), pitch)

    scored = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        scored[first:last + 1] = True

    np.savez_compressed(os.path.join(CACHE_DIR, "ball.npz"),
                        x=ball[:, 0], y=ball[:, 1], speed=ball[:, 2],
                        visible=~np.isnan(ball[:, 0]))
    np.savez_compressed(os.path.join(CACHE_DIR, "players.npz"),
                        x=players[:, :, 0], y=players[:, :, 1], speed=players[:, :, 2],
                        team=team_of_track)  # 0 = home (H), 1 = away (A), per slot
    np.save(os.path.join(CACHE_DIR, "scored.npy"), scored)

    names = dict(zip(("H", "A"), team_names(meta)))
    events = load_smart_tagging.load()
    rows = []
    for e in events:
        team = e["tags"].get("Team")
        rows.append({
            "code": e["code"], "type": e["tags"].get("Type"), "half": e["tags"].get("Half"),
            "team": "H" if team == names["H"] else "A" if team == names["A"] else "",
            "start_frame": e["start_frame"], "end_frame": e["end_frame"],
            "team_raw": team,
        })
    tags = pd.DataFrame(rows)
    tags.drop(columns="team_raw").to_csv(os.path.join(CACHE_DIR, "tags.csv"), index=False)

    # Sanity checks: totals only.
    vis = ~np.isnan(ball[:, 0])
    n_tracked = (~np.isnan(players[:, :, 0])).sum(axis=1)
    print("frames read: {} of {}".format(int(seen.sum()), n_frames))
    print("scored frames: {}".format(int(scored.sum())))
    print("ball visible: {:.1f}% of all, {:.1f}% of scored".format(
        100 * vis.mean(), 100 * vis[scored].mean()))
    print("players tracked per scored frame: mean {:.1f}".format(n_tracked[scored].mean()))
    print("track slots: {} ({} home, {} away)".format(
        len(team_of_track), int((team_of_track == 0).sum()), int((team_of_track == 1).sum())))
    print("smart tags: {}  teams: {}".format(len(tags), tags["team_raw"].fillna("none").value_counts().to_dict()))
    for f in sorted(os.listdir(CACHE_DIR)):
        print("  {:12s} {:.1f} MB".format(f, os.path.getsize(os.path.join(CACHE_DIR, f)) / 1e6))


if __name__ == "__main__":
    main()
