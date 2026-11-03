"""Write human-readable CSV copies of the cache for a frame range, to open in Excel.

    .venv\Scripts\python.exe pipeline\export_preview.py [--start 11185] [--frames 250]

Writes cache/preview/ball.csv, players.csv (one row per frame, columns x/y/speed per slot, slot
team in the header), scored.csv. The folder is inside cache/, so it is gitignored and local only.
"""

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
CACHE_DIR = os.path.join(ROOT, "cache")
OUT_DIR = os.path.join(CACHE_DIR, "preview")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=11185, help="first video frame")
    parser.add_argument("--frames", type=int, default=250, help="how many frames (25 = 1 s)")
    args = parser.parse_args()
    rows = slice(args.start, args.start + args.frames)
    frames = np.arange(args.start, args.start + args.frames)
    os.makedirs(OUT_DIR, exist_ok=True)

    ball = np.load(os.path.join(CACHE_DIR, "ball.npz"))
    pd.DataFrame({"frame": frames, "x_m": ball["x"][rows], "y_m": ball["y"][rows],
                  "speed": ball["speed"][rows], "visible": ball["visible"][rows]}
                 ).round(2).to_csv(os.path.join(OUT_DIR, "ball.csv"), index=False)

    pl = np.load(os.path.join(CACHE_DIR, "players.npz"))
    cols = {"frame": frames}
    for slot, team in enumerate(pl["team"]):
        tag = "{}{:02d}".format("H" if team == 0 else "A", slot + 1)
        for key in ("x", "y", "speed"):
            cols["{}_{}".format(tag, key)] = pl[key][rows, slot]
    pd.DataFrame(cols).round(2).to_csv(os.path.join(OUT_DIR, "players.csv"), index=False)

    scored = np.load(os.path.join(CACHE_DIR, "scored.npy"))
    pd.DataFrame({"frame": frames, "scored": scored[rows]}
                 ).to_csv(os.path.join(OUT_DIR, "scored.csv"), index=False)
    print("wrote 3 CSVs ({} frames from {}) to {}".format(args.frames, args.start, OUT_DIR))


if __name__ == "__main__":
    main()
