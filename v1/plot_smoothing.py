"""Raw vs filled ball position and speed for a few stretches, filled and missing frames shaded.

    .venv\\Scripts\\python.exe v1/plot_smoothing.py

Picks three stretches from the scored frames automatically: a short gap that was filled, a long
gap left missing, and 10 s of plain play. Writes
cache/plots/smoothing.png (gitignored: it shows match data, so it stays on this computer).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from smooth import FPS, SEEN, FILLED, MISSING, runs  # noqa: E402

# Params
CACHE_DIR = os.path.join(ROOT, "cache")
OUT_PATH = os.path.join(CACHE_DIR, "plots", "smoothing.png")
PAD = 5 * FPS          # context before and after a gap
MAX_SPAN = 40 * FPS    # cap a stretch at 40 s
SHADE = {FILLED: ("#4cc78a", "filled (straight line across the gap)"),
         MISSING: ("#9aa0a6", "still missing (over 1 s)")}
RAW, SMOOTH, GIVEN = "#6b7280", "#1f5fbf", "#c2410c"


def pick_stretches(state, scored):
    """[(title, first, last)] - one example of each case."""
    def window(start, end):
        first = start - PAD
        return first, min(end + PAD, first + MAX_SPAN)

    out = []
    filled = [r for r in runs((state == FILLED) & scored)]
    if filled:
        out.append(("Short gap filled (up to 1 s)", *window(*max(filled, key=lambda r: r[1] - r[0]))))
    missing = [r for r in runs((state == MISSING) & scored) if 2 * FPS <= r[1] - r[0] <= 15 * FPS]
    if missing:
        out.append(("Long gap left empty", *window(*missing[0])))
    seen = (state == SEEN) & scored
    for start, end in runs(seen):
        if end - start >= 12 * FPS and start > np.flatnonzero(scored)[0] + 60 * FPS:
            out.append(("Plain play, ball seen", start + FPS, start + 11 * FPS))
            break
    return out


def shade(ax, state, first, last):
    for code, (color, _) in SHADE.items():
        for start, end in runs(state[first:last] == code):
            ax.axvspan(start / FPS, end / FPS, color=color, alpha=0.18, lw=0)


def main():
    raw = np.load(os.path.join(CACHE_DIR, "ball.npz"))
    sm = np.load(os.path.join(CACHE_DIR, "smooth.npz"))
    scored = np.load(os.path.join(CACHE_DIR, "scored.npy"))
    state = sm["ball_state"]
    raw_speed = np.r_[np.nan, np.hypot(np.diff(raw["x"]), np.diff(raw["y"])) * FPS]

    stretches = pick_stretches(state, scored)
    fig, axes = plt.subplots(3, len(stretches), figsize=(4.6 * len(stretches), 8.2),
                             sharex="col", squeeze=False)
    rows = (("x (m, along the pitch)", raw["x"], sm["ball_x"]),
            ("y (m, across the pitch)", raw["y"], sm["ball_y"]),
            ("ball speed (m/s)", raw_speed, sm["ball_speed"]))

    for col, (title, first, last) in enumerate(stretches):
        t = np.arange(last - first) / FPS
        for row, (label, raw_v, sm_v) in enumerate(rows):
            ax = axes[row, col]
            shade(ax, state, first, last)
            ax.plot(t, raw_v[first:last], ".", color=RAW, ms=2.5)
            ax.plot(t, sm_v[first:last], "-", color=SMOOTH, lw=1.6)
            if row == 2:
                ax.plot(t, raw["speed"][first:last], "x", color=GIVEN, ms=3, mew=0.8)
            ax.grid(color="#e5e7eb", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(label)
        axes[0, col].set_title("{}\nframes {:,}-{:,}".format(title, first, last - 1), fontsize=10)
        axes[2, col].set_xlabel("seconds into the stretch")

    handles = [Line2D([], [], ls="", marker=".", color=RAW, ms=8, label="raw (speed: frame to frame)"),
               Line2D([], [], color=SMOOTH, lw=1.6, label="smoothed"),
               Line2D([], [], ls="", marker="x", color=GIVEN, ms=6, label="speed given in the file")]
    handles += [Patch(color=c, alpha=0.35, label=l) for c, l in SHADE.values()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Step 1 - ball: raw vs smoothed", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130)
    print("wrote {} ({} stretches: {})".format(
        OUT_PATH, len(stretches), "; ".join(s[0] for s in stretches)))


if __name__ == "__main__":
    main()
