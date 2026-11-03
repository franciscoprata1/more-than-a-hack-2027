"""Timeline of our dead balls for the whole match, with the set-piece tag starts, plus a zoom on
the first five minutes against the answer key.

    .venv\\Scripts\\python.exe v1/plot_dead_balls.py

Writes cache/plots/dead_balls.png (gitignored: it shows match data, so it stays on this computer).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from smooth import FPS, runs  # noqa: E402
from dead_balls import TAG_LEAD, TAG_MATCH, read_labels, read_tags  # noqa: E402

# Params
CACHE_DIR = os.path.join(ROOT, "cache")
OUT_PATH = os.path.join(CACHE_DIR, "plots", "dead_balls.png")
KICKOFFS = {"1st half": (11185, 0), "2nd half": (113742, 45)}  # kickoff frame, clock minute
OURS, TRUTH = "#4b5563", "#7c3aed"
TAG_OK, TAG_MISS = "#16a34a", "#dc2626"


def minute(frame, half):
    kickoff, offset = KICKOFFS[half]
    return (np.asarray(frame) - kickoff) / FPS / 60 + offset


def bars(ax, mask, row, color, half, first, last):
    spans = [(minute(s, half), (e - s) / FPS / 60) for s, e in runs(mask[first:last + 1]) for s, e in [(s + first, e + first)]]
    ax.broken_barh(spans, (row - 0.35, 0.7), color=color, lw=0)


def tag_ticks(ax, tags, dead, row, half, first, last):
    ends = np.array([e for _, e in runs(dead)])
    for s in tags:
        if first <= s <= last:
            agrees = ((ends >= s) & (ends <= s + TAG_MATCH)).any()
            ax.plot([minute(s + TAG_LEAD, half)] * 2, [row - 0.35, row + 0.35],
                    color=TAG_OK if agrees else TAG_MISS, lw=1.4)


def style(ax, labels):
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e5e7eb", lw=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main():
    dead = np.load(os.path.join(CACHE_DIR, "dead.npy"))
    scored = np.load(os.path.join(CACHE_DIR, "scored.npy"))
    tags = [t["start_frame"] for t in read_tags() if t["code"] == "SET PIECES"]
    labels = read_labels()
    key = np.flatnonzero(labels != "X")

    blocks = [(s, e - 1) for s, e in runs(scored)]
    fig, axes = plt.subplots(3, 1, figsize=(15, 7.5), gridspec_kw={"height_ratios": [1, 1, 1.4]})
    for ax, (half, (first, last)) in zip(axes[:2], zip(KICKOFFS, blocks)):
        bars(ax, dead, 0, OURS, half, first, last)
        tag_ticks(ax, tags, dead, 1, half, first, last)
        style(ax, ["our D", "set-piece tags\n(restart = start + 3.5 s)"])
        ax.set_xlim(minute(first, half), minute(last, half))
        ax.set_title("{}: {} stoppages, {:.0%} of scored time dead".format(
            half, len(runs(dead[first:last + 1])), dead[first:last + 1].mean()), loc="left", fontsize=10)
    axes[1].set_xlabel("match minute")

    ax = axes[2]
    first, last = key[0], key[-1]
    bars(ax, labels == "D", 0, TRUTH, "1st half", first, last)
    bars(ax, dead, 1, OURS, "1st half", first, last)
    tag_ticks(ax, tags, dead, 2, "1st half", first, last)
    style(ax, ["answer key D", "our D", "set-piece tags"])
    ax.set_xlim(minute(first, "1st half"), minute(last, "1st half"))
    ax.set_xlabel("match minute")
    truth, ours = (labels == "D")[first:last + 1], dead[first:last + 1]
    ax.set_title("First five minutes vs the answer key: we catch {:.0%} of true D, {:.0%} of our D is right".format(
        (truth & ours).sum() / truth.sum(), (truth & ours).sum() / max(1, ours.sum())), loc="left", fontsize=10)

    handles = [Patch(color=OURS, label="our dead ball"), Patch(color=TRUTH, label="answer key dead ball"),
               Line2D([], [], color=TAG_OK, lw=2, label="tag: one of our stoppages ends within 10 s"),
               Line2D([], [], color=TAG_MISS, lw=2, label="tag: no stoppage of ours ends there")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Step 2 - dead balls from the tracking", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130)
    print("wrote " + OUT_PATH)


if __name__ == "__main__":
    main()
