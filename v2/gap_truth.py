"""Part 2, step 1: the true ball behind the degraded bench's gaps.

    .venv\\Scripts\\python.exe v2/gap_truth.py

v2/sportec_degrade.py replaces Sportec's frozen dead ball by a walk to the restart spot, then hides
the ball in gaps. It does not keep the ball it hid. The walk has a fixed seed per match, so it is
rebuilt here with the saved fit (cache/v2_degrade_fit.json), and checked: on every frame where the
degraded ball is seen, the rebuilt ball must be the same to the last float32 bit.

Writes cache/sportec_degraded/<match>/ball_truth.npz (x, y). Prints totals only.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sportec_degrade as sd  # noqa: E402
from sportec_download import MATCHES  # noqa: E402

# Params
FPS = 25


def truth(k, mt, carry_speed):
    """The walked ball of match k, before any of it was hidden."""
    m = sd.Match(k, mt)
    m.set_carry(carry_speed)
    return m.bx.astype(np.float32), m.by.astype(np.float32), m.half


def main():
    fit = json.load(open(sd.FIT))
    print("carry speed {} m/s (from {})".format(fit["carry_speed"], os.path.basename(sd.FIT)))
    print("match   hidden min  rebuilt = seen ball  max diff on seen")
    for k, mt in enumerate(MATCHES):
        x, y, half = truth(k, mt, fit["carry_speed"])
        deg = np.load(os.path.join(sd.DST, mt, "ball.npz"))
        seen = ~np.isnan(deg["x"])
        diff = np.nanmax(np.abs(np.r_[x[seen] - deg["x"][seen], y[seen] - deg["y"][seen]]))
        same = diff == 0
        print("{}    {:6.1f}       {}                {:.1e}".format(
            mt, (~seen & (half > 0)).sum() / FPS / 60, "yes" if same else "NO ", diff))
        assert same, "rebuilt ball differs from the degraded copy: rerun v2/sportec_degrade.py first"
        np.savez_compressed(os.path.join(sd.DST, mt, "ball_truth.npz"), x=x, y=y)
    print("wrote ball_truth.npz for {} matches".format(len(MATCHES)))


if __name__ == "__main__":
    main()
