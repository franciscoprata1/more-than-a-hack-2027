"""The profile table: our match against the degraded Sportec bench, every number from our_profile.measure.

    .venv\\Scripts\\python.exe v2/bench_profile.py [--save-old]

Bench numbers are the mean over the 7 degraded matches (restarts = Sportec set piece events, all of
them). --save-old stores the current bench as the "old bench" column (cache/v2_profile_bench_old.csv)
before v2/sportec_degrade.py is rewritten or rerun. Clean Sportec is shown for reference.
Writes cache/v2_profile_table.csv. Our match: totals only.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402
import our_profile as op  # noqa: E402
from bench import MATCHES, CACHE  # noqa: E402

# Params
OLD = os.path.join(CACHE, "v2_profile_bench_old.csv")
TABLE = os.path.join(CACHE, "v2_profile_table.csv")


def sportec_restarts(m):
    ev = m["events"]
    ev = ev[(ev["set_piece"] == 1) & (ev["in_half"] == 1)]
    return np.sort(ev["frame"].values.astype(int))


def bench_mean(suffix):
    rows = []
    for mt in MATCHES:
        m = bench.load(mt + suffix)
        rows.append(op.measure(m, sportec_restarts(m))[0])
    return pd.DataFrame(rows).mean()


def main():
    pd.set_option("display.width", 200)
    ours = pd.Series(op.measure(bench.load("ours"), op.our_restarts(bench.load("ours")))[0])
    if "--save-old" in sys.argv:
        bench_mean("~").to_csv(OLD, header=["old bench"])
    t = pd.DataFrame({"ours": ours})
    if os.path.isfile(OLD):
        t["old bench"] = pd.read_csv(OLD, index_col=0)["old bench"]
    t["new bench"] = bench_mean("~")
    t["clean Sportec"] = bench_mean("")
    t.to_csv(TABLE)
    print(t.to_string(float_format=lambda v: "{:.3f}".format(v)))


if __name__ == "__main__":
    main()
