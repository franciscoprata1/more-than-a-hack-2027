"""Stage 0: download the 7 IDSSE (Sportec / DFL open data, CC BY 4.0) matches once.

    .venv\Scripts\python.exe v2/sportec_download.py

figshare's ndownloader link (kloppy's default) now answers with an empty bot check, so the files
come from the official figshare API link instead. Saved to data/sportec/<match>_<kind>.xml
(gitignored, like all of data/). Cite Bassek et al., Scientific Data, 2025.
"""

import os
import sys
import time
import urllib.request

from kloppy._providers.sportec import get_IDSSE_url

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
OUT_DIR = os.path.join(ROOT, "data", "sportec")
API = "https://api.figshare.com/v2/file/download/"
MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
KINDS = ("meta", "event", "tracking")


def path_of(match, kind):
    return os.path.join(OUT_DIR, "{}_{}.xml".format(match, kind))


def download(match, kind):
    path = path_of(match, kind)
    if os.path.isfile(path) and os.path.getsize(path) > 1000:
        return path
    url = API + get_IDSSE_url(match, kind).rsplit("/", 1)[-1]
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "kloppy"})) as r, \
                    open(path + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(path + ".part", path)
            return path
        except Exception as e:  # network hiccup: wait and retry
            print("  retry {} {} ({})".format(match, kind, e))
            time.sleep(10)
    raise IOError("could not download {} {}".format(match, kind))


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for m in (sys.argv[1:] or MATCHES):
        for k in KINDS:
            p = download(m, k)
            print("{} {:8s} {:.0f} MB".format(m, k, os.path.getsize(p) / 1e6), flush=True)
