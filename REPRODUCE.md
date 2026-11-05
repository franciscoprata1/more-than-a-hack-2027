# Reproducing our submission

From the provided data bundle to `evaluation/results/v2_stage5_n5.csv`, the file we submitted.
The approach itself is described in the technical document; this page is only how to run it.

## What you need

- **Windows 10/11** (what we ran on; the code is plain Python and should run elsewhere, but only
  this setup is checked). An 8-core CPU; no GPU is used.
- **Python 3.11** (we used 3.11.9).
- **The Microsoft Visual C++ Redistributable** (PyTorch on Windows needs it).
- **The data bundle**, unchanged, in `data/tactical/` (`BAR-ATH_tactical_atd.txt`, `.xml`,
  `_smart_tagging.xml`, `_homography.csv`; see [DATA.md](DATA.md)). The video is not used.
- **Internet access** once, to download the 7 open Sportec / IDSSE matches (CC BY 4.0) that every
  model is trained and tuned on. Nothing is trained or tuned on the provided match.
- About **8 hours** of computing in all, most of it one overnight step (step 5).

## 1. Install

Run everything from the repository root.

```
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install numpy==2.4.6 pandas==2.2.3 kloppy==3.18.0 lxml==6.1.3 matplotlib==3.11.2
.venv\Scripts\python.exe -m pip install -r requirements-v1.txt
```

`requirements-v1.txt` installs the CPU build of PyTorch **2.14.0**. Keep these exact versions:
newer pandas and PyTorch builds were blocked by Windows Smart App Control on our machine, and other
versions may change the last digits of the results.

## 2. Our match: the cache (about 2 minutes)

```
.venv\Scripts\python.exe pipeline\build_cache.py
.venv\Scripts\python.exe v1\smooth.py
.venv\Scripts\python.exe v2\our_profile.py
```

`build_cache.py` turns the raw tracking into fast numpy files in `cache/`; `smooth.py` fills short
ball and player gaps and smooths positions; `our_profile.py` measures our match's totals only (how
long the ball is missing, and why), so the training matches can be made to look like it.

## 3. The training matches (Sportec / IDSSE)

```
.venv\Scripts\python.exe v2\sportec_download.py
.venv\Scripts\python.exe v2\sportec_cache.py
.venv\Scripts\python.exe v2\sportec_degrade.py
.venv\Scripts\python.exe v2\bench.py
```

Downloads the 7 matches to `data/sportec/`, converts them to our format, then makes a degraded copy
of each with ball and player gaps like ours (seeded, so the same copy every time). `bench.py`
checks the labels and scores the simple baselines.

## 4. Stages 1-3 (rules tuned on the training matches)

```
.venv\Scripts\python.exe v2\stage1.py
.venv\Scripts\python.exe v2\stage2.py
.venv\Scripts\python.exe v2\stage2_majority.py
.venv\Scripts\python.exe v2\stage3.py
.venv\Scripts\python.exe v2\stage3_owner.py
.venv\Scripts\python.exe v2\stage3_control.py
```

Each stage tunes its settings leave one match out on the 7 training matches and saves a results
grid in `cache/v2_stage*_grid.csv`; the next stage reads it. Run them in this order.
`stage2_majority.py`, `stage3.py` and `stage3_owner.py` are versions we tried and did not keep:
the kept chain is Stage 1 -> 2 -> 3 control, but each script prints the earlier versions beside it
for comparison, so their grids must exist.

## 5. The ball-gap model (about 6-7 hours, overnight)

```
.venv\Scripts\python.exe v2\gap_truth.py
.venv\Scripts\python.exe v2\gap_dataset.py
.venv\Scripts\python.exe v2\gap_train.py --all
.venv\Scripts\python.exe v2\gap_report.py
.venv\Scripts\python.exe v2\gap_blend.py
```

Trains 7 models (each leaves one training match out) that guess where the ball is while it is
missing. If `gap_train.py --all` is interrupted, run it again: it resumes from its last checkpoint.
Seeds are fixed and PyTorch runs deterministic on 8 threads (`THREADS` in `v1/ball_train.py`;
Stage 5 uses 8 too).

Each model stops after 60 epochs, or after 8 epochs without improving, whichever comes first (on
our machine about 1 minute per epoch; no model hit the time limit). There is also a safety limit of
100 minutes per model (`FOLD_CAP_MIN` in `v2/gap_train.py`). On a computer slower than about
1.6 minutes per epoch, raise it so that it never triggers; otherwise a model stops early and the
file changes.

## 6. Stages 4-5 and the submission (about 30 minutes)

```
.venv\Scripts\python.exe v2\stage4.py
.venv\Scripts\python.exe v2\stage5.py
.venv\Scripts\python.exe v2\stage5.py --nets 5
.venv\Scripts\python.exe evaluation\validate_submission.py evaluation\results\v2_stage5_n5.csv
```

Stage 4 adds the ball-gap model's clues; Stage 5 trains a small network on all the clues. The
one-network run comes first because the 5-network run is compared against it. The last Stage 5
command writes **`evaluation/results/v2_stage5_n5.csv`**, the submitted file.

## Checking it is the same file

```
certutil -hashfile evaluation\results\v2_stage5_n5.csv SHA256
```

Our submitted file starts `eed98bc65e124013`.

What we checked on our machine (5 November 2026):

| Rerun from | Result |
|---|---|
| Stage 5, from its saved inputs | byte-identical |
| Stage 5, its inputs (all 8 matches) and settings grid rebuilt from scratch | byte-identical |
| Stage 5 with its thread count pinned to 8 (`THREADS`) | byte-identical |

The ball-gap models (step 5) were trained with PyTorch 2.15.0. Windows later blocked that version
on our machine, so the repository pins 2.14.0, under which Stage 5 reproduces the file exactly. We
have not retrained the ball-gap models under 2.14.0.

## Notes

- `cache/`, `data/` and the results CSVs are not in the repository (the match data may not be
  redistributed); every step above rebuilds them.
- The public 5-minute answer key (`evaluation/gt/`) is only scored as a report at the end of
  Stage 5. Nothing reads it to choose a setting.
- `CLAUDE.md` is our working notes: what every script does and what was tried and dropped.
