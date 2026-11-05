# CLAUDE.md — More than a Hack 2027, possession challenge

## Who and how
- Francisco is the CEO and a beginner coder; Claude writes and runs all code and explains each step in plain words (define jargon).
- Goal: learn by building a possession model for one match (Barcelona vs Athletic Club). Winning is a bonus.
- Plan (phases 0-5, glossary): Claude Doc https://claude.ai/artifact/MwFN2gDYTX6WkK8SQnkcBV. Start each session from the next unfinished phase; tick or annotate it when done. Progress notes are also in auto-memory.
- Decisions: only the `data/tactical/` folder, only this match, optimise spellF1 first, make a fresh public repo once real code starts (it must never contain data files).

## Hard rules
- **Match data stays on this computer.** Claude is a cloud model: scripts print only scores and summary numbers (counts, means), never raw tracking rows, video frames or ground truth.
- Never commit `data/`, `cache/`, `tools/gt_viewer.html`, `evaluation/results/*.csv` (except `baseline_*`) or any `*_possession_gt.csv`. Check `.gitignore` before adding new output folders.
- Commit or push only when asked.
- Change one thing at a time so we know what helped. Set thresholds from football sense first, then look at the score.
- The 5-minute public answer key (frames 11185-18684, 17 possessions) is a code check, too small to rank two versions. Do not tune against it.
- No frame numbers of this game typed into the pipeline: find them by code. Open items: `SECOND_HALF_WAIT` in `v1/dead_balls.py`, `KICKOFFS` in `v1/plot_dead_balls.py`.

## Environment (Windows)
- Always run from `C:\Projects\more-than-a-hack-2027` (moved out of OneDrive on purpose).
- Use the project venv: `.venv\Scripts\python.exe` (Python 3.11). Plain `python` is not on PATH.
- In Git Bash: `.venv/Scripts/python.exe`.

## Where everything is
| Path | What it is |
|---|---|
| `data/tactical/` | Raw gated data: `BAR-ATH_tactical_atd.txt/.xml` (tracking), `..._smart_tagging.xml`, `..._homography.csv`, `BAR-ATH_tactical.mp4`. Do not edit. |
| `data/broadcast/` | Unused (decision D1). |
| `games.py` | Match constants: 199,319 frames, 25 fps, pitch 105.3 x 68.0 m, half frame blocks. Helpers `require`, `half_video_blocks`. |
| `starter/` | Organisers' loaders, baseline, notebooks 01-03. Read-only reference; don't modify. |
| `evaluation/` | Scorer (`run_evaluation.py`, `possession_metrics.py`), `validate_submission.py`, public GT in `gt/`. |
| `evaluation/results/` | Submission CSVs (`frame,label`, 199,319 rows). Only `baseline_*.csv` are tracked. Write our own with `--out` elsewhere or name them `v1_*` (gitignored). |
| `evaluation/reports/`, `scoreboard.csv` | Generated scorer output (gitignored). Reports include `<name>_transitions.csv` (matched/missed/invented possession changes). |
| `pipeline/build_cache.py` | Phase 2: builds `cache/` from raw data (~95 s). |
| `cache/` | Fast cleaned copy (gitignored). See format below. |
| `v1/smooth.py` | Phase 3 step 1: fill ball gaps up to 1 s and player gaps up to 2 s (straight line), longer gaps stay empty, smooth, recompute velocity -> `cache/smooth.npz` (ball_* incl. `ball_state` 0 seen/1 filled/2 missing, `ball_gap` gap length in frames, 0 = seen; player_* incl. `player_gap`, acceleration `player_ax`, `player_ay`, `player_accel`; above 12 m/s a track slot jumped). Use these speeds, not the file's. Never detect touches where gap > 0. |
| `v1/plot_smoothing.py` | Raw vs smoothed plot -> `cache/plots/smoothing.png`. Plots show match data: Claude never opens them, Francisco does. |
| `v1/dead_balls.py` | Phase 3 step 2: stoppages with a Start (ball out: seen past a line, gone missing near one, or trajectory = last-second heading extended to a line) and an End (restart at the right spot, moving the right way) for throw-in, corner, goal kick, goal. Per-type minimum length; player shape required for goal kicks and trajectory exits -> `cache/dead.npy`, `cache/dead_events.csv` (one row per exit looked at, `how` = seen/missing/trajectory); tag check per type + answer-key check. Fouls are step 3. |
| `v1/dead_balls.py --filled` | Same detector, plus the blended ball (`ball_predicted_blend.npz`) for location only: exit line and restart spot; still/speed/direction stay on real frames. Writes `cache/dead_filled.npy`, `cache/dead_events_filled.csv` (adds `filled_line`, `filled_end`, `gap_s`, `long_gap` = over 30 s). The plain outputs are not touched. `--cross-start` also sets Start to the predicted crossing frame -> `..._filled_cross`; `--cross-seen` does that for balls that came back already out too -> `..._filled_cross_seen` (worse: the blend puts the crossing at the gap's end). `v1/dead_compare.py [suffixA suffixB]` prints before vs after, with match clock (45+x in added time) and video time. |
| `v1/plot_dead_balls.py` | Whole-match dead-ball timeline + 5-minute zoom -> `cache/plots/dead_balls.png`. |
| `v1/hidden_gaps.py` | Phase 3 step 1 (ball first plan): picks open play test gaps of 2, 5, 10, 20 s by code, seed 2027 -> `cache/hidden_gaps.npz` (`gap_<s>` start frames, `fold_<s>` 5-minute time block; step 3 must not train on a gap's block). Needs `cache/dead_events.csv`. |
| `v1/ball_baselines.py` | Phase 3 step 2: baselines A straight line, B follow carrier, C pressure point on the hidden gaps -> `cache/ball_baselines.csv`. Numbers the step 3 model must beat. Never tune on these gaps. |
| `v1/ball_dataset.py` | Phase 3 step 3 stage 1: 10 Hz inputs, ball channel, code-made "who has the ball" labels, fold split with a 20 s buffer, 30 s windows -> `cache/ball_dataset.npz`. Proves by assert that no model trains on its own test gaps. |
| `v1/ball_model.py` | Stage 2: the ball model and loss, our own build from the paper. Never copy from the Ballradar repo (no license). |
| `v1/ball_train.py` | Stage 2-3: `--check` (order and reproducibility), `--smoke` (fold 0, 10 epochs), `--fold N`, `--all` -> `cache/ball_model_fold<k>.pt`, `cache/ball_results_fold<k>.npz`. `--plain` drops the line, `--no-test` leaves the hidden gaps closed, `--tag` names the model file. `--all` trains the plain model and adds `_all` to the names; its results also hold the ball paths on the hidden gaps and validation blocks (for a later blend), and `cache/ball_checkpoint_fold<k>_all.pt` is saved every epoch, so running `--all` again resumes exactly. CPU, 8 threads, fixed seeds. The hidden gaps are never used to choose anything. |
| `v1/ball_report.py` | Stage 3 report from the `_all` results: model against A, B, C, error against distance from the nearest line, who has the ball -> `cache/ball_stage3.csv`. |
| `v1/ball_fill.py` | Stage 4: plain model in all 134 real gaps, each by the fold that never trained on it -> `cache/ball_predicted.npz` (`x`, `y`, `predicted` per frame; `gap_start`, `gap_end`, `gap_fold`). Believability only, no true ball there. Gaps over 30 s are low confidence: measured error stops at 20 s. |
| `v1/ball_blend.py` | Plain blended with the straight line, no training: T chosen on validation blocks (3 s), on the pitch / out scores -> `cache/ball_predicted_blend.npz` (same keys plus `t_seconds`). Use this file in later steps, not `ball_predicted.npz`. A predicted ball on the pitch is not proof it is in play: it catches about 3 in 10 out balls deep in a gap. |
| `v1/ball_checks.py` | Looks at a trained fold 0 model without opening the hidden gaps: `lengths plain line` (validation error per gap length), `stoppages line` (stopped play against open play). `cache/ball_model_fold0.pt` is the line model, `_plain` the one without the line, both 10 epochs; `plain_long`, `line_long` are the same two trained to early stopping. |
| `tools/make_gt_viewer.py` | Builds `tools/gt_viewer.html` (video + GT + minimap + our dead balls strip and stoppage list). Rebuild it after `v1/dead_balls.py`. |
| `v2/sportec_download.py`, `v2/sportec_cache.py` | v2 Stage 0: 7 IDSSE matches (CC BY 4.0) into `data/sportec/`, then `cache/sportec/<match>/` in our format (+ `labels.npy` 0 H/1 A/2 D/3 X, `half.npy`, `pitch.npy`, `events.csv`). Referees left out. |
| `v2/our_profile.py`, `v2/sportec_degrade.py`, `v2/bench_profile.py` | Our match's totals (gap causes, restarts, still ball, player dropouts) -> `cache/v2_profile.npz`; `measure()` gives the same numbers for any match. Degraded copy (bench v2) -> `cache/sportec_degraded/<match>/`: gaps placed by cause, weights fitted to our totals (`cache/v2_degrade_fit.json`). `bench_profile.py` prints ours vs old vs new bench -> `cache/v2_profile_table.csv`. Bench v1 results kept in `cache/bench1/`. |
| `v2/bench.py` | `load("J03WPY")` clean, `load("J03WPY~")` degraded, `load("ours")`; `score`, `lomo`, `report_ours` (writes `evaluation/results/<name>.csv`, 5 min score as report only). Run alone: label check + majority smoothing baselines. |
| `v2/stage1.py`, `v2/stage2.py`, `v2/deadball.py`, `v2/stage2_majority.py`, `v2/touches.py`, `v2/stage3.py`, `v2/owner.py`, `v2/stage3_owner.py`, `v2/stage3_control.py` | v2 Stage 1 (nearest team + Viterbi) and Stage 2 (dead ball layer, cues in `deadball.py`); `stage2_majority.py` puts the same stoppages on majority smoothing instead of Stage 1. Stage 3 first try (`stage3.py`, `touches.py`): touches, dropped (our ball is too smooth, 0.2 touches/min). Stage 3 owner (`stage3_owner.py`, cues in `owner.py`): nearest player within 1.5 m owns the ball. Stage 3 kept: `stage3_control.py`: control, duels, ball in flight (nearest-team clue off, lean to last team in control). Grids cached in `cache/v2_stage*_grid.csv`; delete to rerun. Gate (bench v2 on): D F1 up and spellF1 not down (`bench.gate`). Results in the plan doc, v2 tab. |
| `v2/gap_truth.py`, `v2/gap_dataset.py`, `v2/gap_train.py`, `v2/gap_report.py` | Part 2: v1 ball model (plain) trained on the degraded bench. `gap_truth` rebuilds the true ball behind the bench's gaps (seeded walk, checked bit-exact on seen frames) -> `sportec_degraded/<m>/ball_truth.npz`. `gap_dataset` -> `cache/v2_gap/<match>.npz` (incl. ours). `gap_train --smoke / --fold k / --all`: 7 folds, fold k leaves match k out; loss on real gaps + 0-2 extra hidden stretches; ~60 s/epoch, max 60 epochs or 100 min; resumes from `checkpoint_fold<k>.pt`. Writes `pred_<match>_fold<k>.npz` (x, y, predicted, p_home, p_away, p_nobody) for the left-out match and ours. `gap_report` scores against the straight line by length and cause and averages the 7 fills of ours -> `pred_ours_mean.npz` (+ `spread`). Log: `cache/v2_gap/train.log`. |
| `v2/gap_transfer.py` | Transfer test of p_nobody: ball hidden in v1's open-play gaps of ours (7 models averaged) vs bench live / dead stretches (left-out model). Ours open play 0.06 (0% above 0.3), bench live 0.08 (3%), bench dead 0.39 (65%): it transfers. |
| `v2/gap_blend.py`, `v2/stage4.py` | `gap_blend`: model blended with the straight line near gap edges, T chosen LOMO (2-3 s; ours 2 s) -> `cache/v2_gap/blend_<match>.npz`. `stage4`: on frames with the ball missing > 1 s, leans to D from p_nobody (log-odds around P_MID, can also lean away from D), predicted out / near a line / still; on top of Stage 3 control (its two LOMO evidence settings, costs free). Grid `cache/v2_stage4_grid.csv`, log `cache/v2_stage4.log`. Kept (bench LOMO spellF1 0.366, D F1 0.891); the gain is all the p_nobody cue. On ours it changes only 2.7% of frames. |
| `v2/stage5.py`, `v2/stage5_check.py` | Stage 5 combiner: 37 clues of Stages 1-4 kept apart per 5 Hz step + context (2 s / 10 s means, control 5 s before / after) -> small PyTorch network (2 x 64), trained with H/A swapped copies; its log-odds x alpha into Stage 3's Viterbi. Nested LOMO (match k never seen by anything chosen for it). Features cached in `cache/v2_stage5/`, grid `cache/v2_stage5_grid.csv`, log `cache/v2_stage5.log`. Kept: bench spellF1 0.431 (Stage 4 0.366), D F1 0.952. `--nets 5` averages 5 networks (seeds 2027-2031): KEPT, spellF1 0.438, D F1 0.953, better on 5 of 7 -> `evaluation/results/v2_stage5_n5.csv` = our final submission; grid `cache/v2_stage5_n5_grid.csv`, log `cache/v2_stage5_n5.log`. Switch cost picked its lowest value (2) after one extension. Pitch control clues (who reaches the ball first) tried 5 Nov: spellF1 +0.0004, no real gain, code removed. `--truth controlled`: scored on `bench.controlled` labels (a team holding the ball under 2 s does not take possession, the organisers' written rule; bench 437 -> 288 possessions per match), report only: spellF1 0.536, predicts 208 possessions. Tried 5 Nov, not kept: training on those labels (tie), switch cost below 2 (same paths: alpha and costs scale together), a change-detector network (possessions unchanged), snapping changes to the new team's nearest touch (worse at every width). `stage5_check`: shuffle importance per clue + top clues bench vs ours (ours' gaps: slower players, higher p_nobody, i.e. more dead ball). |
| `DATA.md`, `EVALUATION.md`, `RULES.md`, `SUBMISSION.md`, `FAQ.md`, `CHALLENGE.md` | Organisers' docs. Check `DATA.md` for expected numbers. |

## Cache format (`cache/`)
Row `i` = video frame `i` (= ATD frame_id - 1). Positions in metres, missing = NaN.
- `ball.npz`: `x`, `y`, `speed`, `visible` (bool), length 199,319.
- `players.npz`: `x`, `y`, `speed` shaped (199319, 46 slots); `team` per slot (0 = home H, 1 = away A). Slots are tracklets, not people.
- `tags.csv`: Smart Tagging rows `code,type,half,team(H/A/blank),start_frame,end_frame`. Load with `keep_default_na=False`. Tags open ~3.5 s before the real event; throw-ins have no team.
- `scored.npy`: bool, True on the 158,857 frames that count.
Load with `np.load('cache/players.npz')`. Rebuild with `pipeline/build_cache.py`.
`.npz`/`.npy` are binary numpy files: not readable in an editor. For Francisco to look at them, run `pipeline/export_preview.py [--start N --frames N]`, which writes readable CSVs to `cache/preview/` (open in Excel). Note `scored` is `scored.npy` (a single array), not `.npz`.

## Common commands
```
# rebuild the cache
.venv\Scripts\python.exe pipeline\build_cache.py

# validate a submission file
.venv\Scripts\python.exe evaluation\validate_submission.py <file.csv>

# score everything in a results folder (public 5 min GT by default)
.venv\Scripts\python.exe evaluation\run_evaluation.py
.venv\Scripts\python.exe evaluation\run_evaluation.py --bootstrap 1000   # is a gap real or luck?

# rebuild official baselines (use --out so the scoreboard isn't polluted)
.venv\Scripts\python.exe starter\baseline.py --out <dir>

# VIEW the video with ground truth, minimap and baseline strip
.venv\Scripts\python.exe tools\make_gt_viewer.py
# then open tools\gt_viewer.html in Chrome or Edge (must stay in tools/ beside ../data)
```
To view our own submission in the viewer, check `make_gt_viewer.py` (`COMPARE_PATH`) for how the second strip is chosen before changing it.

## Labels and scoring
- Per frame: `H` Barcelona has ball, `A` Athletic has ball, `D` dead ball, `X` don't know. H = home = first team in the ATD XML.
- spellF1 (ranking) matches whole possessions; macroF1 is the tie-break. A `D` in the middle of a possession cuts it in two; `X` is forward-filled, so it is free mid-possession and costly at a change.
- Only frames 11,185-90,651 and 110,808-190,197 are scored. Frames before 113,742 in the 2nd half are all `D`.
- Official baseline to beat: smooth50, spellF1 0.2433 on the full match, 0.3256 on the public 5 minutes.

## Code style
Match the organisers' style: module docstring with usage, `# Params` block of constants, small functions with short docstrings, paths via `HERE`/`ROOT` and `games.py`. New v1 code goes in `pipeline/` (cache) and a new `v1/` folder; don't touch `starter/`.

## Known quirks
- Windows Smart App Control blocked pandas 3.1.0 (`interval` DLL) on 2026-11-02. Fixed by pinning pandas 2.2.3 in the venv; scorer reproduces the baselines. Don't upgrade pandas. Don't bypass the security setting if it happens again.
- scikit-learn 1.9.1 is installed but Smart App Control blocks one of its DLLs (`_random`, needed by `sklearn.ensemble`) on 2026-11-05. Not bypassed; Stage 5 uses PyTorch instead.
- Smart App Control started blocking PyTorch's `torch_cpu.dll` on 2026-11-05 afternoon (error shows `shm.dll` "or one of its dependencies"; Windows log `Microsoft-Windows-CodeIntegrity/Operational` names the real file). It ran fine that morning; a clean reinstall of 2.15.0 did not help. Not bypassed.
- PyTorch: `requirements-v1.txt` pins the CPU build, 2.15.0. `import torch` failing with WinError 126 on `c10.dll` meant the Microsoft Visual C++ Redistributable was missing, not Smart App Control; installed with winget on 2026-11-02.
- Notebook 03's `!python` validator cell fails (no `python` on PATH). Harmless.
- `baseline.py` writes to `evaluation/results` by default.
- Run notebooks headlessly with the venv's nbconvert; keep outputs out of the repo.
