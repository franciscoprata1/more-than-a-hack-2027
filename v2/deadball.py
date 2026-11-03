"""Stage 2 cues: what the tracking says about a dead ball, one frame at a time.

Used by v2/stage2.py. Every threshold here comes from the Laws of the Game or football sense
(Vidal-Codina et al. 2022 read set pieces from player shapes the same way); the few that are tuned
on degraded Sportec are passed in by stage2.py.

    cues(m)  -> dict of per-frame arrays:
      crowd      mean speed of the tracked players, over 1 s (m/s): play stops, players walk
      out        whole ball seen past a line
      throw      a player standing past a touchline (the thrower)
      corner     a player standing at a corner flag (the taker)
      kickoff    both teams wholly in their own halves, someone at the centre spot
      goalkick   ball still in a goal area, no attacker in that penalty area
      freekick   ball still for 1 s outside the penalty areas, no opponent within FK_CLEAR_M
      restart    the ball goes from still to kicked: candidate end of a stoppage
"""

import numpy as np

# Params
FPS = 25
CROWD_WINDOW = FPS               # 1 s mean of the players' speed
STAND_SPEED = 2.0                # m/s: a taker waiting, not a player running past
OUTSIDE_M = 0.5                  # a thrower stands behind the line
CORNER_M = 2.5                   # taker within this of the flag
CENTRE_M = 3.0
OWN_HALF_TOL_M = 1.0
GOAL_AREA = (5.5 + 1.0, 9.16 + 1.0)        # depth from the byline, half-width from the middle (+1 m)
PENALTY_AREA = (16.5 + 1.0, 20.16 + 1.0)
STILL_SPEED = 1.0                # m/s: placed or held
STILL_FOR = FPS // 2
KICK_SPEED = 5.0                 # m/s: the restart was taken
LOOK = FPS                        # within 1 s of resting the ball is kicked (our tracker shows the kick
                                 # over several frames: 6 frames found 3 restarts on our match, 84 with 1 s)
FK_STILL_FOR = FPS
FK_CLEAR_M = 5.0                 # Law 13 says 9.15 m; walls creep in and tracking is rough


def run_mean(v, w):
    """Centred moving mean over w frames, NaN-aware."""
    ok = ~np.isnan(v)
    c = np.convolve(np.where(ok, v, 0.0), np.ones(w), mode="same")
    n = np.convolve(ok.astype(float), np.ones(w), mode="same")
    with np.errstate(invalid="ignore"):
        return np.where(n > 0, c / np.maximum(n, 1), np.nan)


def held_for(mask, k):
    """True where mask has been True for the last k frames (this one included)."""
    c = np.concatenate(([0], np.cumsum(mask)))
    i = np.arange(len(mask))
    return (c[i + 1] - c[np.maximum(0, i + 1 - k)]) >= np.minimum(k, i + 1)


def own_sides(m):
    """Per half, which end (0 = low x, 1 = high x) each team defends: its mean x over the half."""
    px, team, half = m["player_x"], m["team"], m["half"]
    sides = {}
    for h in (1, 2):
        rows = half == h
        for t in (0, 1):
            mx = np.nanmean(px[rows][:, team == t])
            sides[(h, t)] = 0 if mx < m["pitch"][0] / 2 else 1
    return sides


def cues(m):
    L, W = m["pitch"]
    px, py = m["player_x"].astype(np.float64), m["player_y"].astype(np.float64)
    ps = np.hypot(m["player_vx"], m["player_vy"])
    team, half = m["team"], m["half"]
    bx, by = m["ball_x"].astype(np.float64), m["ball_y"].astype(np.float64)
    seen = (m["ball_state"] < 2) & ~np.isnan(bx) & (half > 0)
    bs = np.where(seen, m["ball_speed"], np.nan)
    with np.errstate(invalid="ignore"):
        crowd = run_mean(np.nanmean(ps, axis=1), CROWD_WINDOW)
        standing = ps < STAND_SPEED

        throw = (((py < -OUTSIDE_M) | (py > W + OUTSIDE_M)) & (px > 0) & (px < L) & standing).any(axis=1)
        flags = [(0, 0), (0, W), (L, 0), (L, W)]
        corner = np.zeros(len(bx), dtype=bool)
        for fx, fy in flags:
            corner |= ((np.hypot(px - fx, py - fy) < CORNER_M) & standing).any(axis=1)

        sides = own_sides(m)
        kickoff = np.zeros(len(bx), dtype=bool)
        tracked = ~np.isnan(px)
        for h in (1, 2):
            rows = half == h
            ok = np.ones(rows.sum(), dtype=bool)
            for t in (0, 1):
                x = px[rows][:, team == t]
                if sides[(h, t)] == 0:
                    wrong = x > L / 2 + OWN_HALF_TOL_M
                else:
                    wrong = x < L / 2 - OWN_HALF_TOL_M
                ok &= ~wrong.any(axis=1)
            at_centre = (np.hypot(px[rows] - L / 2, py[rows] - W / 2) < CENTRE_M).any(axis=1)
            enough = tracked[rows].sum(axis=1) >= 16
            kickoff[rows] = ok & at_centre & enough

        still = seen & (bs < STILL_SPEED)
        in_goal_area = np.zeros(len(bx), dtype=bool)
        in_pen_area = np.zeros(len(bx), dtype=bool)
        goalkick = np.zeros(len(bx), dtype=bool)
        for end, x0 in ((0, 0.0), (1, L)):
            ga = (np.abs(bx - x0) < GOAL_AREA[0]) & (np.abs(by - W / 2) < GOAL_AREA[1])
            pa = (np.abs(bx - x0) < PENALTY_AREA[0]) & (np.abs(by - W / 2) < PENALTY_AREA[1])
            in_goal_area |= ga
            in_pen_area |= pa
            for h in (1, 2):
                rows = half == h
                attackers = [t for t in (0, 1) if sides[(h, t)] != end]
                inside = (np.abs(px - x0) < PENALTY_AREA[0]) & (np.abs(py - W / 2) < PENALTY_AREA[1])
                att_in = inside[:, team == attackers[0]].any(axis=1) if attackers else np.zeros(len(bx), bool)
                goalkick |= rows & ga & still & ~att_in

        d = np.hypot(px - bx[:, None], py - by[:, None])
        near = np.nanargmin(np.where(np.isnan(d), np.inf, d), axis=1)
        taker_team = team[near]
        opp = np.full(len(bx), np.inf)
        for t in (0, 1):
            dt = np.where(np.isnan(d[:, team == t]), np.inf, d[:, team == t]).min(axis=1)
            opp = np.where(taker_team != t, dt, opp)
        freekick = held_for(still, FK_STILL_FOR) & ~in_pen_area & (opp > FK_CLEAR_M)

        kicked = np.zeros(len(bx), dtype=bool)
        fast = seen & (bs > KICK_SPEED)
        for k in range(1, LOOK + 1):
            kicked[:-k] |= fast[k:]
        restart = held_for(still, STILL_FOR) & kicked
        restart &= ~np.r_[False, restart[:-1]]

        L_, W_ = L, W
        out = seen & ((bx < -0.11) | (bx > L_ + 0.11) | (by < -0.11) | (by > W_ + 0.11))
    return dict(crowd=crowd, out=out, throw=throw, corner=corner, kickoff=kickoff,
                goalkick=goalkick, freekick=freekick, restart=restart, seen=seen)


SHAPES = ("throw", "corner", "kickoff", "goalkick", "freekick")
