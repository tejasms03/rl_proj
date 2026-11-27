"""Duty checks: is each player doing his job right now? (docs/env_wrapper.md §6)

Per step and player the wrapper reports, in `info`:
    duty_ok      float32[11]   1.0 met, 0.0 not met, NaN the duty doesn't apply
    duty_strict  float32[11]   `ok` alone (in position), without the closing-in credit
    duty_active  int[11]       index into DUTY_CODES of the duty judged, -1 none
    duty_target  float32[11,2] the spot the duty heads for, metres, team frame (+y up), NaN none

"Met" means in position (`ok`) or closing in on the target at >= 1 m/s
(0.5 m over the last 0.5 s), as in the duty lab: the time spent getting
there counts as doing the duty. T2 recover is stricter: only being back,
or sprinting back, counts.

Which duty is judged (possession memory, 1 s, decides who has the ball):
    we have it       his attacking duty (not while he is the one on the ball)
    they have it     his defending duty; T2 instead for the first 5 s after we lost it
    loose > 1 s      his defending duty, if it is one that applies then (D1, D7, D8)
Duty 0 and duties outside v1 give NaN, never a fail.

Every check has the same three methods, on the lab's "view" (the raw
left-team observation in pitch units, with ball_owned_team / _player
replaced by the possession memory), the player's slot and his decoded
instruction block (`instructions.decode`):
    active(view, me, ins) -> bool          does the duty apply on this step?
    target(view, me, ins) -> array | None  pitch units
    ok(view, me, ins)     -> bool          in position right now?

Duties whose rules don't depend on the step length reuse the duty lab's
classes (grf_tools/duty_lab/duties.py) with parameters read from the block.
Mark and run in behind use velocity, so they are rewritten here at 0.1 s
steps (x10: the lab ran 0.04 s steps and multiplied by 4).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from grf_tools.duty_lab import duties as lab
from grf_tools.duty_lab.duties import (_METRES, OUR_GOAL, goal_side_of, is_goal_side, lane_point,
                                       metres, our_carrier, our_carrier_index)
from outfield.config import STEP_S
from outfield.env import instructions as ins_mod

N = 11
DUTY_CODES = tuple([f"D{k}" for k in range(9)] + [f"A{k}" for k in range(8)]
                   + [f"T{k}" for k in range(4)] + ["GK_check"])
CODE = {c: i for i, c in enumerate(DUTY_CODES)}

# --- settings (docs/env_wrapper.md §6) ----------------------------------------------------
CLOSING_S, CLOSING_M = 0.5, 0.5        # closing in: >= 0.5 m nearer over 0.5 s (1 m/s)
TRANSITION_S = 5.0                     # T2 recover applies this long after we lose the ball
ZONE_SHIFT, ZONE_SHIFT_MAX_M = 0.3, 10.0   # D1 / A1: anchor shifts 30 % toward the ball, <= 10 m
PRESS_BASE_M, PRESS_SPAN_M = 10.0, 20.0    # D2 applies with their carrier within 10 + 20 x intensity m
LINE_NEAR_M, LINE_SPAN_M = 12.0, 40.0      # D7 / D8: line height 0 -> 12 m from our goal, 1 -> 52 m
RECOVER_NEAR_M = 5.0                   # T2: back = within 5 m of his spot and goal-side of the ball
RECOVER_SPRINT_MS = 5.0                # T2: or sprinting toward it at >= 5 m/s ...
RECOVER_HEADING_DEG = 45.0             # ... within 45 degrees of the direction to it
# Keeper (G may tune these)
GK_SHAPE_SHIFT, GK_SHAPE_SHIFT_MAX_M, GK_SHAPE_RADIUS_M = 0.1, 5.0, 6.0   # keeper A1
GK_SUPPORT_LANE_M = 2.0                # keeper A2: no opponent within 2 m of the passing line
GK_SUPPORT_DIST_M = (10.0, 35.0)       # ... 10-35 m from our carrier
GK_SUPPORT_GOAL_M = 20.0               # ... and still within 20 m of his goal

V1 = {"D1", "D2", "D4", "D5", "D7", "D8", "A1", "A2", "A3", "A4", "T2"}


# --- unit helpers --------------------------------------------------------------------------

def to_raw(p_m) -> np.ndarray:
    """Metres in the team frame (+y up) -> raw pitch units (+y down)."""
    return np.array([p_m[0] / _METRES[0], -p_m[1] / _METRES[1]])


def to_m(p_raw) -> np.ndarray:
    return np.array([p_raw[0] * _METRES[0], -p_raw[1] * _METRES[1]])


def velocity(direction) -> np.ndarray:
    """Pitch units per second from a *_direction entry (movement per 0.1 s step)."""
    return np.asarray(direction)[:2] / STEP_S


def speed_ms(direction) -> float:
    return float(np.linalg.norm(velocity(direction) * _METRES))


def intercept(target, vel, distance_m: float, cap_s: float = 1.5) -> np.ndarray:
    return np.asarray(target)[:2] + vel * min(distance_m / lab.SPRINT_MS, cap_s)


def line_x(height: float) -> float:
    """Line height (0-1) -> x in pitch units."""
    return (-_METRES[0] + LINE_NEAR_M + LINE_SPAN_M * float(np.clip(height, 0, 1))) / _METRES[0]


def offside_line(view) -> float:
    """The lab's offside line; drills with fewer than 2 opponents use the ball and halfway."""
    if len(view["right_team"]) < 2:
        return max(float(view["ball"][0]), 0.0)
    return lab.offside_line(view)


def lane_clearance(view, a, b) -> float:
    if len(view["right_team"]) == 0:
        return math.inf
    return lab.lane_clearance(view, a, b)


def shifted(anchor, ball, share: float, cap_m: float) -> np.ndarray:
    """anchor moved share of the way toward the ball, at most cap_m (lab HoldZone.centre)."""
    anchor = np.asarray(anchor, float)
    shift = (np.asarray(ball)[:2] - anchor) * share
    size = np.linalg.norm(shift * _METRES)
    return anchor + (shift if size <= cap_m else shift * cap_m / size)


def _pos(view, me):
    return np.asarray(view["left_team"][me])[:2]


# --- checks ----------------------------------------------------------------------------------

class Check:
    code = ""
    closing = True          # False: only `ok` counts (T2)

    def active(self, view, me, ins) -> bool:
        return view["ball_owned_team"] == 1

    def target(self, view, me, ins):
        return None

    def ok(self, view, me, ins) -> bool:
        raise NotImplementedError

    def key(self, ins):
        """What identifies the target: a change resets the closing history."""
        return None


class HoldZone(Check):  # D1 (outfield): lab HoldZone, anchor and radius from the block
    code = "D1"

    def active(self, view, me, ins):
        return view["ball_owned_team"] != 0

    def _lab(self, me, ins):
        return lab.HoldZone(me, anchor=to_raw(ins.anchor), radius_m=ins.radius_m or 8.0)

    def target(self, view, me, ins):
        return self._lab(me, ins).centre(view)

    def ok(self, view, me, ins):
        return bool(self._lab(me, ins).ok(view))

    def key(self, ins):
        return ins.anchor


class KeeperZone(HoldZone):  # D1 (keeper): between the posts, no shift toward the ball
    def target(self, view, me, ins):
        return to_raw(ins.anchor)

    def ok(self, view, me, ins):
        return metres(_pos(view, me), self.target(view, me, ins)) <= (ins.radius_m or ins_mod.KEEPER_ZONE_RADIUS_M)


class Press(Check):  # D2: lab Press, range from press intensity
    code = "D2"

    def active(self, view, me, ins):
        c = lab.carrier(view)
        return c is not None and metres(_pos(view, me), c) <= PRESS_BASE_M + PRESS_SPAN_M * ins.press_intensity

    def target(self, view, me, ins):
        return np.asarray(lab.carrier(view))[:2]

    def ok(self, view, me, ins):
        return metres(_pos(view, me), lab.carrier(view)) <= 2.0


class Cover(Check):  # D4: lab Cover, teammate from the block
    code = "D4"

    def active(self, view, me, ins):
        return (view["ball_owned_team"] == 1 and ins.mate is not None and ins.mate != me
                and ins.mate < len(view["left_team"]))

    def target(self, view, me, ins):
        return lab.Cover(me, mate=ins.mate).target(view)

    def ok(self, view, me, ins):
        return bool(lab.Cover(me, mate=ins.mate).ok(view))

    def key(self, ins):
        return ins.mate


class Mark(Check):  # D5: lab Mark with the x10 velocity, opponent from the block
    code = "D5"

    def active(self, view, me, ins):
        return view["ball_owned_team"] == 1 and ins.opp is not None and ins.opp < len(view["right_team"])

    def target(self, view, me, ins):
        man = view["right_team"][ins.opp]
        ahead = intercept(man, velocity(view["right_team_direction"][ins.opp]),
                          metres(_pos(view, me), man), cap_s=1.0)
        return goal_side_of(ahead, 1.5)

    def ok(self, view, me, ins):
        man = view["right_team"][ins.opp]
        return metres(_pos(view, me), man) <= 3.0 and is_goal_side(_pos(view, me), man)

    def key(self, ins):
        return ins.opp


class HoldLine(Check):  # D7: lab HoldLine; line from line height, lane from formation, group from blocks
    code = "D7"

    def active(self, view, me, ins):
        return view["ball_owned_team"] != 0

    def _lab(self, view, me, ins):
        group = [j for j in self.group if j < len(view["left_team"])] or [me]
        return lab.HoldLine(me, line_x=line_x(ins.line_height), lane_y=self.lane_y[me], group=group)

    def target(self, view, me, ins):
        return self._lab(view, me, ins).spot(view)

    def ok(self, view, me, ins):
        if len([j for j in self.group if j != me]) == 0:  # alone on the line: the line height only
            return abs(_pos(view, me)[0] - line_x(ins.line_height)) * _METRES[0] <= 2.0
        return bool(self._lab(view, me, ins).ok(view))

    def key(self, ins):
        return round(ins.line_height, 3)


class Sweep(Check):  # D8 (keeper): lab Sweep, line from line height
    code = "D8"

    def _lab(self, me, ins):
        return lab.Sweep(me, line_x=line_x(ins.line_height))

    def active(self, view, me, ins):
        return bool(self._lab(me, ins).active(view))

    def target(self, view, me, ins):
        return np.asarray(self._lab(me, ins).target(view))[:2]

    def ok(self, view, me, ins):
        return bool(self._lab(me, ins).ok(view))

    def key(self, ins):
        return round(ins.line_height, 3)


class _Attacking(Check):
    def active(self, view, me, ins):
        return view["ball_owned_team"] == 0 and our_carrier_index(view) != me


class HoldShape(_Attacking, HoldZone):  # A1 (outfield): as D1, while we have it
    code = "A1"
    active = _Attacking.active


class KeeperShape(HoldShape):  # A1 (keeper): a spot in front of goal, 10 % shift, <= 5 m
    def target(self, view, me, ins):
        return shifted(to_raw(ins.anchor), view["ball"], GK_SHAPE_SHIFT, GK_SHAPE_SHIFT_MAX_M)

    def ok(self, view, me, ins):
        return metres(_pos(view, me), self.target(view, me, ins)) <= (ins.radius_m or GK_SHAPE_RADIUS_M)


class Support(_Attacking):  # A2 (outfield): lab Support.ok (open, 8-15 m from our carrier)
    code = "A2"

    def ok(self, view, me, ins):
        c, pos = our_carrier(view), _pos(view, me)
        return 8.0 <= metres(pos, c) <= 15.0 and lane_clearance(view, c, pos) >= 1.5


class KeeperSupport(_Attacking):  # A2 (keeper): available for a back pass
    code = "A2"

    def active(self, view, me, ins):
        c = our_carrier(view)
        return super().active(view, me, ins) and c is not None and c[0] < 0.0

    def _fine(self, view, me):
        c, pos = np.asarray(our_carrier(view))[:2], _pos(view, me)
        lo, hi = GK_SUPPORT_DIST_M
        return (lane_clearance(view, c, pos) >= GK_SUPPORT_LANE_M and pos[0] < c[0]
                and lo <= metres(pos, c) <= hi and metres(pos, OUR_GOAL) <= GK_SUPPORT_GOAL_M)

    def target(self, view, me, ins):
        if self._fine(view, me):
            return _pos(view, me)
        c = np.asarray(our_carrier(view))[:2]
        spot = lab.toward(c, OUR_GOAL, 20.0)
        if metres(spot, OUR_GOAL) > GK_SUPPORT_GOAL_M:
            spot = lab.toward(OUR_GOAL, spot, GK_SUPPORT_GOAL_M)
        return spot

    def ok(self, view, me, ins):
        return self._fine(view, me)


class Width(_Attacking):  # A3: lab Width; the side is the anchor's y sign (+y up = raw -y)
    code = "A3"

    def _lab(self, me, ins):
        side = "left" if (ins.anchor or (0.0, 0.0))[1] >= 0 else "right"  # raw +y (down) is "right"
        return lab.Width(me, side=side)

    def target(self, view, me, ins):
        line = offside_line(view)
        w = self._lab(me, ins)
        x = min(view["ball"][0] + 5.0 / _METRES[0], line - 1.0 / _METRES[0])
        return np.array([x, w.edge() - math.copysign(3.0 / _METRES[1], w.edge())])

    def ok(self, view, me, ins):
        return bool(self._lab(me, ins).ok(view))

    def key(self, ins):
        return None if ins.anchor is None else ins.anchor[1] >= 0


class RunInBehind(_Attacking):  # A4: lab RunInBehind.ok with the x10 velocity; channel = anchor y
    code = "A4"

    def target(self, view, me, ins):
        channel = to_raw(ins.anchor or (0.0, 0.0))[1]
        return np.array([offside_line(view) - 1.0 / _METRES[0], channel])

    def ok(self, view, me, ins):
        pos, line = _pos(view, me), offside_line(view)
        ready = pos[0] < line and (line - pos[0]) * _METRES[0] <= 3.0
        beyond = pos[0] > line and velocity(view["left_team_direction"][me])[0] * _METRES[0] > 3.0
        return bool(ready or beyond)

    def key(self, ins):
        return ins.anchor


class Recover(Check):  # T2 (new): back behind the ball to his defensive spot, at full speed
    code = "T2"
    closing = False

    def __init__(self, spot: Callable):
        self.spot = spot  # (view, me, ins) -> his defensive spot, pitch units

    def active(self, view, me, ins):
        return True  # the checker decides when (first 5 s after we lose it)

    def target(self, view, me, ins):
        return self.spot(view, me, ins)

    def ok(self, view, me, ins):
        pos, spot = _pos(view, me), self.target(view, me, ins)
        if metres(pos, spot) <= RECOVER_NEAR_M and is_goal_side(pos, view["ball"]):
            return True
        v = velocity(view["left_team_direction"][me]) * _METRES
        want = (np.asarray(spot)[:2] - pos) * _METRES
        if np.linalg.norm(v) < RECOVER_SPRINT_MS or np.linalg.norm(want) < 1e-6:
            return False
        cos = float(np.dot(v, want) / (np.linalg.norm(v) * np.linalg.norm(want)))
        return cos >= math.cos(math.radians(RECOVER_HEADING_DEG))


DEFENDING = {1: HoldZone(), 2: Press(), 4: Cover(), 5: Mark(), 7: HoldLine()}
ATTACKING = {1: HoldShape(), 2: Support(), 3: Width(), 4: RunInBehind()}
KEEPER_DEFENDING = {1: KeeperZone(), 8: Sweep()}
KEEPER_ATTACKING = {1: KeeperShape(), 2: KeeperSupport()}


# --- the checker -------------------------------------------------------------------------------

class DutyChecker:
    """Holds the per-player closing-in history; call once per step.

    keeper_check: optional fn(view, slot, instruction) -> bool | None (G's B4,
    e.g. shot positioning). When given, it scores slot 0 while his defending
    duty is D0 and they have the ball: True 1.0, False 0.0, None NaN.
    """

    def __init__(self, keeper_check: Callable | None = None):
        self.keeper_check = keeper_check
        self.window = int(round(CLOSING_S / STEP_S))
        self.recover = Recover(self._defensive_spot)
        self.reset()

    def reset(self):
        self.history: list[list[float]] = [[] for _ in range(N)]
        self.keys: list = [None] * N
        self.last_team = -1        # last team that had the ball (possession memory)
        self.since_lost = math.inf  # steps since we lost it

    def _defensive_spot(self, view, me, ins):
        check = self._checks(me)[0].get(ins.defending)
        if check is not None and check.active(view, me, ins):
            spot = check.target(view, me, ins)
            if spot is not None:
                return np.asarray(spot)[:2]
        home = self.formation_raw[me]
        return shifted(home, view["ball"], ZONE_SHIFT, ZONE_SHIFT_MAX_M)

    def _checks(self, me):
        return (KEEPER_DEFENDING, KEEPER_ATTACKING) if me == 0 else (DEFENDING, ATTACKING)

    def which(self, view, me, ins):
        """(code, check) judged this step for player me, or (None, None)."""
        defending, attacking = self._checks(me)
        team = view["ball_owned_team"]
        if team == 0:
            check = attacking.get(ins.attacking)
            return (f"A{ins.attacking}", check) if check else (None, None)
        if me != 0 and team == 1 and ins.transition == 2 and self.since_lost * STEP_S < TRANSITION_S:
            return "T2", self.recover
        if me == 0 and ins.defending == 0 and team == 1 and self.keeper_check is not None:
            return "GK_check", None
        check = defending.get(ins.defending)
        return (f"D{ins.defending}", check) if check else (None, None)

    def __call__(self, view, blocks: np.ndarray, present: np.ndarray, formation_m: np.ndarray):
        """view: raw left-team obs with the possession memory as owner;
        blocks: [11, >=85] instruction blocks; formation_m: [11, 2] kick-off
        spots in metres. Returns ok, strict, active, target (see module doc)."""
        team = view["ball_owned_team"]
        if team == 1 and self.last_team == 0:
            self.since_lost = 0
        elif self.since_lost != math.inf:
            self.since_lost += 1
        if team != -1:
            self.last_team = team
        self.formation_raw = [to_raw(p) for p in formation_m]
        HoldLine.lane_y = [p[1] for p in self.formation_raw]
        decoded = [ins_mod.decode(blocks[i]) if present[i] else None for i in range(N)]
        HoldLine.group = [j for j in range(1, N) if decoded[j] is not None and decoded[j].defending == 7]

        ok = np.full(N, np.nan, np.float32)
        strict = np.full(N, np.nan, np.float32)
        active = np.full(N, -1, np.int64)
        target = np.full((N, 2), np.nan, np.float32)
        for i in range(N):
            ins = decoded[i]
            code, check = (None, None) if ins is None else self.which(view, i, ins)
            if code == "GK_check":
                result = self.keeper_check(view, i, ins)
                if result is not None:
                    ok[i] = strict[i] = float(bool(result))
                    active[i] = CODE[code]
                self._forget(i)
                continue
            if check is None or not check.active(view, i, ins):
                self._forget(i)
                continue
            spot = check.target(view, i, ins)
            good = bool(check.ok(view, i, ins))
            met = good
            key = (code, check.key(ins))
            if key != self.keys[i]:
                self.history[i], self.keys[i] = [], key
            if spot is not None:
                d = metres(_pos(view, i), spot)
                h = self.history[i]
                h.append(d)
                del h[:-(self.window + 1)]
                if check.closing and len(h) > self.window and h[0] - d >= CLOSING_M:
                    met = True
                target[i] = to_m(np.asarray(spot)[:2])
            ok[i], strict[i] = float(met), float(good)
            active[i] = CODE[code]
        return ok, strict, active, target

    def _forget(self, i):
        self.history[i], self.keys[i] = [], None
