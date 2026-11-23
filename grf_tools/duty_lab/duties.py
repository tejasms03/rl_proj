"""Scripted duties and their checks, built from a few shared blocks.

Each duty is a class bound to one of our players. Per step:
- control(own_obs) -> one game action for that player. Directions, sprint,
  pressure and keeper rush are sticky in the engine, so `_driver` sends one
  change per step until the player's sticky state matches what we want.
  own_obs must be the player's own observation (sticky state is per player).
- check(obs) -> True / False while the duty applies, None when it doesn't.
  True = in position (ok) or closing on the duty's spot at 1 m/s or more:
  the time spent getting there counts as doing the duty. strict(obs) is ok
  alone.
- measure(obs) -> one plain number in metres for the summary (or None).
- refs(obs) -> points to draw (the carrier, the marked man, ...).
check/measure/refs take any left-team observation (absolute coordinates).

Distances are in metres via `_METRES` (same factors as ps_gamepad).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gfootball.env import football_action_set

_METRES = np.array([54.4, 83.6])  # pitch units -> metres, x and y
_ACTIONS = football_action_set.get_action_set({"action_set": "full"})
ACTION = {a._name: i for i, a in enumerate(_ACTIONS)}
_STICKY = [a._name for a in football_action_set.get_sticky_actions({"action_set": "full"})]
_DIRECTIONS = ["right", "bottom_right", "bottom", "bottom_left",
               "left", "top_left", "top", "top_right"]  # 45 deg steps from +x, y down
OUR_GOAL = np.array([-1.0, 0.0])
SPRINT_MS = 7.5  # our top speed, for intercept leads


# --- building blocks -------------------------------------------------------

def metres(a, b) -> float:
    return float(np.linalg.norm((np.asarray(a)[:2] - np.asarray(b)[:2]) * _METRES))


def toward(a, b, gap_m: float) -> np.ndarray:
    """The point gap_m from a in the direction of b."""
    a, b = np.asarray(a)[:2], np.asarray(b)[:2]
    step = (b - a) * _METRES
    return a + step / (np.linalg.norm(step) + 1e-9) * gap_m / _METRES


def goal_side_of(target, gap_m: float) -> np.ndarray:
    """gap_m from target on the line to our goal."""
    return toward(target, OUR_GOAL, gap_m)


def is_goal_side(me, other) -> bool:
    return metres(me, OUR_GOAL) < metres(other, OUR_GOAL)


def lane_point(me, a, b) -> np.ndarray:
    """Closest point to me on the segment a-b (a passing lane)."""
    a, b, me = (np.asarray(p)[:2] * _METRES for p in (a, b, me))
    t = np.clip(np.dot(me - a, b - a) / (np.dot(b - a, b - a) + 1e-9), 0.0, 1.0)
    return (a + t * (b - a)) / _METRES


def carrier_index(obs) -> int | None:
    """Their player on the ball, if they have it."""
    return int(obs["ball_owned_player"]) if obs["ball_owned_team"] == 1 else None


def carrier(obs) -> np.ndarray | None:
    i = carrier_index(obs)
    return None if i is None else obs["right_team"][i]


def our_carrier_index(obs) -> int | None:
    """Our player on the ball, if we have it."""
    return int(obs["ball_owned_player"]) if obs["ball_owned_team"] == 0 else None


def our_carrier(obs) -> np.ndarray | None:
    i = our_carrier_index(obs)
    return None if i is None else obs["left_team"][i]


def offside_line(obs) -> float:
    """x an attacker of ours must not pass when the ball is played: their
    second-last player or the ball, whichever is further forward, and never
    in our own half."""
    second_last = sorted(p[0] for p in obs["right_team"])[-2]
    return max(second_last, obs["ball"][0], 0.0)


def their_lines(obs) -> tuple[float, float]:
    """Mean x of their back four and of their midfield four (the outfielders
    nearest their goal, then the next four)."""
    xs = sorted((p[0] for p in obs["right_team"][1:]), reverse=True)
    return float(np.mean(xs[:4])), float(np.mean(xs[4:8]))


def lane_clearance(obs, a, b) -> float:
    """Metres from the a-b passing line to the nearest of their players."""
    return min(metres(p, lane_point(p, a, b)) for p in obs["right_team"])


def velocity(direction) -> np.ndarray:
    """Pitch units per second from an observation's *_direction entry (the
    entry is the movement over 0.25 s, measured at 40 ms steps)."""
    return np.asarray(direction) * 4.0


def intercept(target, vel, distance_m: float, cap_s: float = 1.5) -> np.ndarray:
    """Where a moving target will be by the time we can get there."""
    return np.asarray(target) + vel * min(distance_m / SPRINT_MS, cap_s)


def ball_path(obs, t: float) -> np.ndarray:
    """Where a free ball will be in t seconds: it slows as it rolls."""
    tau = 1.5  # seconds; rough rolling slow-down
    return (np.asarray(obs["ball"][:2])
            + velocity(obs["ball_direction"][:2]) * tau * (1.0 - math.exp(-t / tau)))


def time_to_ball(obs, pos, horizon_s: float, reach_m: float = 1.0) -> tuple[float, np.ndarray] | None:
    """When (and where) a player at pos can first get to the moving ball,
    sprinting, within horizon_s; None if he can't."""
    for k in range(int(horizon_s * 25) + 1):
        t = k / 25
        spot = ball_path(obs, t)
        if metres(pos, spot) <= SPRINT_MS * t + reach_m:
            return t, spot
    return None


def first_to_ball(obs, me, margin_s: float = 0.2, horizon_s: float = 1.5) -> np.ndarray | None:
    """A free ball (a pass in flight or a loose ball) that we get to at least
    margin_s before any of them, following its path: where to meet it, else
    None. The game's AI players go for such balls by themselves; a scripted
    player must be told to, and only when he will win the race."""
    if obs.get("raw_ball_owned_team", obs["ball_owned_team"]) != -1:
        return None
    ours = time_to_ball(obs, me, horizon_s)
    if ours is None:
        return None
    theirs = [time_to_ball(obs, p, horizon_s) for p in obs["right_team"]]
    first_them = min((t for t, _ in filter(None, theirs)), default=math.inf)
    return ours[1] if ours[0] + margin_s <= first_them else None


_MIRROR = {"left": "right", "right": "left", "top": "bottom", "bottom": "top",
           "top_left": "bottom_right", "bottom_right": "top_left",
           "top_right": "bottom_left", "bottom_left": "top_right"}


def mirrored(direction: str) -> str:
    """The same screen direction for the right team, whose actions are mirrored."""
    return _MIRROR[direction]


def direction_to(me, point) -> str:
    dx, dy = (np.asarray(point)[:2] - np.asarray(me)[:2]) * _METRES
    return _DIRECTIONS[round(math.atan2(dy, dx) / (math.pi / 4)) % 8]


@dataclass
class Want:
    """What a controller wants the player's sticky state to be this step."""
    direction: str | None = None  # None = stand still
    sprint: bool = False
    pressure: bool = False
    keeper_rush: bool = False
    builtin: bool = False  # hand this step to the game's own AI


def _driver(obs, want: Want) -> int:
    held = {name for name, on in zip(_STICKY, obs["sticky_actions"]) if on}
    held_dir = next((d for d in _DIRECTIONS if d in held), None)
    buttons = (("pressure", want.pressure), ("keeper_rush", want.keeper_rush))
    if want.builtin:
        for name in ("pressure", "keeper_rush", "sprint"):
            if name in held:
                return ACTION[f"release_{name}"]
        return ACTION["release_direction"] if held_dir else ACTION["builtin_ai"]
    for name, on in buttons:
        if on != (name in held):
            return ACTION[name if on else f"release_{name}"]
    if want.direction != held_dir:
        return ACTION[want.direction] if want.direction else ACTION["release_direction"]
    if want.sprint != ("sprint" in held):
        return ACTION["sprint" if want.sprint else "release_sprint"]
    return ACTION["idle"]


def go_to(obs, me_pos, point, settle_m: float = 0.6, sprint_m: float = 4.0, **buttons) -> int:
    """Move to a spot: sprint when far, walk when near, stand when there."""
    d = metres(me_pos, point)
    if d <= settle_m:
        return _driver(obs, Want(**buttons))
    return _driver(obs, Want(direction=direction_to(me_pos, point), sprint=d > sprint_m, **buttons))


# --- duties ----------------------------------------------------------------

class Duty:
    label = ""
    unit = ""  # what measure() reports
    WINDOW = 5  # steps of 40 ms over which "closing" is judged

    def __init__(self, me: int, **params):
        self.me = me
        self.p = params
        self.history: list[float] = []

    def pos(self, obs) -> np.ndarray:
        return obs["left_team"][self.me]

    def active(self, obs) -> bool:
        return obs["ball_owned_team"] == 1  # defending duties: while they have it

    def control(self, own_obs) -> int:
        if not self.active(own_obs):
            return _driver(own_obs, Want(builtin=True))
        return self.act(own_obs)

    def act(self, obs) -> int:
        raise NotImplementedError

    def check(self, obs) -> bool | None:
        if not self.active(obs):
            self.history.clear()
            return None
        d = metres(self.pos(obs), self.target(obs))
        self.history.append(d)
        closing = (len(self.history) > self.WINDOW
                   and self.history[-self.WINDOW - 1] - d >= 0.2)  # 1 m/s
        return self.ok(obs) or closing

    def strict(self, obs) -> bool | None:
        return self.ok(obs) if self.active(obs) else None

    def ok(self, obs) -> bool:
        raise NotImplementedError

    def target(self, obs) -> np.ndarray:
        """The spot the duty is heading for."""
        raise NotImplementedError

    def measure(self, obs) -> float | None:
        return None

    def refs(self, obs) -> list:
        c = carrier(obs)
        return [] if c is None else [c]


class _CarrierInRange(Duty):
    """Applies while their carrier is within range_m of us."""

    def active(self, obs) -> bool:
        c = carrier(obs)
        return c is not None and metres(self.pos(obs), c) <= self.p.get("range_m", 20.0)

    def measure(self, obs):
        return metres(self.pos(obs), carrier(obs)) if self.active(obs) else None


class Press(_CarrierInRange):
    """Close down the carrier: run at where he will be, arriving goal-side, then
    the pressure button for the last steps. Only within range (20 m): a lone
    presser chasing a pass 30 m away is wasted running."""

    label = "PRESS the ball carrier (within 20 m)"
    unit = "m to carrier"
    BUTTON_M = 2.5

    def act(self, obs):
        pos, c = self.pos(obs), carrier(obs)
        d = metres(pos, c)
        if d <= self.BUTTON_M:
            return _driver(obs, Want(pressure=True))
        aim = intercept(c, velocity(obs["right_team_direction"][carrier_index(obs)]), d)
        return _driver(obs, Want(direction=direction_to(pos, goal_side_of(aim, 1.0)), sprint=True))

    def ok(self, obs):
        return metres(self.pos(obs), carrier(obs)) <= 2.0

    def target(self, obs):
        return carrier(obs)


class Contain(_CarrierInRange):
    """Stay goal-side of the carrier, 1.5-3 m off him, never diving in."""

    label = "CONTAIN the carrier (goal-side, 1.5-3 m, no tackle)"
    unit = "m to carrier"
    LEAD_S = 0.6
    SPRINT_M = 8.0

    def target(self, obs):
        # Where he'll be shortly: backing off at his speed keeps the gap.
        ahead = carrier(obs) + self.LEAD_S * velocity(obs["right_team_direction"][carrier_index(obs)])
        return goal_side_of(ahead, 2.2)

    def act(self, obs):
        # Sprint only while far: players carry momentum, and arriving at full
        # speed runs you into him.
        return go_to(obs, self.pos(obs), self.target(obs), settle_m=0.3, sprint_m=self.SPRINT_M)

    def ok(self, obs):
        c = carrier(obs)
        return 1.5 <= metres(self.pos(obs), c) <= 3.0 and is_goal_side(self.pos(obs), c)


class Mark(Duty):
    """Goal-side of one opponent, within 3 m, while they have the ball."""

    label = "MARK their striker (goal-side, within 3 m)"
    unit = "m to marked man"

    def man(self, obs):
        return obs["right_team"][self.p["opp"]]

    def target(self, obs):
        # Lead him: a striker running at goal is gone if you aim where he is.
        pos, man = self.pos(obs), self.man(obs)
        ahead = intercept(man, velocity(obs["right_team_direction"][self.p["opp"]]),
                          metres(pos, man), cap_s=1.0)
        return goal_side_of(ahead, 1.5)

    def act(self, obs):
        return go_to(obs, self.pos(obs), self.target(obs), sprint_m=2.0)

    def ok(self, obs):
        return metres(self.pos(obs), self.man(obs)) <= 3.0 and is_goal_side(self.pos(obs), self.man(obs))

    def measure(self, obs):
        return metres(self.pos(obs), self.man(obs)) if self.active(obs) else None

    def refs(self, obs):
        return [self.man(obs)]


class Cover(Duty):
    """5-10 m behind a teammate, between him and our goal (backs up a presser)."""

    label = "COVER the pressing centre-back (5-10 m behind him)"
    unit = "m to teammate"

    def mate(self, obs):
        return obs["left_team"][self.p["mate"]]

    def target(self, obs):
        return goal_side_of(self.mate(obs), 7.0)

    def act(self, obs):
        return go_to(obs, self.pos(obs), self.target(obs))

    def ok(self, obs):
        d = metres(self.pos(obs), self.mate(obs))
        return 5.0 <= d <= 10.0 and is_goal_side(self.pos(obs), self.mate(obs))

    def measure(self, obs):
        return metres(self.pos(obs), self.mate(obs)) if self.active(obs) else None

    def refs(self, obs):
        return [self.mate(obs)]


class ScreenLane(Duty):
    """Stand in the passing lane from their carrier to one opponent (within
    2 m) while the ball is still in front of you; go for a pass in flight only
    when you win the race to it (first_to_ball)."""

    label = "SCREEN the lane to their midfielder"
    unit = "m off the lane"
    IN_FRONT_M = 4.0

    def man(self, obs):
        return obs["right_team"][self.p["opp"]]

    def source(self, obs):
        """Where a pass to him would come from: their carrier, or a loose ball."""
        c = carrier(obs)
        return c if c is not None else np.asarray(obs["ball"])

    def active(self, obs):
        # Only while one of theirs has it (passes in flight included, through
        # possession memory). A truly loose ball goes back to the game's AI,
        # which contests it better than screening from the ball does (13 vs 7
        # receptions by him over 12 seeds).
        i = carrier_index(obs)
        if i is None or i == self.p["opp"]:
            return False
        src = self.source(obs)
        # Only while the ball is still in front of him: once it is past him,
        # the screen is broken and his job is to get back, not this.
        return (metres(self.pos(obs), src) <= self.p.get("range_m", 30.0)
                and is_goal_side(self.pos(obs), src))

    def lane(self, obs):
        # Stand in front of the man, on the ball's side: he moves far less than
        # the ball does, so the spot stays put while they pass it around.
        src, m = self.source(obs), self.man(obs)
        return toward(m, src, min(self.IN_FRONT_M, metres(m, src) / 2))

    def target(self, obs):
        return self.lane(obs)

    def act(self, obs):
        pos = self.pos(obs)
        meet = first_to_ball(obs, pos)
        if meet is not None:  # a free ball I win the race to: take it
            return _driver(obs, Want(direction=direction_to(pos, meet), sprint=True))
        return go_to(obs, pos, self.lane(obs), settle_m=0.5)

    def off_lane(self, obs):
        return metres(self.pos(obs), lane_point(self.pos(obs), self.source(obs), self.man(obs)))

    def ok(self, obs):
        return self.off_lane(obs) <= 2.0

    def measure(self, obs):
        return self.off_lane(obs) if self.active(obs) else None

    def refs(self, obs):
        return [self.man(obs)] + super().refs(obs)


class HoldLine(Duty):
    """Stand at the line height, level with the rest of the line (within 2 m),
    keeping to your lane but shading toward the ball."""

    label = "HOLD THE LINE 30 m from goal (back four)"
    unit = "m off the line"

    def active(self, obs):
        return obs["ball_owned_team"] != 0

    def spot(self, obs):
        return np.array([self.p["line_x"], 0.75 * self.p["lane_y"] + 0.25 * obs["ball"][1]])

    def target(self, obs):
        return self.spot(obs)

    def act(self, obs):
        return go_to(obs, self.pos(obs), self.spot(obs), sprint_m=5.0)

    def ok(self, obs):
        x = self.pos(obs)[0]
        others = [obs["left_team"][i][0] for i in self.p["group"] if i != self.me]
        return (abs(x - self.p["line_x"]) * _METRES[0] <= 2.0
                and abs(x - np.mean(others)) * _METRES[0] <= 2.0)

    def measure(self, obs):
        return abs(self.pos(obs)[0] - self.p["line_x"]) * _METRES[0] if self.active(obs) else None

    def refs(self, obs):
        return []


class HoldZone(Duty):
    """Stay within radius of an anchor that shifts 30 % toward the ball (max 10 m)."""

    label = "HOLD ZONE in front of the back four (8 m)"
    unit = "m from zone centre"

    def active(self, obs):
        return obs["ball_owned_team"] != 0

    def centre(self, obs):
        anchor = np.asarray(self.p["anchor"])
        shift = (obs["ball"][:2] - anchor) * 0.3
        size = np.linalg.norm(shift * _METRES)
        return anchor + (shift if size <= 10 else shift * 10 / size)

    def target(self, obs):
        return self.centre(obs)

    def act(self, obs):
        return go_to(obs, self.pos(obs), self.centre(obs), settle_m=0.4 * self.p.get("radius_m", 8))

    def ok(self, obs):
        return metres(self.pos(obs), self.centre(obs)) <= self.p.get("radius_m", 8)

    def measure(self, obs):
        return metres(self.pos(obs), self.centre(obs)) if self.active(obs) else None

    def refs(self, obs):
        return [self.centre(obs)]


class Sweep(Duty):
    """Keeper well off his line behind a high back line; rushes balls played
    in behind it."""

    label = "KEEPER SWEEPS behind the high line"
    unit = "m off goal line"

    def active(self, obs):
        return obs["ball_owned_team"] != 0 and obs["ball"][0] < 0.2

    def in_behind(self, obs):
        ball = obs["ball"]
        return (obs["ball_owned_team"] == -1 and ball[0] < self.p["line_x"]
                and obs["ball_direction"][0] < 0)

    def target(self, obs):
        ball = obs["ball"]
        if self.in_behind(obs):
            return ball
        return np.array([-1.0 + 0.4 * (self.p["line_x"] + 1.0), 0.3 * ball[1]])

    def act(self, obs):
        pos = self.pos(obs)
        if self.in_behind(obs):
            return _driver(obs, Want(direction=direction_to(pos, obs["ball"]), sprint=True,
                                     keeper_rush=True))
        return go_to(obs, pos, self.target(obs))

    def ok(self, obs):
        x = self.pos(obs)[0]
        return self.in_behind(obs) or ((x + 1.0) * _METRES[0] >= 10.0 and x < self.p["line_x"])

    def measure(self, obs):
        return (self.pos(obs)[0] + 1.0) * _METRES[0] if self.active(obs) else None

    def refs(self, obs):
        return []


class DoubleTeam(_CarrierInRange):
    """Two players on the carrier: one arrives goal-side, one from the side."""

    label = "DOUBLE TEAM the carrier (centre-back + midfielder)"
    unit = "m, farther of the two"

    def partner(self, obs):
        return obs["left_team"][self.p["partner"]]

    def active(self, obs):
        # A joint decision: both go if he is in range of either, or only one commits.
        c = carrier(obs)
        reach = self.p.get("range_m", 20.0)
        return c is not None and min(metres(self.pos(obs), c), metres(self.partner(obs), c)) <= reach

    def target(self, obs):
        pos, c = self.pos(obs), carrier(obs)
        if self.p["role"] == "goal":
            return goal_side_of(c, 1.0)
        # The other one comes from the side: square to the carrier, on our side of him.
        to_goal = (OUR_GOAL - c[:2]) * _METRES
        square = np.array([-to_goal[1], to_goal[0]]) / (np.linalg.norm(to_goal) + 1e-9)
        if np.dot(square, (pos[:2] - c[:2]) * _METRES) < 0:
            square = -square
        return c[:2] + square * 1.5 / _METRES

    def act(self, obs):
        pos = self.pos(obs)
        if metres(pos, carrier(obs)) <= 2.5:
            return _driver(obs, Want(pressure=True))
        return _driver(obs, Want(direction=direction_to(pos, self.target(obs)), sprint=True))

    def ok(self, obs):
        c = carrier(obs)
        return metres(self.pos(obs), c) <= 3.0 and metres(self.partner(obs), c) <= 3.0

    def measure(self, obs):
        if not self.active(obs):
            return None
        c = carrier(obs)
        return max(metres(self.pos(obs), c), metres(self.partner(obs), c))

    def refs(self, obs):
        return [self.partner(obs)] + super().refs(obs)


class ShadowPress(Press):
    """Press the carrier along the line to one opponent, so the press also
    blocks the pass to him (a cover shadow)."""

    label = "PRESS with a cover shadow on their midfielder"
    unit = "m off the lane"

    def man(self, obs):
        return obs["right_team"][self.p["opp"]]

    def active(self, obs):
        return super().active(obs) and carrier_index(obs) != self.p["opp"]

    def target(self, obs):
        return toward(carrier(obs), self.man(obs), 2.0)

    def act(self, obs):
        pos = self.pos(obs)
        return _driver(obs, Want(direction=direction_to(pos, self.target(obs)),
                                 sprint=metres(pos, carrier(obs)) > 3.0))

    def ok(self, obs):
        pos = self.pos(obs)
        lane = metres(pos, lane_point(pos, carrier(obs), self.man(obs)))
        return metres(pos, carrier(obs)) <= 3.0 and lane <= 2.0

    def measure(self, obs):
        if not self.active(obs):
            return None
        return metres(self.pos(obs), lane_point(self.pos(obs), carrier(obs), self.man(obs)))

    def refs(self, obs):
        return [self.man(obs)] + super().refs(obs)


# --- attacking duties (we have the ball) ----------------------------------

class _Attacking(Duty):
    """Applies while we have the ball and someone else is on it: our player on
    the ball is handed to the game's AI (training plan §4.1, rank 1)."""

    def active(self, obs):
        return obs["ball_owned_team"] == 0 and our_carrier_index(obs) != self.me

    def serve_ready(self, obs) -> bool:
        """The moment a willing passer would give him the ball (run.py's
        scripted passer); judged on position, so it applies to whoever is in
        the slot, scripted or the game's AI."""
        return self.active(obs) and self.ok(obs)

    def refs(self, obs):
        c = our_carrier(obs)
        return [] if c is None else [c]


class HoldShape(_Attacking):
    """Stay within radius of an anchor that shifts 30 % toward the ball (max 10 m)."""

    label = "HOLD SHAPE in his zone (8 m)"
    unit = "m from zone centre"

    centre = HoldZone.centre
    target = HoldZone.target
    act = HoldZone.act
    ok = HoldZone.ok

    def measure(self, obs):
        return metres(self.pos(obs), self.centre(obs)) if self.active(obs) else None

    def refs(self, obs):
        return [self.centre(obs)]


class Support(_Attacking):
    """Be a pass option: 8-15 m from our player on the ball, with no opponent
    within 1.5 m of the passing line, onside."""

    label = "SUPPORT the ball carrier (open, 8-15 m)"
    unit = "m clear around the passing line"
    RING_M = 11.0

    def __init__(self, me, **params):
        super().__init__(me, **params)
        self.spot = None

    def candidates(self, obs):
        c = our_carrier(obs)
        line = offside_line(obs)
        for k in range(16):
            a = k * math.pi / 8
            spot = c[:2] + np.array([math.cos(a), math.sin(a)]) * self.RING_M / _METRES
            if abs(spot[1]) <= 0.38 and -0.95 <= spot[0] <= min(line, 0.95):
                yield spot

    def score(self, obs, spot):
        forward = (spot[0] - our_carrier(obs)[0]) * _METRES[0]
        return (min(lane_clearance(obs, our_carrier(obs), spot), 6.0) + 0.1 * forward
                - 0.15 * metres(self.pos(obs), spot))

    def target(self, obs):
        best = max(self.candidates(obs), key=lambda q: self.score(obs, q), default=None)
        # Keep the spot unless another is clearly better: no dithering.
        if self.spot is None or best is not None and (
                self.score(obs, best) > self.score(obs, self.spot) + 1.0):
            self.spot = best if best is not None else self.pos(obs)[:2]
        return self.spot

    def act(self, obs):
        return go_to(obs, self.pos(obs), self.target(obs), settle_m=0.8)

    def ok(self, obs):
        c = our_carrier(obs)
        d = metres(self.pos(obs), c)
        return 8.0 <= d <= 15.0 and lane_clearance(obs, c, self.pos(obs)) >= 1.5

    def measure(self, obs):
        if not self.active(obs):
            return None
        return lane_clearance(obs, our_carrier(obs), self.pos(obs))


class Width(_Attacking):
    """Within 5 m of one touchline, level with the ball, onside."""

    label = "GIVE WIDTH on the right (within 5 m of the touchline)"
    unit = "m from the touchline"
    EDGE = 0.42

    def edge(self):
        return self.EDGE if self.p["side"] == "right" else -self.EDGE

    def target(self, obs):
        x = min(obs["ball"][0] + 5.0 / _METRES[0], offside_line(obs) - 1.0 / _METRES[0])
        return np.array([x, self.edge() - math.copysign(3.0 / _METRES[1], self.edge())])

    def act(self, obs):
        return go_to(obs, self.pos(obs), self.target(obs), settle_m=1.0)

    def from_line(self, obs):
        return abs(self.edge() - self.pos(obs)[1]) * _METRES[1]

    def ok(self, obs):
        return self.from_line(obs) <= 5.0

    def measure(self, obs):
        return self.from_line(obs) if self.active(obs) else None


class RunInBehind(_Attacking):
    """Wait on their last line, onside; when our man on the ball has time,
    run beyond it in your channel."""

    label = "RUN IN BEHIND their back line"
    unit = "m beyond their line (+) / short of it (-)"
    TIME_M = 5.0  # "has time": no opponent this close to our carrier
    RUN_S = 3.0

    def __init__(self, me, **params):
        super().__init__(me, **params)
        self.running = 0

    def has_time(self, obs):
        c = our_carrier(obs)
        return min(metres(p, c) for p in obs["right_team"]) > self.TIME_M

    def serve_ready(self, obs):
        # The run's trigger: on the line, onside, and the passer has time.
        pos, line = self.pos(obs), offside_line(obs)
        return (self.active(obs) and pos[0] < line and (line - pos[0]) * _METRES[0] <= 2.0
                and self.has_time(obs))

    def target(self, obs):
        line = offside_line(obs)
        if self.running:
            return np.array([min(line + 15.0 / _METRES[0], 0.95), self.p["channel"]])
        return np.array([line - 1.0 / _METRES[0], self.p["channel"]])

    def act(self, obs):
        pos, line = self.pos(obs), offside_line(obs)
        if self.running:
            self.running -= 1
        elif abs(pos[0] - line) * _METRES[0] <= 2.0 and pos[0] < line and self.has_time(obs):
            self.running = int(self.RUN_S * 25)
        return go_to(obs, pos, self.target(obs), settle_m=0.6,
                     sprint_m=0.0 if self.running else 4.0)

    def ok(self, obs):
        pos, line = self.pos(obs), offside_line(obs)
        ready = pos[0] < line and (line - pos[0]) * _METRES[0] <= 3.0
        beyond = pos[0] > line and velocity(obs["left_team_direction"][self.me])[0] * _METRES[0] > 3.0
        return bool(ready or beyond)

    def measure(self, obs):
        if not self.active(obs):
            return None
        return (self.pos(obs)[0] - offside_line(obs)) * _METRES[0]


class DropToReceive(_Attacking):
    """Between their midfield and defensive lines, within 15 m of the ball."""

    label = "DROP TO RECEIVE between their lines"
    unit = "m from the gap between their lines"

    def target(self, obs):
        back, mid = their_lines(obs)
        c = our_carrier(obs)
        x = (back + mid) / 2
        y = 0.5 * c[1] + 0.5 * self.pos(obs)[1]
        spot = np.array([x, y])
        # No further than 12 m from the ball, or he is not an option.
        if metres(spot, c) > 12.0:
            spot = toward(c, spot, 12.0)
        return spot

    def act(self, obs):
        return go_to(obs, self.pos(obs), self.target(obs), settle_m=0.8)

    def ok(self, obs):
        back, mid = their_lines(obs)
        x = self.pos(obs)[0]
        return mid < x < back and metres(self.pos(obs), our_carrier(obs)) <= 15.0

    def measure(self, obs):
        if not self.active(obs):
            return None
        back, mid = their_lines(obs)
        x = self.pos(obs)[0]
        return 0.0 if mid < x < back else min(abs(x - mid), abs(x - back)) * _METRES[0]


class RestDefence(_Attacking):
    """Stay behind the ball inside a zone while we attack, against the counter."""

    label = "REST DEFENCE: left-back stays back"
    unit = "m from zone centre"
    RADIUS_M = 8.0

    def target(self, obs):
        zx, zy = self.p["zone"]
        return np.array([min(zx, obs["ball"][0] - 5.0 / _METRES[0]), zy])

    def act(self, obs):
        return go_to(obs, self.pos(obs), self.target(obs), settle_m=1.0)

    def ok(self, obs):
        return (metres(self.pos(obs), self.p["zone"]) <= self.RADIUS_M
                and self.pos(obs)[0] < obs["ball"][0])

    def measure(self, obs):
        return metres(self.pos(obs), self.p["zone"]) if self.active(obs) else None

    def refs(self, obs):
        return [np.asarray(self.p["zone"])]


class AttackZone(_Attacking):
    """Be in a zone of the box (default the penalty spot) for the cross."""

    label = "ARRIVE in the box for the cross (penalty spot)"
    unit = "m from the spot"
    REACH_M = 3.0

    def target(self, obs):
        return np.asarray(self.p["zone"])

    def act(self, obs):
        return go_to(obs, self.pos(obs), self.target(obs), settle_m=0.8)

    def ok(self, obs):
        return metres(self.pos(obs), self.target(obs)) <= self.REACH_M

    def measure(self, obs):
        return metres(self.pos(obs), self.target(obs)) if self.active(obs) else None

    def refs(self, obs):
        return [self.target(obs)]


class Overlap(_Attacking):
    """While a teammate (the winger) has the ball: run past him on the
    outside, beyond and wider, onside."""

    label = "OVERLAP the winger (right-back)"
    unit = "m from the overlap spot"
    AHEAD_M, OUTSIDE_M = 8.0, 5.0

    def mate(self, obs):
        return obs["left_team"][self.p["mate"]]

    def active(self, obs):
        return obs["ball_owned_team"] == 0 and our_carrier_index(obs) == self.p["mate"]

    def target(self, obs):
        m = self.mate(obs)
        side = 1.0 if m[1] >= 0 else -1.0
        x = min(m[0] + self.AHEAD_M / _METRES[0], offside_line(obs) - 1.0 / _METRES[0])
        y = side * min(abs(m[1]) + self.OUTSIDE_M / _METRES[1], 0.42 - 2.0 / _METRES[1])
        return np.array([x, y])

    def act(self, obs):
        return go_to(obs, self.pos(obs), self.target(obs), settle_m=0.8, sprint_m=2.0)

    def ok(self, obs):
        me, m = self.pos(obs), self.mate(obs)
        return ((me[0] - m[0]) * _METRES[0] >= 2.0 and (abs(me[1]) - abs(m[1])) * _METRES[1] >= 2.0
                and metres(me, m) <= 15.0)

    def serve_ready(self, obs):
        return self.active(obs) and self.ok(obs) and self.pos(obs)[0] < offside_line(obs)

    def measure(self, obs):
        return metres(self.pos(obs), self.target(obs)) if self.active(obs) else None

    def refs(self, obs):
        return [self.mate(obs)]


class PinFullBack(_Attacking):
    """Stay high and wide on their full-back, so he can't leave his line."""

    label = "PIN their left-back (high and wide on him)"
    unit = "m: their left-back from his goal line"

    def man(self, obs):
        return obs["right_team"][self.p["opp"]]

    def target(self, obs):
        # Level with him and just outside him: he can't step in without
        # leaving our winger free down the line.
        m = self.man(obs)
        side = 1.0 if m[1] >= 0 else -1.0
        x = min(m[0], offside_line(obs) - 1.0 / _METRES[0])
        y = side * min(abs(m[1]) + 4.0 / _METRES[1], 0.42 - 2.0 / _METRES[1])
        return np.array([x, y])

    def act(self, obs):
        return go_to(obs, self.pos(obs), self.target(obs), settle_m=1.0)

    def ok(self, obs):
        # On him (within 8 m), level or higher, and outside him. Not "on the
        # touchline": their full-back tucks in, and the pin goes with him.
        me, m = self.pos(obs), self.man(obs)
        return (metres(me, m) <= 8.0 and (m[0] - me[0]) * _METRES[0] <= 3.0
                and (abs(me[1]) - abs(m[1])) * _METRES[1] >= -1.0)

    def measure(self, obs):
        # Pinned = their full-back kept deep: his distance from his own goal line.
        return (1.0 - self.man(obs)[0]) * _METRES[0] if self.active(obs) else None

    def refs(self, obs):
        return [self.man(obs)]


class InvertedFullBack(HoldShape):
    """Full-back steps into central midfield while we have the ball."""

    label = "INVERTED left-back (into central midfield)"
    unit = "m from his midfield spot"

    def centre(self, obs):
        x = min(obs["ball"][0] - 15.0 / _METRES[0], 0.2)
        return np.array([x, self.p["side"] * 0.08])


class ThirdManRun(RunInBehind):
    """The third man: while a teammate who came short has the ball, make the
    run in behind (wait on their line onside, go when he has time) for his
    through ball. Otherwise wait a few metres behind and to his side."""

    label = "THIRD-MAN RUN in behind off the striker who came short"
    unit = "m beyond their line (+) / short of it (-)"
    TIME_M = 3.0  # the striker who came short is always marked: a quick ball

    def mate(self, obs):
        return obs["left_team"][self.p["mate"]]

    def mate_has_it(self, obs):
        return our_carrier_index(obs) == self.p["mate"]

    def has_time(self, obs):
        return self.mate_has_it(obs) and super().has_time(obs)

    def target(self, obs):
        if self.mate_has_it(obs) or self.running:
            return super().target(obs)
        return np.array([self.mate(obs)[0] - 3.0 / _METRES[0], self.p["channel"]])

    def ok(self, obs):
        if self.mate_has_it(obs) or self.running:
            return super().ok(obs)
        return metres(self.pos(obs), self.target(obs)) <= 5.0

    def refs(self, obs):
        return [self.mate(obs)]


DUTIES = {
    "press": Press, "contain": Contain, "mark": Mark, "cover": Cover,
    "screen_lane": ScreenLane, "hold_line": HoldLine, "hold_zone": HoldZone,
    "sweep": Sweep, "double_team": DoubleTeam, "shadow_press": ShadowPress,
    "hold_shape": HoldShape, "support": Support, "width": Width,
    "run_in_behind": RunInBehind, "drop_to_receive": DropToReceive,
    "rest_defence": RestDefence, "attack_zone": AttackZone, "overlap": Overlap,
    "pin_fullback": PinFullBack, "inverted_fullback": InvertedFullBack,
    "third_man_run": ThirdManRun,
}
