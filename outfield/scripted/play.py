"""A scripted 3-player attacking play: "triangle attack".

Our three outfield players keep a triangle across three lanes (left, centre,
right). The man on the ball drives at goal, passes to an open teammate when
he is closed down or a teammate is clearly better placed, and shoots in
range. The two without the ball take the free lanes ahead of him. Loose
balls and their possession go to the game's AI, and so does our keeper.

Everything here works on the left team's view (absolute coordinates, we
attack +x). Per step, `TrianglePlay.actions(obs)` takes the env's list of
per-player observations and returns one engine action per player:
an int, or (action, kick_power) when a kick is pressed.

The traps from docs/CONTEXT.md §7 handled here:
- Possession blinks: `Possession` (outfield.env.possession) keeps the last owner for up to 1 s.
- Kick wind-up: a player who pressed a kick stays on our script ("idle")
  until the ball has left his foot, never handed to the game's AI.
- Sticky buttons: `drive` sends one change per step until the player's held
  state (directions, sprint, dribble) matches what we want.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from gfootball.env import football_action_set

from outfield.config import METRES_X, METRES_Y, STEP_S
from outfield.env.possession import Possession  # the wrapper's 1 s possession memory

_V4 = [a._name for a in football_action_set.get_action_set({"action_set": "v4"})]
ACTION = {name: i for i, name in enumerate(_V4)}
STICKY = [a._name for a in football_action_set.get_sticky_actions({"action_set": "v4"})]
DIRECTIONS = ["right", "bottom_right", "bottom", "bottom_left",
              "left", "top_left", "top", "top_right"]  # 45° steps from +x, y down

_SCALE = np.array([METRES_X, METRES_Y])
GOAL = np.array([1.0, 0.0])
KEEPER = 0

# The play's numbers, in metres and seconds.
SHOOT_RANGE_M = 22.0     # shoot from inside this distance to goal...
SHOOT_WIDTH_M = 12.0     # ...and this close to the centre line
SHOOT_PRESSED_M = 28.0   # closed down this close to goal: shoot rather than lose it
PRESSED_M = 4.0          # an opponent this close means "closed down"
SPRINT_FREE_M = 10.0     # sprint with the ball only if nobody is this close (sprint touches go 5-6 m ahead)
OPEN_LANE_M = 2.5        # a pass lane is open if no opponent is this close to it
OPEN_MAN_M = 3.0         # a teammate is open if no opponent is this close to him
PASS_RANGE_M = (6.0, 30.0)
BETTER_BY_M = 5.0        # pass unpressured if he is this much closer to goal
LOOK_AHEAD_M = 6.0       # the carrier picks the direction with most room this far out...
ROOM_ENOUGH_M = 5.0      # ...where more room than this doesn't count...
GOAL_PULL = 0.5          # ...minus this many metres of room per metre further from goal
TOUCHLINE_KEEP_M = 24.0  # never dribble wider than this from the centre line (touchline at 34 m)
LANES_Y = (-0.22, 0.0, 0.22)  # left, centre, right lanes (≈ 18 m apart)
LANE_NAMES = dict(zip(LANES_Y, ("left", "centre", "right")))
AHEAD_X = 0.15           # off-ball players sit this far ahead of the ball (≈ 8 m)
MAX_X_WIDE, MAX_X_CENTRE = 0.72, 0.82
WINDUP_S = 1.0           # give up on a kick that has not happened by then
SHORT_PASS_M = (3.0, 32.0)  # grf_tools.measure.POWER_RANGES: power 0 -> 3 m, 1 -> 35 m
SHOT_POWER = 0.9


# --- geometry ----------------------------------------------------------------

def metres(a, b) -> float:
    return float(np.linalg.norm((np.asarray(a)[:2] - np.asarray(b)[:2]) * _SCALE))


def direction_to(me, point) -> str:
    dx, dy = (np.asarray(point)[:2] - np.asarray(me)[:2]) * _SCALE
    return DIRECTIONS[round(math.atan2(dy, dx) / (math.pi / 4)) % 8]


def lane_clearance(a, b, opponents) -> float:
    """Metres from the a-b passing line to the nearest opponent."""
    a, b = np.asarray(a)[:2] * _SCALE, np.asarray(b)[:2] * _SCALE
    best = math.inf
    for p in opponents:
        p = np.asarray(p)[:2] * _SCALE
        t = np.clip(np.dot(p - a, b - a) / (np.dot(b - a, b - a) + 1e-9), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p - (a + t * (b - a)))))
    return best


def nearest_opponent_m(obs, pos) -> float:
    return min(metres(pos, p) for p in obs["right_team"])


def ball_in(obs, seconds: float) -> np.ndarray:
    """Where the ball will be, straight-line. ball_direction is the movement
    over one step (0.1 s at 10 ticks; docs/CONTEXT.md §7.6)."""
    return np.asarray(obs["ball"][:2]) + np.asarray(obs["ball_direction"][:2]) * seconds / STEP_S


def roomiest_direction(obs, me) -> str:
    """Of the forward and sideways directions, the one whose point LOOK_AHEAD_M
    out has the most room (up to ROOM_ENOUGH_M from their outfielders) and is
    closest to goal. Room beyond ROOM_ENOUGH_M doesn't count, or the carrier
    runs into empty corners; spots near the touchline are ruled out."""
    best, best_score = direction_to(me, GOAL), -math.inf
    for name in ("top", "top_right", "right", "bottom_right", "bottom"):
        k = DIRECTIONS.index(name)
        step = np.array([math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)]) * LOOK_AHEAD_M
        spot = np.asarray(me)[:2] + step / _SCALE
        if abs(spot[1]) * METRES_Y > TOUCHLINE_KEEP_M or spot[0] > 0.97:
            continue
        room = min(min(metres(spot, p) for p in obs["right_team"][1:]), ROOM_ENOUGH_M)
        score = room - GOAL_PULL * metres(spot, GOAL)
        if score > best_score:
            best, best_score = name, score
    return best


def action_name(action) -> str:
    """'short_pass (power 0.47)' for a kick, else the plain action name."""
    if isinstance(action, tuple):
        return f"{_V4[action[0]]} (power {action[1]:.2f})"
    return _V4[action]


# --- buttons -----------------------------------------------------------------

@dataclass
class Want:
    """The held state we want for a player this step."""
    direction: str | None = None  # None = stand still
    sprint: bool = False
    builtin: bool = False         # hand this step to the game's AI


def drive(own_obs, want: Want) -> int:
    """One action that moves the player's sticky state toward `want`.
    own_obs must be that player's own observation: sticky state is per player."""
    held = {name for name, on in zip(STICKY, own_obs["sticky_actions"]) if on}
    held_dir = next((d for d in DIRECTIONS if d in held), None)
    if want.builtin:
        # Hand over clean: held buttons would otherwise leak into the AI's play.
        for name in ("sprint", "dribble"):
            if name in held:
                return ACTION[f"release_{name}"]
        return ACTION["release_direction"] if held_dir else ACTION["builtin_ai"]
    if want.direction != held_dir:
        return ACTION[want.direction] if want.direction else ACTION["release_direction"]
    if want.sprint != ("sprint" in held):
        return ACTION["sprint" if want.sprint else "release_sprint"]
    return ACTION["idle"]


# --- the play ----------------------------------------------------------------------

@dataclass
class Kick:
    """A kick in progress: aim, press, then wait for the ball to leave."""
    button: str
    target: np.ndarray
    power: float
    to: int | None = None  # the teammate a pass is for
    stage: str = "aim"   # aim -> press -> windup
    steps: int = 0

    def describe(self) -> str:
        what = f"pass to #{self.to}" if self.to is not None else self.button
        return f"{what}: {self.stage} (power {self.power:.2f})"


@dataclass
class TrianglePlay:
    players: tuple[int, ...] = (1, 2, 3)
    possession: Possession = field(default_factory=Possession)
    kicks: dict[int, Kick] = field(default_factory=dict)
    log: list[tuple[int, int, str, int | None]] = field(default_factory=list)  # (step, player, kick, to)
    receiver: int | None = None  # who our last pass is meant for, while it travels
    intent: dict[int, str] = field(default_factory=dict)  # player -> what he's doing, in words
    step: int = 0

    def actions(self, obs_list) -> list:
        view = obs_list[0]  # any left-team view: absolute coordinates
        team, owner = self.possession(view)
        raw_team, raw_player = int(view["ball_owned_team"]), int(view["ball_owned_player"])
        if raw_team != -1 or team != 0:
            self.receiver = None  # someone has it: the pass has arrived (or been cut out)
        out: list = [None] * len(obs_list)
        out[KEEPER] = drive(obs_list[KEEPER], Want(builtin=True))
        self.intent = {KEEPER: "game AI (keeper)"}
        lanes = self._lanes(view, owner) if team == 0 else {}
        for i in self.players:
            own = obs_list[i]
            if i in self.kicks:
                out[i] = self._kick(i, own, raw_team, raw_player)
            elif raw_team == 0 and raw_player == i:
                out[i] = self._carry(i, own, view)
            elif team == 0 and raw_team != 0 and i == self._chaser(view):
                out[i] = drive(own, Want(builtin=True))  # a loose ball of ours: the AI collects it
                self.intent[i] = "game AI: get to the loose ball"
            elif team == 0 and i in lanes:
                out[i] = self._support(i, own, view, lanes[i])
                self.intent[i] = f"support: {LANE_NAMES[lanes[i]]} lane"
            else:
                out[i] = drive(own, Want(builtin=True))  # theirs or nobody's: the AI plays
                self.intent[i] = "game AI: defending" if team == 1 else "game AI: free ball"
        self.step += 1
        return out

    # The man on the ball.
    def _carry(self, i, own, view):
        me = view["left_team"][i]
        to_goal = metres(me, GOAL)
        pressed = nearest_opponent_m(view, me) <= PRESSED_M
        central = abs(me[1] * METRES_Y) <= SHOOT_WIDTH_M
        if central and (to_goal <= SHOOT_RANGE_M or (pressed and to_goal <= SHOOT_PRESSED_M)):
            self.intent[i] = f"shoot from {to_goal:.0f} m"
            return self._start_kick(i, own, "shot", GOAL, SHOT_POWER)
        mate = self._open_mate(i, view)
        if mate is not None:
            better = to_goal - metres(view["left_team"][mate], GOAL) >= BETTER_BY_M
            if pressed or better:
                target = view["left_team"][mate]
                lo, hi = SHORT_PASS_M
                power = float(np.clip((metres(me, target) - lo) / (hi - lo), 0.0, 1.0))
                self.receiver = mate
                self.intent[i] = f"pass to #{mate} ({'closed down' if pressed else 'he is better placed'})"
                return self._start_kick(i, own, "short_pass", target, power, to=mate)
        free = nearest_opponent_m(view, me) >= SPRINT_FREE_M
        heading = roomiest_direction(view, me)
        self.intent[i] = f"carry: dribble {heading.replace('_', ' ')}{' at a sprint' if free else ''}"
        return drive(own, Want(direction=heading, sprint=free))

    def _open_mate(self, i, view):
        me = view["left_team"][i]
        best, best_gain = None, -math.inf
        for j in self.players:
            if j == i:
                continue
            him = view["left_team"][j]
            d = metres(me, him)
            if not PASS_RANGE_M[0] <= d <= PASS_RANGE_M[1]:
                continue
            if nearest_opponent_m(view, him) < OPEN_MAN_M:
                continue
            if lane_clearance(me, him, view["right_team"]) < OPEN_LANE_M:
                continue
            gain = metres(me, GOAL) - metres(him, GOAL)
            if gain > best_gain:
                best, best_gain = j, gain
        return best

    # Kicks: aim (turn toward the target), press with power, then hold manual
    # until the ball has left his foot.
    def _start_kick(self, i, own, button, target, power, to=None):
        self.kicks[i] = Kick(button, np.asarray(target)[:2], power, to)
        self.log.append((self.step, i, button, to))
        return self._kick(i, own, 0, i)

    def _kick(self, i, own, raw_team, raw_player):
        k = self.kicks[i]
        me = own["left_team"][i]
        self.intent.setdefault(i, k.describe())
        if k.stage == "aim":
            k.stage = "press"
            want = direction_to(me, k.target)
            held = {n for n, on in zip(STICKY, own["sticky_actions"]) if on}
            if want not in held:
                return ACTION[want]
        if k.stage == "press":
            k.stage = "windup"
            return (ACTION[k.button], k.power)
        k.steps += 1
        still_on = raw_team == 0 and raw_player == i
        if still_on and k.steps * STEP_S < WINDUP_S:
            return ACTION["idle"]  # never hand over mid wind-up: it cancels the kick
        del self.kicks[i]
        return ACTION["release_direction"]

    # The two without the ball.
    def _lanes(self, view, owner_index):
        """Off-ball player -> lane y: the two lanes the carrier isn't in, each to
        the nearer free player."""
        carrier_y = view["left_team"][owner_index][1] if owner_index in self.players else view["ball"][1]
        taken = min(LANES_Y, key=lambda y: abs(y - carrier_y))
        free = [y for y in LANES_Y if y != taken]
        others = [j for j in self.players if j != owner_index]
        if len(others) != 2:
            return {}
        a, b = others
        ya, yb = (view["left_team"][a][1], view["left_team"][b][1])
        keep = abs(ya - free[0]) + abs(yb - free[1])
        swap = abs(ya - free[1]) + abs(yb - free[0])
        return {a: free[0], b: free[1]} if keep <= swap else {a: free[1], b: free[0]}

    def _support(self, i, own, view, lane_y):
        me = view["left_team"][i]
        cap = MAX_X_CENTRE if lane_y == 0.0 else MAX_X_WIDE
        spot = np.array([min(view["ball"][0] + AHEAD_X, cap), lane_y])
        d = metres(me, spot)
        if d <= 1.0:
            return drive(own, Want())
        return drive(own, Want(direction=direction_to(me, spot), sprint=d > 5.0))

    def _chaser(self, view):
        """Who goes for a loose ball of ours: the man our pass is meant for,
        else whoever is nearest where the ball will be in a second."""
        if self.receiver is not None:
            return self.receiver
        spot = ball_in(view, 1.0)
        return min(self.players, key=lambda j: metres(view["left_team"][j], spot))
