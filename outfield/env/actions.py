"""Five action heads in, one engine command per player out.

Heads (docs/training_plan.md §2.2):
    move    17  idle + 16 directions, 22.5° apart, counter-clockwise from "toward
                their goal" in the team frame (+y up the screen). Sent as the
                engine's analog direction (verified: within ~2° of the target,
                and it survives a builtin_ai hand-off).
    speed    3  walk, sprint, close control (sticky sprint / dribble, toggled on change)
    ball     8  none, short pass, high pass, long pass, shot, curl shot, chip, through ball
    defend   5  none, pressure, team pressure, sliding, keeper rush
    power    1  kick power in [0, 1], used only when a kick is pressed

The engine takes one button per player per step. Priority: a kick, then
sliding, then one sticky toggle (sprint, dribble, pressure, team pressure,
keeper rush, in that order), else idle. Movement goes alongside every one of
them as the analog direction, so it is never delayed.

Rules from the duty lab (docs/CONTEXT.md §7):
- Kick wind-up: after a kick is pressed the player stays on our control,
  holding the kick's direction, until the ball has left his foot (or 1 s),
  whatever the policy says; a hand-off to the game's AI then would cancel
  the kick silently.
- AI plays the ball (`ai_on_ball`): our player on the ball, not winding up a
  kick, is handed to the game's AI (held buttons released first, one per step).
- Keeper rush tells the game's *AI* keeper to come out: outfield players only.
- The keeper (slot 0) may only walk or sprint, pass (short, high, long), and
  choose no defensive action: see `masks`.

`overridden[i]` is True on every step where the policy's choice for player
i was not what reached the engine (keeper left to the AI, AI on the ball,
kick wind-up): training should leave those steps out of the policy loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gfootball.env import football_action_set

from outfield.config import STEP_S

N = 11
_V4 = [a._name for a in football_action_set.get_action_set({"action_set": "v4"})]
ACTION = {name: i for i, name in enumerate(_V4)}
STICKY = [a._name for a in football_action_set.get_sticky_actions({"action_set": "v4"})]
DIRECTIONS = STICKY[:8]

MOVE_N = 17
SPEED = ("walk", "sprint", "close_control")
BALL = ("none", "short_pass", "high_pass", "long_pass", "shot", "curl_shot", "chip", "through_ball")
DEFEND = ("none", "pressure", "team_pressure", "sliding", "keeper_rush")
HEADS = {"move": MOVE_N, "speed": len(SPEED), "ball": len(BALL), "defend": len(DEFEND)}
KICK_CLASS = {"short_pass": "pass", "high_pass": "pass", "long_pass": "pass", "through_ball": "pass",
              "shot": "shot", "curl_shot": "shot", "chip": "shot"}
TOGGLES = ("sprint", "dribble", "pressure", "team_pressure", "keeper_rush")

BALL_REACH_M = 2.0     # kicks allowed with the ball this close...
BALL_HEIGHT_M = 2.5    # ...and no higher than this (headers count)
INCOMING_S = 1.5       # ...or a loose ball whose path comes that close within this time
WINDUP_S = 1.0         # give up waiting for a kick after this

METRES = np.array([54.4, -83.6])


def move_vector(k: int) -> tuple[float, float]:
    """Head-A index -> unit analog direction (team frame, +y up = engine frame)."""
    if k <= 0:
        return 0.0, 0.0
    angle = (k - 1) * 2 * math.pi / 16
    return math.cos(angle), math.sin(angle)


def direction_index(dx: float, dy: float) -> int:
    """Inverse of move_vector for a direction in the team frame (metres, y up)."""
    if dx == 0 and dy == 0:
        return 0
    return 1 + int(round(math.atan2(dy, dx) / (2 * math.pi / 16))) % 16


# --- masks -----------------------------------------------------------------------------

def ball_reachable(raw, present: np.ndarray) -> np.ndarray:
    """[11] bool: the ball is at the player's feet (≤ 2 m, low enough), or a
    loose ball's path comes that close within INCOMING_S."""
    us = np.zeros((N, 2))
    n = len(raw["left_team"])
    us[:n] = np.asarray(raw["left_team"])[:, :2] * METRES
    ball = np.asarray(raw["ball"])
    ball_m = ball[:2] * METRES
    near = (np.linalg.norm(us - ball_m, axis=1) <= BALL_REACH_M) & (ball[2] <= BALL_HEIGHT_M)
    if raw["ball_owned_team"] == -1:
        step_m = np.asarray(raw["ball_direction"])[:2] * METRES
        for k in range(1, int(INCOMING_S / STEP_S) + 1):
            near |= np.linalg.norm(us - (ball_m + k * step_m), axis=1) <= BALL_REACH_M
    return near & present


def masks(raw, possession_team: int, present: np.ndarray, keeper_agent: bool) -> dict[str, np.ndarray]:
    """Which choices each head allows next step. Absent players: only index 0."""
    m = {name: np.zeros((N, size), bool) for name, size in HEADS.items()}
    for name in m:
        m[name][:, 0] = True
    m["move"][present] = True
    m["speed"][present] = True
    reach = ball_reachable(raw, present)
    m["ball"][reach, 1:] = True
    if possession_team == 1:  # they have it (possession memory)
        rows = present.copy()
        m["defend"][rows, 1] = True                       # pressure
        m["defend"][rows, 2] = True                       # team pressure
        m["defend"][rows, 3] = True                       # sliding
        m["defend"][rows, 4] = True                       # keeper rush (outfield only, below)
    # The keeper (slot 0) has a smaller set (agreed with G, 2026-11-24): he
    # walks or sprints, distributes with passes, and saves through
    # keeper_auto_save, not with tackles.
    m["speed"][0, 2] = False                              # no close control
    m["ball"][0, 4:8] = False                             # no shot, curl shot, chip, through ball
    m["defend"][0, 1] = False                             # no pressure
    m["defend"][0, 2] = False                             # no team pressure
    m["defend"][0, 3] = False                             # no sliding
    m["defend"][0, 4] = False                             # no keeper rush: it calls the *AI* keeper out
    if not keeper_agent:                                  # slot 0 is the game's AI: nothing to choose
        for name in m:
            m[name][0, 1:] = False
    return m


# --- the adapter -----------------------------------------------------------------------------

@dataclass
class Kick:
    button: str
    power: float
    direction: tuple[float, float]
    steps: int = 0
    had_ball: bool = False


class Adapter:
    def __init__(self, ai_on_ball: bool = True, keeper_agent: bool = False):
        self.ai_on_ball, self.keeper_agent = ai_on_ball, keeper_agent
        self.kicks: dict[int, Kick] = {}

    def reset(self):
        self.kicks = {}

    def commands(self, raw_list, actions: dict[str, np.ndarray], mask: dict[str, np.ndarray],
                 present: np.ndarray, step: int):
        """Engine actions for our n players, plus per-player bookkeeping:
        overridden [11], the button class pressed ([11] of None / 'pass' /
        'shot' / 'defend') and kick events."""
        n = len(raw_list)
        raw = raw_list[0]
        owner = (int(raw["ball_owned_team"]), int(raw["ball_owned_player"]))
        out: list = [ACTION["idle"]] * n
        overridden = np.zeros(N, bool)
        pressed: list[str | None] = [None] * N
        events: list[dict] = []
        for i in range(n):
            if not present[i]:
                continue
            held = {name for name, on in zip(STICKY, raw_list[i]["sticky_actions"]) if on}
            if i == 0 and not self.keeper_agent:
                out[i], overridden[i] = _hand_to_ai(held), True
                continue
            kick = self.kicks.get(i)
            if kick is not None:
                kick.steps += 1
                on_ball = owner == (0, i)
                kick.had_ball |= on_ball
                left_foot = kick.had_ball and not on_ball
                if not left_foot and kick.steps * STEP_S < WINDUP_S and owner[0] != 1:
                    out[i], overridden[i] = (ACTION["idle"], -1.0, *kick.direction), True
                    continue
                del self.kicks[i]
                events.append(dict(slot=i, button=kick.button, power=kick.power, step=step,
                                   event="left_foot" if left_foot else "expired"))
            if self.ai_on_ball and owner == (0, i):
                out[i], overridden[i] = _hand_to_ai(held), True
                continue
            out[i], pressed[i] = self._policy(i, held, actions, mask, owner, step, events)
        return out, overridden, pressed, events

    def _policy(self, i, held, actions, mask, owner, step, events):
        def choice(head):
            k = int(actions[head][i])
            return k if 0 <= k < mask[head].shape[1] and mask[head][i, k] else 0
        dx, dy = move_vector(choice("move"))
        speed, ball, defend = choice("speed"), choice("ball"), choice("defend")
        power = float(np.clip(actions["power"][i], 0.0, 1.0))
        if ball:
            button = BALL[ball]
            self.kicks[i] = Kick(button, power, (dx, dy), had_ball=owner == (0, i))
            events.append(dict(slot=i, button=button, power=power, step=step, event="pressed"))
            return (ACTION[button], power, dx, dy), KICK_CLASS[button]
        if DEFEND[defend] == "sliding":
            return (ACTION["sliding"], -1.0, dx, dy), "defend"
        want = {"sprint": speed == 1, "dribble": speed == 2, "pressure": defend == 1,
                "team_pressure": defend == 2, "keeper_rush": defend == 4}
        for name in TOGGLES:
            if want[name] != (name in held):
                cls = "defend" if want[name] and name in ("pressure", "team_pressure", "keeper_rush") else None
                return (ACTION[name if want[name] else f"release_{name}"], -1.0, dx, dy), cls
        return (ACTION["idle"], -1.0, dx, dy), None


def _hand_to_ai(held) -> int:
    """Release held buttons first (one per step), else builtin_ai: buttons
    still held would leak into the AI's play."""
    for name in TOGGLES:
        if name in held:
            return ACTION[f"release_{name}"]
    if any(d in held for d in DIRECTIONS):
        return ACTION["release_direction"]
    return ACTION["builtin_ai"]
