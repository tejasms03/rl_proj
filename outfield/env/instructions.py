"""The 91-number instruction block: PROVISIONAL layout, until M1 and G agree.

docs/training_plan.md §2.1 fixes the sections and sizes; the order inside
each section and the scaling are this module's proposal (written up in
docs/env_wrapper.md). Only this file changes when the layout is frozen.

    I1 role (14)        role family one-hot (9), side one-hot (3), slot / 10, taker flag
    I2 duties (21)      defending D0-D8 (9), attacking A0-A7 (8), transition T0-T3 (4)
    I3 targets (29)     opponent slot one-hot (11 + none), teammate slot one-hot (11 + none),
                        anchor x / 55, anchor y / 36 (metres, team frame, y up),
                        radius / 10 m, line height (0-1), press intensity (0-1)
    I4 dials (6)        line_height, press_trigger, width, tempo, risk, compactness
    I5 set play (15)    restart type (6), set-play duty (5), zone x / 55, zone y / 36,
                        arrival time / 10 s, time since restart / 10 s
    M memory (6)        written by the wrapper, never by the manager:
                        last 3 button actions as class / 3 (none 0, pass 1/3, shot 2/3,
                        defend 1), steps since my last touch / 100 (capped),
                        steps since possession changed / 100 (capped),
                        instruction changed in the last 5 s (0/1)

The manager (or a preset, or the sampler) writes the first 85 numbers
(I1-I5); the wrapper always writes M.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from outfield.config import STEP_S

N = 11
DEFENDING = ("none", "hold_zone", "press", "contain", "cover", "mark", "screen_lane",
             "hold_line", "sweep")                                   # D0-D8
ATTACKING = ("none", "hold_shape", "support", "width", "run_in_behind", "drop_to_receive",
             "attack_zone", "rest_defence")                          # A0-A7
TRANSITION = ("none", "counter_press", "recover", "break")           # T0-T3
V1_DEFENDING = (1, 2, 4, 5, 7)
V1_ATTACKING = (1, 2, 3, 4)
V1_TRANSITION = (0, 2)
KEEPER_DEFENDING = (0, 8)
KEEPER_ATTACKING = (0, 1)
ROLE_FAMILIES = ("GK", "CB", "FB", "DM", "CM", "AM", "W", "ST", "none")
SIDES = ("left", "centre", "right")
DIALS = ("line_height", "press_trigger", "width", "tempo", "risk", "compactness")
V1_DIALS = ("line_height", "press_trigger", "width")
RESTARTS = ("kickoff", "goal_kick", "free_kick_attack", "free_kick_defend",
            "corner_attack", "corner_defend")
SET_DUTIES = ("deliver", "attack_zone", "rest_defence", "mark", "hold_zone")
ACTION_CLASSES = ("none", "pass", "shot", "defend")

# GRF role (e_PlayerRole: GK CB LB RB DM CM LM RM AM CF) -> family, side.
ROLE_TO_FAMILY = (0, 1, 2, 2, 3, 4, 6, 6, 5, 7)
ROLE_TO_SIDE = (1, 1, 0, 2, 1, 1, 0, 2, 1, 1)


def _layout(blocks):
    out, start = {}, 0
    for name, size in blocks:
        out[name] = slice(start, start + size)
        start += size
    return out, start


LAYOUT, SIZE = _layout([
    ("role_family", 9), ("side", 3), ("slot", 1), ("taker", 1),                     # I1
    ("defending", 9), ("attacking", 8), ("transition", 4),                          # I2
    ("opp_target", 12), ("mate_target", 12), ("anchor", 2), ("radius", 1),
    ("line_height", 1), ("press_intensity", 1),                                     # I3
    ("dials", 6),                                                                   # I4
    ("restart", 6), ("set_duty", 5), ("set_zone", 2), ("arrival", 1), ("since_restart", 1),  # I5
    ("memory", 6),                                                                  # M
])
MANAGER_SIZE = LAYOUT["memory"].start  # 85: what the manager / sampler writes
assert SIZE == 91 and MANAGER_SIZE == 85


@dataclass
class Instruction:
    """One player's instructions (I1-I3). Positions in metres, team frame."""
    role: int = -1                  # GRF role, for I1
    slot: int = 0
    taker: bool = False
    defending: int = 0
    attacking: int = 0
    transition: int = 0
    opp: int | None = None          # opponent slot the duty is about (mark, ...)
    mate: int | None = None         # teammate slot (cover, ...)
    anchor: tuple[float, float] | None = None
    radius_m: float = 0.0
    line_height: float = 0.0
    press_intensity: float = 0.0


@dataclass
class Plan:
    """A whole team's instructions plus the dials (I4). Set plays (I5) stay
    zero in v1."""
    players: list[Instruction]
    dials: dict[str, float] = field(default_factory=dict)
    name: str = "random"


def encode(plan: Plan, present: np.ndarray) -> np.ndarray:
    """[11, 85]: the manager part of every player's block."""
    out = np.zeros((N, MANAGER_SIZE), np.float32)
    dials = np.array([plan.dials.get(d, 0.0) for d in DIALS], np.float32)
    for i, ins in enumerate(plan.players):
        if not present[i]:
            continue
        row = out[i]
        if 0 <= ins.role < len(ROLE_TO_FAMILY):
            row[LAYOUT["role_family"].start + ROLE_TO_FAMILY[ins.role]] = 1.0
            row[LAYOUT["side"].start + ROLE_TO_SIDE[ins.role]] = 1.0
        else:
            row[LAYOUT["role_family"].start + ROLE_FAMILIES.index("none")] = 1.0
            row[LAYOUT["side"].start + 1] = 1.0
        row[LAYOUT["slot"]] = ins.slot / 10.0
        row[LAYOUT["taker"]] = float(ins.taker)
        row[LAYOUT["defending"].start + ins.defending] = 1.0
        row[LAYOUT["attacking"].start + ins.attacking] = 1.0
        row[LAYOUT["transition"].start + ins.transition] = 1.0
        row[LAYOUT["opp_target"].start + (N if ins.opp is None else ins.opp)] = 1.0
        row[LAYOUT["mate_target"].start + (N if ins.mate is None else ins.mate)] = 1.0
        if ins.anchor is not None:
            row[LAYOUT["anchor"]] = (ins.anchor[0] / 55.0, ins.anchor[1] / 36.0)
        row[LAYOUT["radius"]] = ins.radius_m / 10.0
        row[LAYOUT["line_height"]] = ins.line_height
        row[LAYOUT["press_intensity"]] = ins.press_intensity
        row[LAYOUT["dials"]] = dials
    return out


def decode(row: np.ndarray) -> Instruction:
    """One player's manager part (85 or 91 numbers) back to an Instruction,
    for duty checks (which must also work on blocks passed in from outside)."""
    def pick(name):
        part = row[LAYOUT[name]]
        return int(np.argmax(part)) if part.max() > 0 else 0
    opp, mate = pick("opp_target"), pick("mate_target")
    anchor = row[LAYOUT["anchor"]]
    return Instruction(
        slot=int(round(float(row[LAYOUT["slot"]][0]) * 10)), taker=bool(row[LAYOUT["taker"]][0] > 0.5),
        defending=pick("defending"), attacking=pick("attacking"), transition=pick("transition"),
        opp=None if opp == N else opp, mate=None if mate == N else mate,
        anchor=(float(anchor[0]) * 55.0, float(anchor[1]) * 36.0),
        radius_m=float(row[LAYOUT["radius"]][0]) * 10.0,
        line_height=float(row[LAYOUT["line_height"]][0]),
        press_intensity=float(row[LAYOUT["press_intensity"]][0]))


# --- the memory block (wrapper-written) ------------------------------------------------

class Memory:
    """The 6 numbers of block M, per player."""

    CHANGED_S = 5.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.actions = np.zeros((N, 3), np.float32)      # last 3 button actions, class / 3
        self.since_touch = np.full(N, 10_000, np.int64)
        self.since_change = np.full(N, 10_000, np.int64)
        self.previous: np.ndarray | None = None

    def update(self, pressed: list[str | None], touching: int | None, possession_changed_steps: int,
               manager_part: np.ndarray) -> np.ndarray:
        """pressed[i]: the class of the button player i pressed this step, or
        None. touching: our player on the ball (raw), if any. Returns [11, 6]."""
        for i, cls in enumerate(pressed):
            if cls is not None and cls != "none":
                self.actions[i] = np.roll(self.actions[i], 1)
                self.actions[i, 0] = ACTION_CLASSES.index(cls) / 3.0
        self.since_touch += 1
        if touching is not None:
            self.since_touch[touching] = 0
        if self.previous is not None:
            changed = np.any(np.abs(manager_part - self.previous) > 1e-6, axis=1)
            self.since_change[changed] = 0
        self.previous = manager_part.copy()
        self.since_change += 1
        out = np.zeros((N, 6), np.float32)
        out[:, :3] = self.actions
        out[:, 3] = np.minimum(self.since_touch, 100) / 100.0
        out[:, 4] = min(possession_changed_steps, 100) / 100.0
        out[:, 5] = (self.since_change * STEP_S <= self.CHANGED_S).astype(np.float32)
        return out


# --- the sampler ---------------------------------------------------------------------------

PRESETS = ("high_press", "low_block", "possession", "counter", "wing_play", "false_nine")


class Sampler:
    """Random v1 instructions for training (training_plan §1, outfield.md O1.2).

    The whole team is re-drawn every 30-60 s of game time. With probability
    `preset_p` the draw is one of the six presets instead (built from v1
    duties only: where a preset needs a duty that isn't in v1, the player
    keeps his default or gets the closest v1 duty, see `preset`). Anchors come from the formation, i.e. where each
    player stood at kick-off. The combination grammar isn't written down yet
    (docs/env_wrapper.md, open items): draws are independent per player.
    """

    def __init__(self, seed: int = 0, redraw_s=(30.0, 60.0), preset_p: float = 0.2):
        self.rng = np.random.default_rng(seed)
        self.redraw_s, self.preset_p = redraw_s, preset_p
        self.next_draw = 0
        self.plan: Plan | None = None

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.next_draw, self.plan = 0, None

    def __call__(self, step: int, roles, formation, us_pos, them_pos, present, them_present) -> Plan:
        if self.plan is None or step >= self.next_draw:
            if self.rng.random() < self.preset_p:
                name = PRESETS[self.rng.integers(len(PRESETS))]
                self.plan = preset(name, roles, formation, us_pos, them_pos, present, them_present)
            else:
                self.plan = self._random(roles, formation, us_pos, them_pos, present, them_present)
            lo, hi = self.redraw_s
            self.next_draw = step + int(self.rng.uniform(lo, hi) / STEP_S)
        return self.plan

    def _random(self, roles, formation, us_pos, them_pos, present, them_present) -> Plan:
        rng = self.rng
        players = []
        opps = [j for j in range(N) if them_present[j]]
        for i in range(N):
            ins = Instruction(role=int(roles[i]) if present[i] else -1, slot=i)
            if not present[i]:
                players.append(ins)
                continue
            keeper = int(roles[i]) == 0
            ins.defending = int(rng.choice(KEEPER_DEFENDING if keeper else V1_DEFENDING))
            ins.attacking = int(rng.choice(KEEPER_ATTACKING if keeper else V1_ATTACKING))
            ins.transition = 0 if keeper else int(rng.choice(V1_TRANSITION))
            _targets(ins, i, rng, formation, us_pos, them_pos, present, opps)
            players.append(ins)
        dials = {d: float(rng.uniform()) for d in V1_DIALS}
        return Plan(players, dials, "random")


def _targets(ins: Instruction, i, rng, formation, us_pos, them_pos, present, opps):
    """Fill the targets a duty needs, so the instruction is carry-out-able."""
    home = formation[i]
    if ins.defending in (1, 7) or ins.attacking in (1,):  # zones, line, shape
        ins.anchor = (float(home[0] + rng.normal(0, 3)), float(home[1] + rng.normal(0, 3)))
        ins.radius_m = float(rng.uniform(5, 10))
    if ins.defending == 7:
        ins.line_height = float(rng.uniform(0.1, 0.9))
    if ins.defending == 2:
        ins.press_intensity = float(rng.uniform(0.3, 1.0))
    if ins.defending == 5 and opps:  # mark: an opponent, nearer ones likelier
        d = np.array([np.linalg.norm(them_pos[j] - us_pos[i]) for j in opps])
        w = np.exp(-d / 15.0)
        ins.opp = int(opps[rng.choice(len(opps), p=w / w.sum())])
    if ins.defending == 4:  # cover: a teammate (not the keeper)
        mates = [j for j in range(1, N) if present[j] and j != i]
        ins.mate = int(rng.choice(mates)) if mates else None
    if ins.attacking == 3:  # width: the side goes in the anchor's y (proposal: I3 has no side field)
        side = 1.0 if home[1] >= 0 else -1.0
        ins.anchor = (float(home[0]), side * 30.0)


def preset(name: str, roles, formation, us_pos, them_pos, present, them_present) -> Plan:
    """The six tactics of training_plan §1 as v1 duty assignments."""
    fam = [ROLE_TO_FAMILY[int(r)] if present[i] else -1 for i, r in enumerate(roles)]
    F = {f: k for k, f in enumerate(ROLE_FAMILIES)}
    opps = [j for j in range(N) if them_present[j]]

    def nearest_opp(i):
        if not opps:
            return None
        return int(min(opps, key=lambda j: np.linalg.norm(them_pos[j] - us_pos[i])))

    strikers = [i for i in range(N) if fam[i] == F["ST"]]
    players = []
    for i in range(N):
        ins = Instruction(role=int(roles[i]) if present[i] else -1, slot=i)
        if not present[i]:
            players.append(ins)
            continue
        home = formation[i]
        ins.defending, ins.attacking = 1, 1  # default: hold zone / hold shape
        ins.anchor, ins.radius_m = (float(home[0]), float(home[1])), 8.0
        f = fam[i]
        if f == F["GK"]:
            ins.defending, ins.attacking, ins.anchor = 0, 0, None
        elif name == "high_press":
            if f == F["ST"]:
                ins.defending, ins.press_intensity = 2, 0.9
            elif f in (F["CM"], F["AM"]) and strikers:
                ins.defending, ins.mate = 4, strikers[0]
            elif f in (F["W"], F["DM"]):
                ins.defending, ins.opp = 5, nearest_opp(i)
            elif f in (F["CB"], F["FB"]):
                ins.defending, ins.line_height = 7, 0.8
        elif name == "low_block":
            if f in (F["CB"], F["FB"]):
                ins.defending, ins.line_height = 7, 0.15
        elif name == "possession":
            if f == F["CM"]:
                ins.attacking = 2
            elif f == F["FB"]:
                ins.attacking, ins.anchor = 3, (float(home[0]), 30.0 if home[1] >= 0 else -30.0)
        elif name == "counter":
            ins.transition = 2
            if f in (F["ST"], F["W"]):
                ins.attacking = 4
        elif name == "wing_play":
            if f == F["FB"]:
                ins.attacking, ins.anchor = 3, (float(home[0]), 30.0 if home[1] >= 0 else -30.0)
            elif f == F["W"]:
                ins.attacking = 4
        elif name == "false_nine":
            if f == F["ST"]:
                ins.attacking = 2  # A5 drop to receive is not in v1: support is the closest v1 duty
            elif f == F["W"]:
                ins.attacking = 4
        players.append(ins)
    dials = {"high_press": dict(line_height=0.8, press_trigger=0.9, width=0.5),
             "low_block": dict(line_height=0.15, press_trigger=0.2, width=0.3),
             "possession": dict(line_height=0.6, press_trigger=0.5, width=0.7),
             "counter": dict(line_height=0.3, press_trigger=0.4, width=0.5),
             "wing_play": dict(line_height=0.5, press_trigger=0.5, width=0.9),
             "false_nine": dict(line_height=0.5, press_trigger=0.5, width=0.6)}[name]
    return Plan(players, dials, name)
