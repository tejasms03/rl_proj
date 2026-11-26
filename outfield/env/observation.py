"""The 869 numbers each player sees, and the critic's global state.

Follows docs/observation_spec.md exactly (v1, 869 with action set v4):
self 58, ball 13, teammates 10 x 47, opponents 11 x 28, context 20.

Conventions (spec, "Conventions"):
- Metres first: x_m = x * 54.4, y_m = y * -83.6, so +y is up the screen
  (toward the left touchline when attacking +x). This is also the frame of
  the engine's analog movement input, so move directions need no flip.
- Velocities: *_direction is the movement over one step (0.1 s at 10 ticks),
  so m/s = metres per step / STEP_S (docs/CONTEXT.md §7.6).
- Normalised to roughly [-1, 1]: positions / (55, 36), player speed / 10,
  ball speed / (30, 30, 10), heights / 5, distances / 55.
- Teammate rows: the other 10 slots in slot order (self skipped). Absent
  slots (drills with fewer players) are all zeros, as are absent players'
  own rows.
- Possession: the ball-owner one-hot (my team / opponents / nobody) uses the
  1 s possession memory; per-player "has the ball" uses the engine's raw
  owner, so it is true only for the player actually touching it.

Every block is addressable by name through ACTOR_LAYOUT / STATE_LAYOUT.
"""

from __future__ import annotations

import numpy as np

from outfield.config import STEP_S

N = 11
METRES = np.array([54.4, -83.6])  # raw pitch units -> metres, y up
POS_SCALE = np.array([55.0, 36.0])
PLAYER_SPEED = 10.0
BALL_SCALE = np.array([55.0, 36.0, 5.0])
BALL_SPEED = np.array([30.0, 30.0, 10.0])
DIST_SCALE = 55.0
OPP_GOAL = np.array([54.4, 0.0])
OWN_GOAL = np.array([-54.4, 0.0])
STEPS_SCALE = 3000.0
N_ROLES = 10      # GK CB LB RB DM CM LM RM AM CF (GRF's e_PlayerRole order)
N_GAME_MODES = 7  # normal, kickoff, goal kick, free kick, corner, throw-in, penalty
PITCH_CONDITIONS = ("normal", "dry", "wet", "long_grass")
N_STATS = 22      # football_env_core.PLAYER_ATTRIBUTE_NAMES[:22]
N_STICKY = 13     # v4: 8 directions, sprint, dribble, keeper_rush, pressure, team_pressure
# Opponents' key attributes (what you'd read off a player card); columns of
# PLAYER_ATTRIBUTE_NAMES: physical_velocity, physical_acceleration,
# physical_balance, technical_standingtackle, mental_defensivepositioning.
KEY_STATS = (3, 2, 0, 7, 19)


def _layout(blocks):
    out, start = {}, 0
    for name, size in blocks:
        out[name] = slice(start, start + size)
        start += size
    return out, start


SELF_BLOCKS = [("pos", 2), ("vel", 2), ("speed", 1), ("tired", 1), ("yellow", 1), ("active", 1),
               ("role", N_ROLES), ("attributes", 26), ("has_ball", 1), ("sticky", N_STICKY)]
BALL_BLOCKS = [("pos", 3), ("rel", 2), ("dist", 1), ("vel", 3), ("owner", 3), ("mine", 1)]
MATE_BLOCKS = [("rel", 2), ("dist", 1), ("pos", 2), ("vel", 2), ("role", N_ROLES),
               ("active", 1), ("yellow", 1), ("tired", 1), ("has_ball", 1), ("attributes", 26)]
OPP_BLOCKS = [("rel", 2), ("dist", 1), ("pos", 2), ("vel", 2), ("role", N_ROLES),
              ("active", 1), ("yellow", 1), ("tired", 1), ("has_ball", 1), ("ball_dist", 1),
              ("key_attributes", 6)]
CONTEXT_BLOCKS = [("opp_goal_rel", 2), ("opp_goal_dist", 1), ("opp_goal_angle", 2),
                  ("own_goal_rel", 2), ("score_diff", 1), ("time_left", 1),
                  ("game_mode", N_GAME_MODES), ("pitch", len(PITCH_CONDITIONS))]

SELF, SELF_SIZE = _layout(SELF_BLOCKS)
BALL, BALL_SIZE = _layout(BALL_BLOCKS)
MATE, MATE_SIZE = _layout(MATE_BLOCKS)
OPP, OPP_SIZE = _layout(OPP_BLOCKS)
CONTEXT, CONTEXT_SIZE = _layout(CONTEXT_BLOCKS)
ACTOR_LAYOUT, ACTOR_SIZE = _layout([("self", SELF_SIZE), ("ball", BALL_SIZE),
                                    ("teammates", 10 * MATE_SIZE), ("opponents", N * OPP_SIZE),
                                    ("context", CONTEXT_SIZE)])
assert (SELF_SIZE, BALL_SIZE, MATE_SIZE, OPP_SIZE, CONTEXT_SIZE) == (58, 13, 47, 28, 20)
assert ACTOR_SIZE == 869

# Critic state: everything, in fixed slot order (training only).
STATE_PLAYER_BLOCKS = [("present", 1), ("pos", 2), ("vel", 2), ("role", N_ROLES), ("active", 1),
                       ("yellow", 1), ("tired", 1), ("has_ball", 1), ("attributes", 26)]
STATE_PLAYER, STATE_PLAYER_SIZE = _layout(STATE_PLAYER_BLOCKS)
STATE_GLOBAL, STATE_GLOBAL_SIZE = _layout(
    [("ball_pos", 3), ("ball_vel", 3), ("ball_spin", 3), ("owner", 3), ("score_diff", 1),
     ("time_left", 1), ("game_mode", N_GAME_MODES), ("pitch", len(PITCH_CONDITIONS))])


def state_layout(instruction_size: int):
    return _layout([("global", STATE_GLOBAL_SIZE), ("ours", N * STATE_PLAYER_SIZE),
                    ("theirs", N * STATE_PLAYER_SIZE), ("instructions", N * instruction_size)])


# --- per-team arrays -----------------------------------------------------------------

def _one_hot(index, size: int) -> np.ndarray:
    index = np.asarray(index, dtype=int)
    flat = index.reshape(-1)
    out = np.zeros((flat.size, size), np.float32)
    ok = (flat >= 0) & (flat < size)
    out[np.nonzero(ok)[0], flat[ok]] = 1.0
    return out.reshape(index.shape + (size,))


def attributes(rows: np.ndarray) -> np.ndarray:
    """[n, 24] raw attribute rows -> [n, 26]: 22 stats as is, height
    (h - 1.80) / 0.15, preferred foot one-hot (unspecified, right, left)."""
    rows = np.asarray(rows, np.float32)
    out = np.zeros((len(rows), 26), np.float32)
    if len(rows):
        out[:, :N_STATS] = rows[:, :N_STATS]
        out[:, N_STATS] = (rows[:, N_STATS] - 1.80) / 0.15
        out[:, N_STATS + 1:] = _one_hot(rows[:, N_STATS + 1].astype(int), 3)
    return out


class Team:
    """One side's players as padded [11, ...] arrays (absent slots zero)."""

    def __init__(self, raw, side: str, owner_raw: tuple[int, int]):
        n = len(raw[f"{side}_team"])
        self.present = np.zeros(N, bool)
        self.present[:n] = True
        self.pos = np.zeros((N, 2), np.float32)
        self.pos[:n] = np.asarray(raw[f"{side}_team"])[:, :2] * METRES
        self.vel = np.zeros((N, 2), np.float32)
        self.vel[:n] = np.asarray(raw[f"{side}_team_direction"])[:, :2] * METRES / STEP_S
        self.role = np.zeros((N, N_ROLES), np.float32)
        self.role[:n] = _one_hot(np.asarray(raw[f"{side}_team_roles"]), N_ROLES)
        self.active = np.zeros(N, np.float32)
        self.active[:n] = np.asarray(raw[f"{side}_team_active"], np.float32)
        self.yellow = np.zeros(N, np.float32)
        self.yellow[:n] = np.asarray(raw[f"{side}_team_yellow_card"], np.float32)
        self.tired = np.zeros(N, np.float32)
        self.tired[:n] = np.asarray(raw[f"{side}_team_tired_factor"], np.float32)
        self.attrs_raw = np.zeros((N, 24), np.float32)
        self.attrs_raw[:n] = np.asarray(raw[f"{side}_team_attributes"], np.float32)
        self.attrs = attributes(self.attrs_raw) * self.present[:, None]
        team_id = 0 if side == "left" else 1
        self.has_ball = np.zeros(N, np.float32)
        if owner_raw[0] == team_id and 0 <= owner_raw[1] < n:
            self.has_ball[owner_raw[1]] = 1.0


# --- the encoder -----------------------------------------------------------------------

class Encoder:
    def __init__(self, pitch_condition: str = "normal"):
        self.pitch = _one_hot(PITCH_CONDITIONS.index(pitch_condition), len(PITCH_CONDITIONS))

    def actor(self, raw, sticky: np.ndarray, memory_team: int):
        """raw: the left team's raw observation (any agent's; positions are
        absolute). sticky: [11, 13] each player's held buttons. memory_team:
        the possession memory's team. Returns ([11, 869], us, them, ball)."""
        owner_raw = (int(raw["ball_owned_team"]), int(raw["ball_owned_player"]))
        us, them = Team(raw, "left", owner_raw), Team(raw, "right", owner_raw)
        ball = np.asarray(raw["ball"], np.float32)
        ball_m = np.array([ball[0] * METRES[0], ball[1] * METRES[1], ball[2]], np.float32)
        bdir = np.asarray(raw["ball_direction"], np.float32)
        ball_v = np.array([bdir[0] * METRES[0], bdir[1] * METRES[1], bdir[2]], np.float32) / STEP_S
        owner = _one_hot({0: 0, 1: 1}.get(memory_team, 2), 3)

        out = np.zeros((N, ACTOR_SIZE), np.float32)
        # Self.
        s = np.zeros((N, SELF_SIZE), np.float32)
        s[:, SELF["pos"]] = us.pos / POS_SCALE
        s[:, SELF["vel"]] = us.vel / PLAYER_SPEED
        s[:, SELF["speed"]] = np.linalg.norm(us.vel, axis=1, keepdims=True) / PLAYER_SPEED
        s[:, SELF["tired"]] = us.tired[:, None]
        s[:, SELF["yellow"]] = us.yellow[:, None]
        s[:, SELF["active"]] = us.active[:, None]
        s[:, SELF["role"]] = us.role
        s[:, SELF["attributes"]] = us.attrs
        s[:, SELF["has_ball"]] = us.has_ball[:, None]
        s[:, SELF["sticky"]] = sticky
        # Ball.
        b = np.zeros((N, BALL_SIZE), np.float32)
        rel = ball_m[None, :2] - us.pos
        b[:, BALL["pos"]] = ball_m / BALL_SCALE
        b[:, BALL["rel"]] = rel / POS_SCALE
        b[:, BALL["dist"]] = np.linalg.norm(rel, axis=1, keepdims=True) / DIST_SCALE
        b[:, BALL["vel"]] = ball_v / BALL_SPEED
        b[:, BALL["owner"]] = owner
        b[:, BALL["mine"]] = us.has_ball[:, None]
        # Teammates: rows for every slot j, then drop j == i.
        rel_us = us.pos[None, :, :] - us.pos[:, None, :]                  # [i, j, 2]
        mates = np.zeros((N, N, MATE_SIZE), np.float32)
        mates[..., MATE["rel"]] = rel_us / POS_SCALE
        mates[..., MATE["dist"]] = np.linalg.norm(rel_us, axis=2, keepdims=True) / DIST_SCALE
        mates[..., MATE["pos"]] = (us.pos / POS_SCALE)[None]
        mates[..., MATE["vel"]] = (us.vel / PLAYER_SPEED)[None]
        mates[..., MATE["role"]] = us.role[None]
        mates[..., MATE["active"]] = us.active[None, :, None]
        mates[..., MATE["yellow"]] = us.yellow[None, :, None]
        mates[..., MATE["tired"]] = us.tired[None, :, None]
        mates[..., MATE["has_ball"]] = us.has_ball[None, :, None]
        mates[..., MATE["attributes"]] = us.attrs[None]
        mates *= us.present[None, :, None]
        keep = ~np.eye(N, dtype=bool)
        mates = mates[keep].reshape(N, N - 1, MATE_SIZE)
        # Opponents.
        rel_them = them.pos[None, :, :] - us.pos[:, None, :]
        opp = np.zeros((N, N, OPP_SIZE), np.float32)
        opp[..., OPP["rel"]] = rel_them / POS_SCALE
        opp[..., OPP["dist"]] = np.linalg.norm(rel_them, axis=2, keepdims=True) / DIST_SCALE
        opp[..., OPP["pos"]] = (them.pos / POS_SCALE)[None]
        opp[..., OPP["vel"]] = (them.vel / PLAYER_SPEED)[None]
        opp[..., OPP["role"]] = them.role[None]
        opp[..., OPP["active"]] = them.active[None, :, None]
        opp[..., OPP["yellow"]] = them.yellow[None, :, None]
        opp[..., OPP["tired"]] = them.tired[None, :, None]
        opp[..., OPP["has_ball"]] = them.has_ball[None, :, None]
        ball_gap = np.linalg.norm(them.pos - ball_m[None, :2], axis=1) / DIST_SCALE
        opp[..., OPP["ball_dist"]] = ball_gap[None, :, None]
        key = np.concatenate([them.attrs[:, list(KEY_STATS)], them.attrs[:, N_STATS:N_STATS + 1]], axis=1)
        opp[..., OPP["key_attributes"]] = key[None]
        opp *= them.present[None, :, None]
        # Context.
        c = np.zeros((N, CONTEXT_SIZE), np.float32)
        to_goal = OPP_GOAL[None] - us.pos
        dist = np.linalg.norm(to_goal, axis=1, keepdims=True)
        c[:, CONTEXT["opp_goal_rel"]] = to_goal / POS_SCALE
        c[:, CONTEXT["opp_goal_dist"]] = dist / DIST_SCALE
        c[:, CONTEXT["opp_goal_angle"]] = np.stack(
            [to_goal[:, 1], to_goal[:, 0]], axis=1) / np.maximum(dist, 1e-6)  # sin, cos
        c[:, CONTEXT["own_goal_rel"]] = (OWN_GOAL[None] - us.pos) / POS_SCALE
        score = raw["score"]
        c[:, CONTEXT["score_diff"]] = np.clip(score[0] - score[1], -5, 5) / 5.0
        c[:, CONTEXT["time_left"]] = raw["steps_left"] / STEPS_SCALE
        c[:, CONTEXT["game_mode"]] = _one_hot(int(raw["game_mode"]), N_GAME_MODES)
        c[:, CONTEXT["pitch"]] = self.pitch

        out[:, ACTOR_LAYOUT["self"]] = s
        out[:, ACTOR_LAYOUT["ball"]] = b
        out[:, ACTOR_LAYOUT["teammates"]] = mates.reshape(N, -1)
        out[:, ACTOR_LAYOUT["opponents"]] = opp.reshape(N, -1)
        out[:, ACTOR_LAYOUT["context"]] = c
        out *= us.present[:, None]  # absent players' own rows are zero
        return out, us, them, (ball_m, ball_v)

    def state(self, raw, us: Team, them: Team, ball, memory_team: int,
              instructions: np.ndarray) -> np.ndarray:
        """The critic's global state: ball (with spin), every player in slot
        order with full attributes, score, clock, and every player's
        instruction block."""
        ball_m, ball_v = ball
        g = np.zeros(STATE_GLOBAL_SIZE, np.float32)
        g[STATE_GLOBAL["ball_pos"]] = ball_m / BALL_SCALE
        g[STATE_GLOBAL["ball_vel"]] = ball_v / BALL_SPEED
        # Spin: rotation angle per step -> rad/s, scaled to roughly [-1, 1].
        g[STATE_GLOBAL["ball_spin"]] = np.asarray(raw["ball_rotation"], np.float32) / STEP_S / 100.0
        g[STATE_GLOBAL["owner"]] = _one_hot({0: 0, 1: 1}.get(memory_team, 2), 3)
        score = raw["score"]
        g[STATE_GLOBAL["score_diff"]] = np.clip(score[0] - score[1], -5, 5) / 5.0
        g[STATE_GLOBAL["time_left"]] = raw["steps_left"] / STEPS_SCALE
        g[STATE_GLOBAL["game_mode"]] = _one_hot(int(raw["game_mode"]), N_GAME_MODES)
        g[STATE_GLOBAL["pitch"]] = self.pitch
        sides = []
        for team in (us, them):
            p = np.zeros((N, STATE_PLAYER_SIZE), np.float32)
            p[:, STATE_PLAYER["present"]] = team.present[:, None]
            p[:, STATE_PLAYER["pos"]] = team.pos / POS_SCALE
            p[:, STATE_PLAYER["vel"]] = team.vel / PLAYER_SPEED
            p[:, STATE_PLAYER["role"]] = team.role
            p[:, STATE_PLAYER["active"]] = team.active[:, None]
            p[:, STATE_PLAYER["yellow"]] = team.yellow[:, None]
            p[:, STATE_PLAYER["tired"]] = team.tired[:, None]
            p[:, STATE_PLAYER["has_ball"]] = team.has_ball[:, None]
            p[:, STATE_PLAYER["attributes"]] = team.attrs
            sides.append((p * team.present[:, None]).ravel())
        return np.concatenate([g, *sides, np.asarray(instructions, np.float32).ravel()])
