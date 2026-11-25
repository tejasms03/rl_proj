"""OutfieldEnv: the wrapper O1 (outfield), G (keeper) and M2 (manager) train on.

    env = OutfieldEnv("11_vs_11_easy_stochastic", seed=1)
    obs = env.reset()
    obs, rewards, terminated, truncated, info = env.step(actions, instructions=None)

Our team is the left team, against the game's AI. All our players are
controlled (the engine assigns players to agents in a fixed order only when
every player is controlled); slot 0, the keeper, is left to the game's AI
unless keeper="agent". Shapes are always 11 rows: drills with fewer players
pad with zero rows and `obs["present"]` False.

Interface (docs/env_wrapper.md):
  obs["actor"]     float32[11, 960]  869 base (observation.py) + 91 instruction block
  obs["state"]     float32[S]        the critic's global state
  obs["present"]   bool[11]
  obs["masks"]     {head: bool[11, n]} for move (17), speed (3), ball (8), defend (5)
  actions          {"move","speed","ball","defend": int[11], "power": float[11]}
  instructions     float32[11, 85] or None (None = the built-in sampler, if any)
  rewards          {"score": float32[11]}: placeholder (goal for +1, against -1);
                   the real reward terms are planned separately
  info             overridden bool[11], duty_ok float32[11] (NaN until duty checks
                   land), kicks (events), outcome, plan (name), raw (raw obs)
"""

from __future__ import annotations

import numpy as np

import gfootball.env as football_env

from outfield.config import TRAINING_CONFIG
from outfield.env import actions as act
from outfield.env import instructions as ins
from outfield.env import observation as ob
from outfield.env.possession import Possession
from outfield.env.scenarios import counts, is_full_match

N = 11
OBS_SIZE = ob.ACTOR_SIZE + ins.SIZE
STATE_LAYOUT, STATE_SIZE = ob.state_layout(ins.SIZE)
assert OBS_SIZE == 960

_ENGINE_TICKS: int | None = None  # one engine setting per process (docs/CONTEXT.md §7.7)


class OutfieldEnv:
    def __init__(self, scenario: str = "11_vs_11_easy_stochastic", seed: int = 0, *,
                 ai_on_ball: bool = True, keeper: str = "builtin",
                 sampler: ins.Sampler | None | str = "default", render: bool = False,
                 config: dict | None = None):
        global _ENGINE_TICKS
        assert keeper in ("builtin", "agent")
        self.scenario, self.seed, self.render = scenario, seed, render
        self.config = {**TRAINING_CONFIG, **(config or {})}
        ticks = int(self.config.get("physics_steps_per_frame", 10))
        if _ENGINE_TICKS is not None and _ENGINE_TICKS != ticks:
            raise RuntimeError(
                f"this process already ran the engine at {_ENGINE_TICKS} ticks per step; a second "
                f"setting ({ticks}) would silently keep the first. Use another process.")
        _ENGINE_TICKS = ticks
        self.n_left, self.n_right = counts(scenario)
        self.present = np.zeros(N, bool)
        self.present[:self.n_left] = True
        self.keeper_agent = keeper == "agent"
        self.adapter = act.Adapter(ai_on_ball=ai_on_ball, keeper_agent=self.keeper_agent)
        self.sampler = ins.Sampler(seed) if sampler == "default" else sampler
        self.encoder = ob.Encoder(self.config.get("pitch_condition", "normal"))
        self.possession = Possession()
        self.memory = ins.Memory()
        self._env = None
        self._env_seed = None

    # --- lifecycle -------------------------------------------------------------------------

    def _make(self, seed: int):
        if self._env is not None:
            self._env.close()
        self._env = football_env.create_environment(
            self.scenario, representation="raw", render=self.render,
            number_of_left_players_agent_controls=self.n_left,
            number_of_right_players_agent_controls=0,
            other_config_options={**self.config, "game_engine_random_seed": seed})
        self._env_seed = seed

    def reset(self, seed: int | None = None) -> dict:
        """seed: replay that exact game (a fresh engine); None: the next game
        of this engine (its seed moves on by itself)."""
        if self._env is None or seed is not None:
            self._make(self.seed if seed is None else seed)
        raw_list = self._env.reset()
        actives = [int(o["active"]) for o in raw_list]
        assert actives == list(range(self.n_left)), f"agent i must control player i, got {actives}"
        self.raw_list = raw_list
        self.steps = 0
        self.score = tuple(raw_list[0]["score"])
        self.possession.reset()
        self.possession(raw_list[0])
        self.memory.reset()
        self.adapter.reset()
        if self.sampler is not None:
            self.sampler.reset(None if seed is None else seed)
        us = np.zeros((N, 2), np.float32)
        us[:self.n_left] = np.asarray(raw_list[0]["left_team"])[:, :2] * ob.METRES
        self.formation = us  # hold-zone / shape anchors: where everyone starts
        self.manager_part = np.zeros((N, ins.MANAGER_SIZE), np.float32)
        self.plan_name = None
        return self._observe(pressed=[None] * N, instructions=None)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    # --- stepping ------------------------------------------------------------------------------

    def step(self, actions: dict[str, np.ndarray], instructions: np.ndarray | None = None):
        commands, overridden, pressed, kicks = self.adapter.commands(
            self.raw_list, actions, self.masks, self.present, self.steps)
        raw_list, _, done, _ = self._env.step(commands)
        self.raw_list = raw_list
        self.steps += 1
        raw = raw_list[0]
        self.possession(raw)
        obs = self._observe(pressed, instructions)

        score = tuple(raw["score"])
        delta = (score[0] - self.score[0]) - (score[1] - self.score[1])
        self.score = score
        rewards = {"score": np.where(self.present, float(delta), 0.0).astype(np.float32)}

        timed_out = raw["steps_left"] <= 0
        terminated = bool(done and (not timed_out or is_full_match(self.scenario)))
        truncated = bool(done and not terminated)
        info = {"overridden": overridden, "duty_ok": np.full(N, np.nan, np.float32),
                "kicks": kicks, "outcome": _outcome(raw, delta, self.possession.team) if done else None,
                "plan": self.plan_name, "raw": raw_list}
        return obs, rewards, terminated, truncated, info

    def _observe(self, pressed, instructions) -> dict:
        raw = self.raw_list[0]
        sticky = np.zeros((N, ob.N_STICKY), np.float32)
        for i, o in enumerate(self.raw_list):
            sticky[i] = o["sticky_actions"]
        actor, us, them, ball = self.encoder.actor(raw, sticky, self.possession.team)

        if instructions is not None:
            manager = np.asarray(instructions, np.float32).reshape(N, ins.MANAGER_SIZE)
            self.plan_name = "given"
        elif self.sampler is not None:
            roles = np.full(N, -1)
            roles[:self.n_left] = raw["left_team_roles"]
            plan = self.sampler(self.steps, roles, self.formation, us.pos, them.pos,
                                self.present, them.present)
            manager, self.plan_name = ins.encode(plan, self.present), plan.name
        else:
            manager = np.zeros((N, ins.MANAGER_SIZE), np.float32)
        manager = manager * self.present[:, None]
        owner = (int(raw["ball_owned_team"]), int(raw["ball_owned_player"]))
        touching = owner[1] if owner[0] == 0 else None
        memory = self.memory.update(pressed, touching, self.possession.changed_steps, manager)
        block = np.concatenate([manager, memory * self.present[:, None]], axis=1)
        self.manager_part = manager

        self.masks = act.masks(raw, self.possession.team, self.present, self.keeper_agent)
        return {"actor": np.concatenate([actor, block], axis=1),
                "state": self.encoder.state(raw, us, them, ball, self.possession.team, block),
                "present": self.present.copy(),
                "masks": {k: v.copy() for k, v in self.masks.items()}}


def _outcome(raw, delta, possession_team) -> str:
    if delta > 0:
        return "goal_for"
    if delta < 0:
        return "goal_against"
    if raw["steps_left"] <= 0:
        return "time"
    if raw["game_mode"] != 0:
        return "out_of_play"
    if raw["ball_owned_team"] == 1 or possession_team == 1:
        return "lost_ball"
    return "other"
