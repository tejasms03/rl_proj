"""Run episodes with the built-in AI driving the controlled players."""

from __future__ import annotations

import dataclasses
from typing import Any

from grf_tools.trace import make_env

from gfootball.env import football_action_set  # isort: skip

_V2 = [a._name for a in football_action_set.get_action_set({"action_set": "v2"})]
BUILTIN_AI = _V2.index("builtin_ai")


@dataclasses.dataclass
class Episode:
    score: tuple[int, int]
    first_owner: int  # team that first gained possession (0 left, 1 right, -1 none)
    steps: int


def ai_episode(scenario: str, seed: int, config: dict[str, Any] | None = None) -> Episode:
    """One episode where every controlled player is played by the built-in AI."""
    env = make_env(scenario, seed, {"action_set": "v2", **(config or {})})
    try:
        env.reset()
        done, owner, steps = False, -1, 0
        while not done:
            obs, _, done, _ = env.step(BUILTIN_AI)
            o = obs[0]
            steps += 1
            if owner == -1 and o["ball_owned_team"] != -1:
                owner = int(o["ball_owned_team"])
    finally:
        env.close()
    return Episode(tuple(int(s) for s in o["score"]), owner, steps)
