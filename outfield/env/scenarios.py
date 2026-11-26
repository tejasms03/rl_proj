"""Scenes the wrapper can run, and how many players each side has.

Importing this module registers our own scenes next to the game's built-in
ones: the duty lab's `lab_*` scenes and `outfield_3v3`.
"""

from __future__ import annotations

from functools import lru_cache

import grf_tools.duty_lab.scenes  # noqa: F401  registers lab_* scenes
import outfield.scripted.scene  # noqa: F401  registers outfield_3v3
from gfootball.env import config as gfootball_config


@lru_cache(maxsize=None)
def counts(name: str) -> tuple[int, int]:
    """(left players, right players) the scene puts on the pitch."""
    scenario = gfootball_config.Config({"level": name}).ScenarioConfig()
    return int(scenario.controllable_left_players), int(scenario.controllable_right_players)


def is_full_match(name: str) -> bool:
    """A real 11v11 match: the end of time is the final whistle (terminal),
    not a drill's time limit (truncation)."""
    return name.startswith("11_vs_11")
