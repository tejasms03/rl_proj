"""The 3v3 scene: three outfield players and a keeper a side.

Registered as `gfootball.scenarios.outfield_3v3` without touching the
installed package (the scenario loader checks sys.modules first), as the duty
lab does. Import this module before creating the env.

Player indices on both teams: 0 GK, 1 central midfielder (starts on the
ball), 2 left wing (top of the screen), 3 right wing.

Coordinates: ball and left team are absolute (we attack +x, y grows down the
screen); right-team players are in that team's own mirrored frame, so their
(-0.45, -0.12) is (0.45, 0.12) on screen.
"""

from __future__ import annotations

import sys
import types

from gfootball.scenarios import (
    Team, e_PlayerRole_CB, e_PlayerRole_CM, e_PlayerRole_GK, e_PlayerRole_LM, e_PlayerRole_RM)

NAME = "outfield_3v3"
EPISODE_STEPS = 300  # 30 s of game at 0.1 s per step


def build_scenario(builder):
    config = builder.config()
    config.game_duration = EPISODE_STEPS
    config.deterministic = False  # the seed decides; the same seed replays exactly
    config.offsides = False  # one defender would be the offside line: keep the first play simple
    config.end_episode_on_score = True
    config.end_episode_on_out_of_play = True
    config.end_episode_on_possession_change = True  # one attack per episode
    builder.SetBallPosition(-0.02, 0.0)  # ~1 m in front of our midfielder

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.04, 0.0, e_PlayerRole_CM)
    builder.AddPlayer(-0.10, -0.20, e_PlayerRole_LM)
    builder.AddPlayer(-0.10, 0.20, e_PlayerRole_RM)

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.45, -0.12, e_PlayerRole_CB)
    builder.AddPlayer(-0.45, 0.12, e_PlayerRole_CB)
    builder.AddPlayer(-0.20, 0.0, e_PlayerRole_CM)


def register() -> None:
    module = types.ModuleType(f"gfootball.scenarios.{NAME}")
    module.build_scenario = build_scenario
    sys.modules[module.__name__] = module


register()
