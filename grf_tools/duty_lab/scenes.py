"""Lab scenes, registered as `gfootball.scenarios.lab_<name>` without touching
the installed package (the scenario loader checks sys.modules first).

Every scene is full 11v11: two 4-4-2s placed for a phase of play. Player
indices are the same on both teams: 0 GK, 1 LB, 2-3 CB, 4 RB, 5 LM, 6-7 CM,
8 RM, 9-10 CF.

Coordinates: ball and left team are absolute (we attack +x, y grows down the
screen); right-team players are in that team's own mirrored frame, so their
(-0.75, -0.09) is (0.75, 0.09) on screen.
"""

from __future__ import annotations

import sys
import types

from gfootball.scenarios import (
    Team, e_PlayerRole_CB, e_PlayerRole_CF, e_PlayerRole_CM, e_PlayerRole_GK, e_PlayerRole_LB,
    e_PlayerRole_LM, e_PlayerRole_RB, e_PlayerRole_RM)

_ROLES = [e_PlayerRole_GK, e_PlayerRole_LB, e_PlayerRole_CB, e_PlayerRole_CB, e_PlayerRole_RB,
          e_PlayerRole_LM, e_PlayerRole_CM, e_PlayerRole_CM, e_PlayerRole_RM,
          e_PlayerRole_CF, e_PlayerRole_CF]
LANES = [-0.27, -0.09, 0.09, 0.27]  # y of the four across the back and midfield


def formation(defence: float, midfield: float, attack: float) -> list[tuple[float, float]]:
    """4-4-2 at the given depths, in the team's own frame (attacking +x)."""
    return ([(-1.0, 0.0)] + [(defence, y) for y in LANES] + [(midfield, y) for y in LANES]
            + [(attack, -0.07), (attack, 0.07)])


def _moved(spots, index, spot):
    spots = list(spots)
    spots[index] = spot
    return spots


def _teams(builder, ball, ours, theirs):
    config = builder.config()
    config.game_duration = 3000  # the runner decides how long a scene lasts
    config.deterministic = False  # the seed decides; the same seed replays exactly
    config.offsides = True
    config.end_episode_on_score = True
    config.end_episode_on_out_of_play = True
    config.end_episode_on_possession_change = False
    builder.SetBallPosition(*ball)
    for team, spots in ((Team.e_Left, ours), (Team.e_Right, theirs)):
        builder.SetTeam(team)
        for (x, y), role in zip(spots, _ROLES):
            builder.AddPlayer(x, y, role)


# The ball starts 1.6 m in front of the player meant to have it; he walks onto
# it (run.py's opening rules: theirs in defending scenes, ours in "our_*").
SCENES = {
    # Their centre-back (index 2) has it deep; we are set up to press.
    "build_up": lambda b: _teams(b, (0.72, 0.09), formation(-0.2, 0.1, 0.42),
                                 formation(-0.75, -0.52, -0.1)),
    # Their central midfielder (index 6) has it at halfway; we defend our half.
    "attack": lambda b: _teams(b, (-0.08, 0.09), formation(-0.5, -0.25, 0.05),
                               formation(-0.3, 0.05, 0.3)),
    # As "attack", but our back four starts high: room in behind for the keeper.
    "high_line": lambda b: _teams(b, (-0.13, 0.09), formation(-0.25, 0.0, 0.3),
                                  formation(-0.2, 0.1, 0.2)),
    # We have it. Our centre-back (index 2) deep, they press.
    "our_build": lambda b: _teams(b, (-0.72, -0.09), formation(-0.75, -0.45, -0.1),
                                  formation(-0.2, 0.1, 0.42)),
    # Our central midfielder (index 6) has it at halfway; they defend deep.
    "our_attack": lambda b: _teams(b, (0.08, -0.09), formation(-0.3, 0.05, 0.3),
                                   formation(-0.5, -0.25, 0.05)),
    # Our right winger (index 8) has it near their byline; they defend their box.
    "our_wing": lambda b: _teams(b, (0.68, 0.30), _moved(formation(-0.2, 0.3, 0.6), 8, (0.65, 0.30)),
                                 formation(-0.75, -0.55, -0.2)),
    # Our right winger (index 8) has it on the wing at halfway with a moment of
    # space (their midfield 17 m off), our right-back (4) 8 m behind him.
    "our_wide": lambda b: _teams(b, (0.18, 0.27),
                                 _moved(formation(-0.3, 0.15, 0.4), 4, (0.03, 0.27)),
                                 formation(-0.7, -0.5, -0.2)),
}


def register() -> None:
    for name, build in SCENES.items():
        module = types.ModuleType(f"gfootball.scenarios.lab_{name}")
        module.build_scenario = build
        sys.modules[module.__name__] = module


register()
