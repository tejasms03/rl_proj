# rl_project test drill: a loose ball on the centre spot and one outfield
# player per side, equally far from it. Whoever wins the ball ends the
# episode. Used to check that speed stats change who gets there first.
# Not part of upstream GRF.

from . import *


def build_scenario(builder):
  builder.config().game_duration = 100
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = True
  builder.config().left_team_difficulty = 1.0
  builder.config().right_team_difficulty = 1.0
  builder.SetBallPosition(0.0, 0.0)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=False)
  builder.AddPlayer(-0.4, 0.0, e_PlayerRole_CF)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=False)
  builder.AddPlayer(-0.4, 0.0, e_PlayerRole_CF)
