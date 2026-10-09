# rl_project measurement drill: a midfielder with the ball and one teammate
# about 40 m ahead, no defenders. Used by grf_tools.measure to record ball flight
# from real passes. Not part of upstream GRF.

from . import *


def build_scenario(builder):
  builder.config().game_duration = 200
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = False
  builder.SetBallPosition(-0.2, 0.0)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=False)
  builder.AddPlayer(-0.21, 0.0, e_PlayerRole_CM)
  builder.AddPlayer(0.53, 0.0, e_PlayerRole_CF, lazy=True, controllable=False)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(1.0, 0.0, e_PlayerRole_GK, lazy=True)
