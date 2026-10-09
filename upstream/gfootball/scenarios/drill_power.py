# rl_project measurement drill: a passer with the ball and 80 m of open
# pitch ahead; the only teammate is parked far out wide (passing needs one).
# Used to measure how far each kick power sends the ball. Not part of
# upstream GRF.

from . import *


def build_scenario(builder):
  builder.config().game_duration = 600
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = False
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = False
  builder.SetBallPosition(-0.7, 0.0)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=False)
  builder.AddPlayer(-0.71, 0.0, e_PlayerRole_CM)
  builder.AddPlayer(-0.9, 0.4, e_PlayerRole_LB, lazy=True, controllable=False)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(1.0, 0.0, e_PlayerRole_GK, lazy=True)
