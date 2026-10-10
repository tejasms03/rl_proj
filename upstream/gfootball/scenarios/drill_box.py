# rl_project test drill: attacker on the ball at the edge of the opponent box
# with two active (non-lazy) defenders closing in, to check that kicks
# register under pressure in the box. Not part of upstream GRF.

from . import *


def build_scenario(builder):
  builder.config().game_duration = 400
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = False
  builder.SetBallPosition(0.72, 0.05)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=False)
  builder.AddPlayer(0.71, 0.05, e_PlayerRole_CF)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(0.76, 0.03, e_PlayerRole_CB)
  builder.AddPlayer(0.75, 0.10, e_PlayerRole_CB)
