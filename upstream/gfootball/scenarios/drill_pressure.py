# rl_project test drill: drill_shot with a (lazy) defender standing about a
# metre from the shooter, to measure shooting under pressure. Not part of
# upstream GRF.

from . import *


def build_scenario(builder):
  builder.config().game_duration = 600
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = False
  builder.SetBallPosition(0.6, 0.0)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=False)
  builder.AddPlayer(0.59, 0.0, e_PlayerRole_CF)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(1.0, 0.4, e_PlayerRole_GK, lazy=True)
  builder.AddPlayer(0.615, 0.015, e_PlayerRole_CB, lazy=True)
