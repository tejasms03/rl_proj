# rl_project measurement drill: one outfield player standing still with the
# ball far away, free to sprint 60+ m straight ahead. Used to measure top
# speed and acceleration. The ball sits with our own lazy keeper so the
# engine never steers the sprinter toward a loose ball. Not part of
# upstream GRF.

from . import *


def build_scenario(builder):
  builder.config().game_duration = 120
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = False
  builder.config().end_episode_on_out_of_play = False
  builder.config().end_episode_on_possession_change = False
  builder.SetBallPosition(-0.99, 0.0)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=False)
  builder.AddPlayer(-0.7, 0.0, e_PlayerRole_CF)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(1.0, 0.0, e_PlayerRole_GK, lazy=True)
