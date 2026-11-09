# rl_project test drill: an AI attacker runs at you with the ball from the
# halfway line; you control the one defender in front of your goal. Used to
# measure how well the gamepad's defending buttons win the ball. Not part of
# upstream GRF.

from . import *


def build_scenario(builder):
  builder.config().game_duration = 600
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = False
  builder.SetBallPosition(-0.45, 0.0)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=False)
  builder.AddPlayer(-0.78, 0.05, e_PlayerRole_CB)

  builder.SetTeam(Team.e_Right)
  # Right-team positions are in that team's own (mirrored) frame.
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True)
  builder.AddPlayer(0.44, 0.0, e_PlayerRole_CF)
