# rl_project measurement drill: an empty pitch for launching the ball with an
# exact velocity and spin (grf_tools.ballistics). Every player is lazy and
# parked at a corner flag so nobody touches the ball. Not part of upstream GRF.

from . import *


def build_scenario(builder):
  builder.config().game_duration = 3000
  builder.config().deterministic = True
  builder.config().offsides = False
  builder.config().end_episode_on_score = False
  builder.config().end_episode_on_out_of_play = False
  builder.config().end_episode_on_possession_change = False
  builder.SetBallPosition(0.0, 0.0)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.42, e_PlayerRole_GK, lazy=True, controllable=False)
  builder.AddPlayer(-1.0, -0.42, e_PlayerRole_CB, lazy=True)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(1.0, 0.42, e_PlayerRole_GK, lazy=True, controllable=False)
