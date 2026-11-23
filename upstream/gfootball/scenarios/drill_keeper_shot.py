# rl_project goalkeeper drill: the right team's keeper alone in goal. A test
# script launches shots at the goal with launch_ball (no shooter run-in, so the
# keeper has to save, not rush out). The left team has one player parked far
# away so an agent can be attached to it without touching play.
# Used by grf_tools.keeper_shot_test. Not part of upstream GRF.

from . import *


def build_scenario(builder):
  builder.config().game_duration = 80  # 8 s, plenty for one shot
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = True  # keeper holds it = saved
  builder.SetBallPosition(0.64, 0.0)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=False)
  builder.AddPlayer(-0.2, 0.40, e_PlayerRole_CF, lazy=True)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # own frame: x = -1 is its goal
