# rl_project goalkeeper drill: same as drill_keeper_pass, plus one opponent
# standing in the passing lane between our keeper and the CM, so a short pass
# gets intercepted. Used by grf_tools.keeper_intercept_test to check the
# keeper reward's -0.8 for a pass that doesn't reach a teammate.

from . import *


def build_scenario(builder):
  builder.config().game_duration = 100  # 10 s
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = False
  builder.config().end_episode_on_out_of_play = False
  builder.config().end_episode_on_possession_change = False
  builder.SetBallPosition(-0.95, 0.0)  # ball at our keeper's feet

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(-0.5, 0.2, e_PlayerRole_CM, lazy=True)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True)  # own frame: x = -1 is its goal
  # In the passing lane: midpoint of keeper (-0.95, 0) and CM (-0.5, 0.2) is
  # (-0.725, 0.1) in our frame, which is (0.725, -0.1) in the right team's frame.
  builder.AddPlayer(0.725, -0.1, e_PlayerRole_CM)
