# rl_project goalkeeper drill: the right team's keeper alone in goal. A test
# script launches shots at the goal with launch_ball (no shooter run-in, so the
# keeper has to save, not rush out). The left team has one player parked far
# away so an agent can be attached to it without touching play.
# Used by grf_tools.keeper_shot_test. Not part of upstream GRF.

from . import *

# keeper holds the ball in his box, a teammates stands upfield; used to check what high_pass looks like for a keeper

def build_scenario(builder):
  builder.config().game_duration = 100  # 10 s, enough to watch the pass land
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = False
  builder.config().end_episode_on_out_of_play = False
  builder.config().end_episode_on_possession_change = False 
  builder.SetBallPosition(-0.95, 0.0) #ball at keeper's feet infront of the goal (keeper stands at x = -1)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(-0.5, 0.2, e_PlayerRole_CM, lazy=True)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True)  # own frame: x = -1 is its goal
