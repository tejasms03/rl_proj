# rl_project goalkeeper drill: one attacker with the ball, run by the built-in
# AI, shoots at the right team's goalkeeper. The right team has only the
# keeper, so an agent that controls one right-team player controls the keeper.
# Used by grf_tools.keeper_save_test. Not part of upstream GRF.

from . import *

# Where the attacker starts (left team's frame: x = 1 is the goal it attacks).
# The episode number picks one, so a run covers every spot evenly.
SHOOTER_SPOTS = [(0.62, -0.20), (0.62, -0.10), (0.62, 0.0), (0.62, 0.10), (0.62, 0.20)]


def build_scenario(builder):
  builder.config().game_duration = 300
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = True  # keeper holds it = saved

  x, y = SHOOTER_SPOTS[builder.EpisodeNumber() % len(SHOOTER_SPOTS)]
  builder.SetBallPosition(x, y)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=False)
  builder.AddPlayer(x - 0.01, y, e_PlayerRole_CF)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # own frame: x = -1 is its goal
