# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Class responsible for generating scenarios."""

import importlib
import os
import pkgutil
import random
import sys

from absl import flags
from absl import logging

import gfootball_engine as libgame

Player = libgame.FormationEntry
Role = libgame.e_PlayerRole
Team = libgame.e_Team

# rl_project P1: names accepted in AddPlayer(stats={...}).
PLAYER_STATS = {name: int(value) for name, value in libgame.e_PlayerStat.names.items()}
PREFERRED_FOOT = {None: 0, 'right': 1, 'left': 2}

# rl_project patch switches (ScenarioConfig fields, all off by default).
# Scenarios may set them via builder.config(); an env option of the same
# name (other_config_options) takes precedence.
PATCH_SWITCHES = ('realistic_player_physics', 'realistic_ball_physics',
                  'pitch_condition', 'ball_pace', 'kick_power_control',
                  'pass_direction_assist', 'pass_power_assist',
                  'situational_accuracy', 'use_magnet',
                  'kick_windup_speed', 'left_team_name', 'right_team_name',
                  'left_team_short_name', 'right_team_short_name',
                  'injury_time', 'pass_reach_assist', 'shot_power_floor',
                  'defend_auto_switch', 'switch_to_ball', 'jockey', 'auto_receive',
                  'air_ball_switch', 'landing_marker', 'selection_ring')
PITCH_CONDITIONS = {'normal': 0, 'dry': 1, 'wet': 2, 'long_grass': 3}
BALL_PACES = {'game': 0, 'realistic': 1}
# Switches set by name; the engine stores their index.
NAMED_SWITCHES = {'pitch_condition': PITCH_CONDITIONS, 'ball_pace': BALL_PACES}


def _SetPlayerAttributes(player, stats=None, height=None, preferred_foot=None,
                         name=None):
  """rl_project P1: validate and store per-player overrides on an entry."""
  for name, value in (stats or {}).items():
    if name not in PLAYER_STATS:
      raise ValueError('Unknown player stat %r; expected one of %s' %
                       (name, sorted(PLAYER_STATS)))
    if not 0.0 <= value <= 1.0:
      raise ValueError('Stat %s=%r is outside [0, 1]' % (name, value))
    player.set_stat(PLAYER_STATS[name], float(value))
  if height is not None:
    if not 1.4 <= height <= 2.2:
      raise ValueError('height=%r m is outside [1.4, 2.2]' % (height,))
    player.height = float(height)
  if preferred_foot not in PREFERRED_FOOT:
    raise ValueError('preferred_foot must be one of %s' %
                     (sorted(k for k in PREFERRED_FOOT if k),))
  if preferred_foot is not None:
    player.preferred_foot = PREFERRED_FOOT[preferred_foot]
  if name is not None:
    player.name = str(name)

FLAGS = flags.FLAGS


def all_scenarios():
  path = os.path.abspath(__file__)
  path = os.path.join(os.path.dirname(os.path.dirname(path)), 'scenarios')
  scenarios = []
  for m in pkgutil.iter_modules([path]):
    # There was API change in pkgutil between Python 3.5 and 3.6...
    if m.__class__ == tuple:
      scenarios.append(m[1])
    else:
      scenarios.append(m.name)
  return scenarios


class Scenario(object):

  def __init__(self, config):
    # Game config controls C++ engine and is derived from the main config.
    self._scenario_cfg = libgame.ScenarioConfig.make()
    self._config = config
    self._active_team = Team.e_Left
    scenario = None
    try:
      scenario = importlib.import_module('gfootball.scenarios.{}'.format(config['level']))
    except ImportError as e:
      logging.error('Loading scenario "%s" failed' % config['level'])
      logging.error(e)
      sys.exit(1)
    scenario.build_scenario(self)
    self.SetTeam(libgame.e_Team.e_Left)
    self._FakePlayersForEmptyTeam(self._scenario_cfg.left_team)
    self.SetTeam(libgame.e_Team.e_Right)
    self._FakePlayersForEmptyTeam(self._scenario_cfg.right_team)
    self._ApplyPlayerOverrides()
    self._ApplyPatchSwitches()
    self._KeepMatchLength()
    self._BuildScenarioConfig()

  def _KeepMatchLength(self):
    """rl_project: with `keep_match_length`, scenarios' game_duration (written
    for the default 10 physics ticks per step) keeps its length in game time
    at other physics_steps_per_frame, e.g. a full match at 50 Hz input."""
    if 'keep_match_length' in self._config and self._config['keep_match_length']:
      ticks = self._config['physics_steps_per_frame']
      self._scenario_cfg.game_duration = int(
          round(self._scenario_cfg.game_duration * 10 / ticks))

  def _ApplyPatchSwitches(self):
    for name in PATCH_SWITCHES:
      if name in self._config:
        value = self._config[name]
        if name in NAMED_SWITCHES:
          names = NAMED_SWITCHES[name]
          if value not in names:
            raise ValueError('%s must be one of %s, got %r' %
                             (name, sorted(names), value))
          value = names[value]
        setattr(self._scenario_cfg, name, value)

  def _ApplyPlayerOverrides(self):
    """rl_project P1: apply the `player_overrides` env option to any scenario.

    Format: {'left' | 'right': {slot | '*': {'stats': {...}, 'height': m,
    'preferred_foot': 'left' | 'right', 'name': str}}}, where slot is the player's index in
    the order the scenario added it and '*' means every player on that side.
    """
    overrides = (self._config['player_overrides']
                 if 'player_overrides' in self._config else None)
    if not overrides:
      return
    teams = {'left': self._scenario_cfg.left_team,
             'right': self._scenario_cfg.right_team}
    unknown = set(overrides) - set(teams)
    if unknown:
      raise ValueError('player_overrides sides must be left/right, got %s' %
                       sorted(unknown))
    for side, slots in overrides.items():
      team = teams[side]
      for slot, spec in slots.items():
        indices = range(len(team)) if slot == '*' else [int(slot)]
        for i in indices:
          if not 0 <= i < len(team):
            raise ValueError('player_overrides: %s team has no slot %d' % (side, i))
          entry = team[i]
          _SetPlayerAttributes(entry, **spec)
          team[i] = entry

  def _FakePlayersForEmptyTeam(self, team):
    if len(team) == 0:
      self.AddPlayer(-1.000000, 0.420000, libgame.e_PlayerRole.e_PlayerRole_GK, True)

  def _BuildScenarioConfig(self):
    """Builds scenario config from gfootball.environment config."""
    self._scenario_cfg.real_time = self._config['real_time']
    self._scenario_cfg.left_agents = self._config.number_of_left_players()
    self._scenario_cfg.right_agents = self._config.number_of_right_players()
    # This is needed to record 'game_engine_random_seed' in the dump.
    if 'game_engine_random_seed' not in self._config._values:
      self._config.set_scenario_value('game_engine_random_seed',
                                      random.randint(0, 2000000000))
    if not self._scenario_cfg.deterministic:
      self._scenario_cfg.game_engine_random_seed = (
          self._config['game_engine_random_seed'])
      if 'reverse_team_processing' not in self._config:
        self._config['reverse_team_processing'] = (
            bool(self._config['game_engine_random_seed'] % 2))
    if 'reverse_team_processing' in self._config:
      self._scenario_cfg.reverse_team_processing = (
          self._config['reverse_team_processing'])

  def config(self):
    return self._scenario_cfg

  def SetTeam(self, team):
    self._active_team = team

  def AddPlayer(self, x, y, role, lazy=False, controllable=True, stats=None,
                height=None, preferred_foot=None, name=None):
    """Build player for the current scenario.

    Args:
      x: x coordinate of the player in the range [-1, 1].
      y: y coordinate of the player in the range [-0.42, 0.42].
      role: Player's role in the game (goal keeper etc.).
      lazy: Computer doesn't perform any automatic actions for lazy player.
      controllable: Whether player can be controlled.
      stats: Optional {stat name: value in [0, 1]} overriding the built-in
        profile, e.g. {'physical_velocity': 0.9}. Names: PLAYER_STATS.
      height: Optional height in metres overriding the built-in profile.
      preferred_foot: Optional 'right' or 'left'; None leaves it unspecified.
      name: Optional display name (shown above the player and in messages).
    """
    player = Player(x, y, role, lazy, controllable)
    _SetPlayerAttributes(player, stats, height, preferred_foot, name)
    if self._active_team == Team.e_Left:
      self._scenario_cfg.left_team.append(player)
    else:
      self._scenario_cfg.right_team.append(player)

  def SetBallPosition(self, ball_x, ball_y):
    self._scenario_cfg.ball_position[0] = ball_x
    self._scenario_cfg.ball_position[1] = ball_y

  def EpisodeNumber(self):
    return self._config['episode_number']

  def ScenarioConfig(self):
    return self._scenario_cfg
