# rl_project: PlayStation-layout gamepad player with FIFA-style buttons and
# hold-to-charge kicks. Not part of upstream GRF.
#
# Use as player type "ps_gamepad", e.g. players=["ps_gamepad:left_players=1"].
# Button numbers below are what SDL reports for DualShock 4 / DualSense
# controllers on macOS; run `python -m grf_tools.play_gamepad --test` to check
# yours and remap with PS_BUTTONS / PS_AXES if needed.

from gfootball.env import event_queue
from gfootball.env import football_action_set as A
from gfootball.env.players import gamepad
import pygame

PS_BUTTONS = {
    'cross': 0, 'circle': 1, 'square': 2, 'triangle': 3,
    'share': 4, 'ps': 5, 'options': 6, 'l3': 7, 'r3': 8, 'l1': 9, 'r1': 10,
    'up': 11, 'down': 12, 'left': 13, 'right': 14,
}
PS_AXES = {'left_x': 0, 'left_y': 1, 'l2': 4, 'r2': 5}

# Kicks: re-pressed every step while held, so the engine's power gauge
# charges (it releases kick buttons after each step). Released = kick.
KICKS = {
    'cross': A.action_short_pass,
    'circle': A.action_shot,
    'square': A.action_high_pass,
    'triangle': A.action_through_ball,
    'r1': A.action_curl_shot,
    'l3': A.action_chip,
    'r3': A.action_long_pass,
}
# Without the ball: pressed / released like normal buttons.
DEFENDING = {
    'cross': A.action_pressure,
    'circle': A.action_sliding,
    'square': A.action_team_pressure,
    'triangle': A.action_keeper_rush,
}
TOGGLES = {'l1': A.action_switch}
TRIGGERS = {'r2': A.action_sprint, 'l2': A.action_dribble}


class Player(gamepad.Player):
  """FIFA-style PlayStation controller; one action per step, in priority:
  held kick, direction change, other button changes."""

  def __init__(self, player_config, env_config):
    # GRF numbers players per type name; the base class reads 'player_gamepad'.
    if 'player_gamepad' not in player_config:
      player_config['player_gamepad'] = player_config['player_ps_gamepad']
    gamepad.Player.__init__(self, player_config, env_config)
    self._held = {}  # action -> currently held (as last sent)
    self._direction = A.action_idle

  def _button(self, name):
    index = PS_BUTTONS[name]
    return index < self._joystick.get_numbuttons() and self._joystick.get_button(index)

  def _axis(self, name):
    index = PS_AXES[name]
    return self._joystick.get_axis(index) if index < self._joystick.get_numaxes() else -1.0

  def _has_ball(self, observation):
    return (observation['ball_owned_team'] == 0 and
            observation['ball_owned_player'] == observation['active'])

  def _toggle(self, action, pressed):
    if not action.is_in_actionset(self._env_config):
      return None
    if bool(self._held.get(action)) != bool(pressed):
      self._held[action] = pressed
      return action if pressed else A.disable_action(action)
    return None

  def take_action(self, observations):
    assert len(observations) == 1, 'Gamepad controls one player'
    observation = observations[0]
    for _ in event_queue.get('gamepad', self._index):
      pass  # drain events; we read the current state below
    with_ball = self._has_ball(observation)

    # 1. Kicks (with the ball): keep pressing while held to charge power.
    if with_ball:
      for name, action in KICKS.items():
        if self._button(name) and action.is_in_actionset(self._env_config):
          return action

    # 2. Direction from the left stick (8-way).
    x, y = self._axis('left_x'), self._axis('left_y')
    # Left stick or D-pad.
    left = x < -0.5 or bool(self._button('left'))
    right = x > 0.5 or bool(self._button('right'))
    top = y < -0.5 or bool(self._button('up'))
    bottom = y > 0.5 or bool(self._button('down'))
    direction = A.action_idle
    for action, on in ((A.action_top_left, top and left), (A.action_top_right, top and right),
                       (A.action_bottom_left, bottom and left),
                       (A.action_bottom_right, bottom and right)):
      if on:
        direction = action
    if direction == A.action_idle:
      for action, on in ((A.action_right, right), (A.action_left, left),
                         (A.action_top, top), (A.action_bottom, bottom)):
        if on:
          direction = action
    if direction != self._direction:
      self._direction = direction
      return A.action_release_direction if direction == A.action_idle else direction

    # 3. Defensive buttons, switch, sprint and dribble triggers.
    if not with_ball:
      for name, action in DEFENDING.items():
        result = self._toggle(action, self._button(name))
        if result:
          return result
    for name, action in TOGGLES.items():
      result = self._toggle(action, self._button(name))
      if result:
        return result
    for name, action in TRIGGERS.items():
      result = self._toggle(action, self._axis(name) > 0.0)
      if result:
        return result
    return A.action_idle
