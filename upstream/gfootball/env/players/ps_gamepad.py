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
PS_AXES = {'left_x': 0, 'left_y': 1, 'right_x': 2, 'right_y': 3, 'l2': 4, 'r2': 5}

# Kicks: re-pressed every step while held, so the engine's power gauge
# charges (it releases kick buttons after each step). Released = kick.
# (button, modifier held) -> action, FIFA style; checked in this order.
KICKS = [
    ('circle', 'l1', A.action_chip),        # L1 + O: chip
    ('circle', 'r1', A.action_curl_shot),   # R1 + O: finesse / curl shot
    ('circle', None, A.action_shot),
    ('square', 'r1', A.action_long_pass),   # R1 + []: driven long ball
    ('square', None, A.action_high_pass),
    ('cross', None, A.action_short_pass),
    ('triangle', None, A.action_through_ball),
]
# Without the ball (FIFA layout). GRF has no standing-tackle button: a
# defender near the ball tackles automatically, and holding "pressure" makes
# him chase the ball and tackle by any means, so that is the tackle button.
DEFENDING = {
    'circle': A.action_pressure,       # hold: chase and tackle
    'square': A.action_sliding,        # slide tackle
    'cross': A.action_team_pressure,   # hold: a teammate presses too
    'triangle': A.action_keeper_rush,
}
TOGGLES = {'l1': A.action_switch}
# Three states decide what the face buttons do (see _state):
#   ATTACK    we own the ball, or it is loose          -> kicks
#   CONTESTED opponents touch it, but it is a duel at   -> kicks (shoot/pass
#             your feet (you are about as close as the     through the
#             nearest opponent)                            challenge)
#   DEFEND    otherwise                                 -> DEFENDING buttons
ATTACK, CONTESTED, DEFEND = 'attack', 'contested', 'defend'
CONTEST_RADIUS_M = 2.0  # ball within this distance of you
CONTEST_MARGIN_M = 0.5  # and you no further than nearest opponent + this
_METRES = (54.4, -83.6)  # observation units -> metres (x, y)
TRIGGERS = {'r2': A.action_sprint, 'l2': A.action_dribble}


class Player(gamepad.Player):
  """FIFA-style PlayStation controller. Each step: the exact stick direction
  plus one button action (held kick first, then other button changes)."""

  def __init__(self, player_config, env_config):
    # GRF numbers players per type name; the base class reads 'player_gamepad'.
    if 'player_gamepad' not in player_config:
      player_config['player_gamepad'] = player_config['player_ps_gamepad']
    gamepad.Player.__init__(self, player_config, env_config)
    self._held = {}  # action -> currently held (as last sent)
    self._tapped = set()  # buttons pressed since the last step (even if released)
    self._flicked = False  # right stick is out (one switch per flick)

  def _button(self, name):
    """Held now, or pressed at any moment since the last step (quick taps)."""
    index = PS_BUTTONS[name]
    if index in self._tapped:
      return True
    return index < self._joystick.get_numbuttons() and self._joystick.get_button(index)

  def _axis(self, name):
    index = PS_AXES[name]
    return self._joystick.get_axis(index) if index < self._joystick.get_numaxes() else -1.0

  def _state(self, observation):
    # Ownership blinks to "nobody" between dribble touches and for incoming
    # balls (the engine queues a kick until the player reaches the ball), and
    # to the opponents whenever a defender pokes at it in a duel.
    if observation['ball_owned_team'] != 1:
      return ATTACK
    def dist(position):
      return (((position[0] - ball[0]) * _METRES[0]) ** 2 +
              ((position[1] - ball[1]) * _METRES[1]) ** 2) ** 0.5
    ball = observation['ball']
    mine = dist(observation['left_team'][observation['active']])
    theirs = min(dist(p) for p in observation['right_team'])
    if mine < CONTEST_RADIUS_M and mine <= theirs + CONTEST_MARGIN_M:
      return CONTESTED
    return DEFEND

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
    # Remember every press since the last step, so taps shorter than a step
    # still count; the current state of each button is read below.
    self._tapped = {event.button for event in event_queue.get('gamepad', self._index)
                    if event.type == pygame.JOYBUTTONDOWN}
    self.state = self._state(observation)
    on_ball = (observation['ball_owned_team'] == 0 and
               observation['ball_owned_player'] == observation['active'])
    action = self._button_action(self.state != DEFEND, on_ball)
    x, y = self._axis('left_x'), self._axis('left_y')
    x += float(bool(self._button('right'))) - float(bool(self._button('left')))
    y += float(bool(self._button('down'))) - float(bool(self._button('up')))
    length = (x * x + y * y) ** 0.5
    # Exact stick angle (or D-pad), sent every step alongside the button
    # action so you can aim while charging. Dead zone 0.3; full speed at any
    # tilt (sprint is R2). Stick y points down; the engine's +y is up.
    dx, dy = (x / length, -y / length) if length > 0.3 else (0.0, 0.0)
    # Right stick flick: switch to the teammate that way (FIFA). One switch
    # per flick; not while you have the ball at your feet.
    sx, sy = 0.0, 0.0
    rx, ry = self._axis('right_x'), self._axis('right_y')
    reach = (rx * rx + ry * ry) ** 0.5
    if reach > 0.7 and not self._flicked:
      self._flicked = True
      on_ball = (observation['ball_owned_team'] == 0 and
                 observation['ball_owned_player'] == observation['active'])
      if not on_ball:
        sx, sy = rx / reach, -ry / reach
    elif reach < 0.4:
      self._flicked = False
    return (action, -1.0, dx, dy, sx, sy)

  def _button_action(self, with_ball, on_ball):
    # 1. Kicks (with the ball): keep pressing while held to charge power.
    if with_ball:
      for name, modifier, action in KICKS:
        if (self._button(name) and (modifier is None or self._button(modifier)) and
            action.is_in_actionset(self._env_config)):
          return action

    # 2. Defensive buttons; off the ball release any still held (else a
    #    pressure held while winning the ball stays on during the attack).
    for name, action in DEFENDING.items():
      result = self._toggle(action, self._button(name) and not with_ball)
      if result:
        return result
    # L1 switches whenever the ball is not at your feet (loose balls too);
    # on the ball it is the chip modifier (L1 + O).
    if not on_ball:
      for name, action in TOGGLES.items():
        result = self._toggle(action, self._button(name))
        if result:
          return result
    else:
      for name, action in TOGGLES.items():
        result = self._toggle(action, False)
        if result:
          return result
    for name, action in TRIGGERS.items():
      result = self._toggle(action, self._axis(name) > 0.0)
      if result:
        return result
    return A.action_idle
