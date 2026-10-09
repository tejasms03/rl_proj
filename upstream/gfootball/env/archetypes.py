# rl_project phase 0: player archetypes for the manager agent.
#
# Each archetype is a set of the 22 engine stats (0-1) plus height, inspired by
# real player roles. Every archetype has the same stat budget (mean 0.70), so
# none is strictly better than another of its position: picking one means
# trading, e.g., pace for strength. Not part of upstream GRF.

from __future__ import annotations

STAT_NAMES = (
    'physical_balance', 'physical_reaction', 'physical_acceleration',
    'physical_velocity', 'physical_stamina', 'physical_agility',
    'physical_shotpower', 'technical_standingtackle', 'technical_slidingtackle',
    'technical_ballcontrol', 'technical_dribble', 'technical_shortpass',
    'technical_highpass', 'technical_header', 'technical_shot',
    'technical_volley', 'mental_calmness', 'mental_workrate',
    'mental_resilience', 'mental_defensivepositioning',
    'mental_offensivepositioning', 'mental_vision')

BUDGET = 0.70  # mean stat of every archetype

# Groups of stats, used to describe archetypes compactly.
_GROUPS = {
    'pace': ('physical_acceleration', 'physical_velocity', 'physical_agility'),
    'strength': ('physical_balance', 'physical_shotpower'),
    'engine': ('physical_stamina', 'mental_workrate', 'mental_resilience'),
    'reactions': ('physical_reaction', 'mental_calmness'),
    'tackling': ('technical_standingtackle', 'technical_slidingtackle'),
    'aerial': ('technical_header',),
    'control': ('technical_ballcontrol', 'technical_dribble'),
    'passing': ('technical_shortpass', 'technical_highpass', 'mental_vision'),
    'finishing': ('technical_shot', 'technical_volley'),
    'defending': ('mental_defensivepositioning',),
    'attacking': ('mental_offensivepositioning',),
}

# position -> archetype -> (height m, {group: emphasis}). Emphasis is
# relative (+ stronger, - weaker); stats are then rescaled to the budget.
_SPECS = {
    'GK': {
        'shot_stopper': (1.93, {'reactions': 3, 'aerial': 2, 'strength': 1, 'pace': -1, 'passing': -1}),
        'sweeper_keeper': (1.88, {'reactions': 2, 'pace': 1, 'passing': 2, 'aerial': -1}),
    },
    'CB': {
        'stopper': (1.88, {'tackling': 3, 'strength': 2, 'defending': 2, 'passing': -2, 'control': -1}),
        'ball_playing_cb': (1.86, {'passing': 3, 'control': 1, 'defending': 1, 'strength': -1, 'aerial': -1}),
        'aerial_cb': (1.95, {'aerial': 3, 'strength': 2, 'tackling': 1, 'pace': -2, 'control': -1}),
    },
    'FB': {
        'attacking_fullback': (1.76, {'pace': 2, 'engine': 2, 'passing': 1, 'attacking': 1, 'defending': -1, 'aerial': -1}),
        'defensive_fullback': (1.81, {'tackling': 2, 'defending': 2, 'strength': 1, 'attacking': -2}),
    },
    'DM': {
        'ball_winner': (1.82, {'tackling': 3, 'engine': 2, 'defending': 2, 'finishing': -2, 'attacking': -1}),
        'deep_playmaker': (1.78, {'passing': 3, 'reactions': 2, 'control': 1, 'tackling': -1, 'pace': -1}),
    },
    'CM': {
        'box_to_box': (1.83, {'engine': 3, 'pace': 1, 'tackling': 1, 'finishing': 1, 'passing': -1}),
        'playmaker': (1.74, {'passing': 3, 'control': 2, 'reactions': 1, 'strength': -2, 'tackling': -1}),
        'mezzala': (1.78, {'control': 2, 'attacking': 2, 'finishing': 1, 'passing': 1, 'defending': -2}),
    },
    'W': {
        'speedster': (1.75, {'pace': 3, 'control': 1, 'attacking': 1, 'aerial': -2, 'strength': -1}),
        'inverted_winger': (1.76, {'control': 2, 'finishing': 2, 'attacking': 1, 'defending': -2}),
        'crossing_winger': (1.78, {'passing': 2, 'pace': 1, 'engine': 1, 'finishing': -1}),
    },
    'ST': {
        'poacher': (1.80, {'finishing': 3, 'attacking': 3, 'reactions': 1, 'engine': -2, 'passing': -1}),
        'target_man': (1.93, {'aerial': 3, 'strength': 3, 'finishing': 1, 'pace': -2, 'control': -1}),
        'false_nine': (1.77, {'passing': 2, 'control': 2, 'attacking': 1, 'aerial': -2, 'strength': -1}),
        'pressing_forward': (1.81, {'engine': 3, 'pace': 1, 'tackling': 1, 'finishing': 1, 'reactions': -1}),
    },
}

# Engine roles (e_PlayerRole name suffix) -> archetype position group.
ROLE_POSITIONS = {
    'GK': 'GK', 'CB': 'CB', 'LB': 'FB', 'RB': 'FB', 'DM': 'DM', 'CM': 'CM',
    'LM': 'W', 'RM': 'W', 'AM': 'CM', 'CF': 'ST',
}


def _build(emphasis: dict[str, float]) -> dict[str, float]:
  raw = {name: 1.0 for name in STAT_NAMES}
  for group, weight in emphasis.items():
    for name in _GROUPS[group]:
      raw[name] += 0.15 * weight
  # Rescale so the mean is exactly BUDGET, keeping values inside [0.05, 1].
  for _ in range(20):
    mean = sum(raw.values()) / len(raw)
    raw = {k: min(1.0, max(0.05, v * BUDGET / mean)) for k, v in raw.items()}
  return {k: round(v, 3) for k, v in raw.items()}


ARCHETYPES: dict[str, dict[str, dict]] = {
    position: {
        name: {'stats': _build(emphasis), 'height': height}
        for name, (height, emphasis) in specs.items()
    }
    for position, specs in _SPECS.items()
}


def archetype(position: str, name: str) -> dict:
  """{'stats': {...}, 'height': m} for use in AddPlayer or player_overrides."""
  return ARCHETYPES[position][name]
