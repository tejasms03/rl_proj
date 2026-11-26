"""Engine settings shared by everything in outfield/ (scripts now, the env
wrapper and training later), so scripted games and training games play the
same game. Decisions behind each switch: docs/CONTEXT.md §3.
"""

TRAINING_CONFIG = {
    "action_set": "v4",                # v3 + builtin_ai, pressure, team_pressure, keeper_rush (29)
    "physics_steps_per_frame": 10,     # one step = 0.1 s (the game's default)
    "realistic_player_physics": True,  # engine v1.0 patches
    "realistic_ball_physics": True,
    "ball_pace": "game",
    "situational_accuracy": True,
    "kick_power_control": True,        # the agent sets kick power in [0, 1]
    "use_magnet": False,               # players go exactly where they are sent
    "keeper_auto_save": True,          # a keeper we control still saves
    "skill_stats": True,               # passing, finishing, heading, tackling stats matter (2026-11-24)
}

STEP_S = 0.1  # game seconds per env step at 10 ticks

# Pitch units -> metres (x, y). x in [-1, 1], y in [-0.42, 0.42], y grows
# down the screen. Same factors as grf_tools.duty_lab.duties.
METRES_X = 54.4
METRES_Y = 83.6
