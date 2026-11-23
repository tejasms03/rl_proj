import time
import gfootball.env as football_env
from gfootball.env import football_action_set

V1 = {"realistic_player_physics": True, "realistic_ball_physics": True,
      "ball_pace": "game", "situational_accuracy": True}

env = football_env.create_environment(
    "drill_keeper_pass",                                   # blank 1: your drill's name
    representation="raw", render=True,
    number_of_left_players_agent_controls=1,
    other_config_options={"action_set": "full", "real_time": True, **V1})
names = [a._name for a in football_action_set.get_action_set({"action_set": "full"})]

for pass_type in ["short_pass", "high_pass", "long_pass"]:
    print("Testing", pass_type)
    env.reset()
    for _ in range(5):
        env.step(names.index("right"))         # blank 2: walk onto the ball
    for _ in range(3):
        env.step(names.index("bottom_right"))         # blank 3: face the teammate
    env.step(names.index(pass_type))               # blank 4: press the pass
    done = False
    while not done:
        _, _, done, _ = env.step(names.index("idle"))
    time.sleep(2)

env.close()