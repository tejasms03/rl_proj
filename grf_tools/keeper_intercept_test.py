import time
import gfootball.env as football_env
from gfootball.env import football_action_set
from grf_tools.keeper import keeper_reward

V1 = {"realistic_player_physics": True, "realistic_ball_physics": True,
      "ball_pace": "game", "situational_accuracy": True}

env = football_env.create_environment(
    "drill_keeper_intercept",   # an opponent stands in the passing lane
    representation="raw", render=True,
    number_of_left_players_agent_controls=1,
    other_config_options={"action_set": "full", "real_time": True, **V1})
names = [a._name for a in football_action_set.get_action_set({"action_set": "full"})]

for pass_type in ["short_pass"]:
    print("Testing", pass_type)
    env.reset()
    for _ in range(5):
        env.step(names.index("right"))         # blank 2: walk onto the ball
    for _ in range(3):
        env.step(names.index("bottom_right"))         # blank 3: face the teammate
    obs, _, _, _ = env.step(names.index(pass_type))       # press the pass, keep the photo
    memory = {"pass_in_air": False}                       # fresh note for this pass
    done = False
    step = 0
    while not done:
        step += 1
        prev_obs = obs                                     # save the photo before the step
        obs, _, done, _ = env.step(names.index("idle"))   # the game moves on
        reward = keeper_reward(prev_obs[0], obs[0], False, memory)
        print(step, reward)
    time.sleep(2)

env.close()