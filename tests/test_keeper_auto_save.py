"""keeper_auto_save (2026-11-23): an agent-controlled keeper saves on his own.

Upstream, the dive/catch logic only runs for AI-controlled keepers, so a keeper
under agent control concedes every shot. The scene: the right team's keeper
alone in goal, shots launched at the goal from 20 m (launch_ball), our agent on
the keeper sending idle. Movement must stay with the agent when no shot comes.
"""

import concurrent.futures
import math
import multiprocessing

SHOTS = [(-2.8, 25.0), (2.8, 25.0), (-1.6, 22.0), (1.6, 22.0), (0.0, 20.0)]


def _register_scene():
    import sys
    import types
    from gfootball.scenarios import Team, e_PlayerRole_CF, e_PlayerRole_GK

    def build_scenario(builder):
        config = builder.config()
        config.game_duration = 80
        config.deterministic = False
        config.offsides = False
        config.end_episode_on_score = True
        config.end_episode_on_out_of_play = True
        config.end_episode_on_possession_change = True
        builder.SetBallPosition(0.64, 0.0)
        builder.SetTeam(Team.e_Left)
        builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=False)
        builder.AddPlayer(-0.2, 0.40, e_PlayerRole_CF, lazy=True)
        builder.SetTeam(Team.e_Right)
        builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)

    module = types.ModuleType("gfootball.scenarios.test_keeper_shot")
    module.build_scenario = build_scenario
    sys.modules[module.__name__] = module


def _env(auto_save):
    import warnings
    warnings.filterwarnings("ignore")
    import gfootball.env as football_env
    _register_scene()
    return football_env.create_environment(
        "test_keeper_shot", representation="raw",
        number_of_left_players_agent_controls=1, number_of_right_players_agent_controls=1,
        other_config_options={"action_set": "v4", "keeper_auto_save": auto_save,
                              "use_magnet": False, "realistic_player_physics": True,
                              "realistic_ball_physics": True, "ball_pace": "game"})


def _goals(auto_save):
    from gfootball.env import football_action_set
    names = [a._name for a in football_action_set.get_action_set({"action_set": "v4"})]
    idle = names.index("idle")
    env = _env(auto_save)
    goals = 0
    try:
        for target_y, speed in SHOTS:
            env.reset()
            dx, dy = 20.0, target_y
            d = math.hypot(dx, dy)
            env.unwrapped._env._env.launch_ball(35.0, 0.0, 0.11, speed * dx / d, speed * dy / d,
                                               4.0, 0.0, 0.0, 0.0)
            done = False
            while not done:
                _, _, done, info = env.step([idle, idle])
                goals += info["score_reward"] > 0
    finally:
        env.close()
    return goals


def _moves(auto_save):
    from gfootball.env import football_action_set
    names = [a._name for a in football_action_set.get_action_set({"action_set": "v4"})]
    env = _env(auto_save)
    try:
        obs = env.reset()
        y0 = obs[1]["left_team"][0][1]
        for _ in range(20):
            obs, _, _, _ = env.step([names.index("idle"), names.index("top")])
        return (y0 - obs[1]["left_team"][0][1]) * 83.6
    finally:
        env.close()


def _in_own_process(fn, *args):
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        return pool.submit(fn, *args).result()


def test_agent_keeper_concedes_without_the_switch():
    assert _in_own_process(_goals, False) >= 4


def test_agent_keeper_saves_with_the_switch():
    assert _in_own_process(_goals, True) <= 1


def test_switch_leaves_movement_to_the_agent():
    off, on = _in_own_process(_moves, False), _in_own_process(_moves, True)
    assert off > 6.0 and abs(on - off) < 1.5
