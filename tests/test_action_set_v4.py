"""Action set v4: v3 plus builtin_ai, keeper_rush, pressure, team_pressure and
the three release actions (2026-11-23)."""

import concurrent.futures
import multiprocessing

from gfootball.env import football_action_set

V4 = {a._name: i for i, a in enumerate(football_action_set.get_action_set({"action_set": "v4"}))}


def test_v4_extends_v3_without_moving_indices():
    v3 = football_action_set.get_action_set({"action_set": "v3"})
    v4 = football_action_set.get_action_set({"action_set": "v4"})
    assert len(v4) == 29
    assert v4[:22] == v3
    assert [a._name for a in v4[22:]] == [
        "builtin_ai", "keeper_rush", "pressure", "team_pressure",
        "release_keeper_rush", "release_pressure", "release_team_pressure"]


def test_v4_sticky_actions_keep_v3_order():
    v3 = [a._name for a in football_action_set.get_sticky_actions({"action_set": "v3"})]
    v4 = [a._name for a in football_action_set.get_sticky_actions({"action_set": "v4"})]
    assert v4[:len(v3)] == v3 and len(v3) == 10
    assert v4[10:] == ["keeper_rush", "pressure", "team_pressure"]


def _press_and_release():
    import warnings
    warnings.filterwarnings("ignore")
    import gfootball.env as football_env
    env = football_env.create_environment(
        "11_vs_11_stochastic", representation="raw",
        number_of_left_players_agent_controls=1,
        other_config_options={"action_set": "v4"})
    try:
        env.reset()
        held = []
        for name in ["idle", "pressure", "idle", "release_pressure", "team_pressure",
                     "release_team_pressure", "builtin_ai"]:
            obs, _, _, _ = env.step(V4[name])
            held.append(obs[0]["sticky_actions"].tolist())
        return held
    finally:
        env.close()


def test_v4_pressure_holds_until_released():
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        held = pool.submit(_press_and_release).result()
    assert all(len(h) == 13 for h in held)
    pressure, team = 11, 12
    assert held[0][pressure] == 0
    assert held[1][pressure] == 1 and held[2][pressure] == 1  # stays held
    assert held[3][pressure] == 0
    assert held[4][team] == 1 and held[5][team] == 0
