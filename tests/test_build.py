"""T0: the engine imports, builds every scenario, and steps."""

import os

import gfootball
import gfootball.env as football_env
import numpy as np
import pytest

SCENARIO_DIR = os.path.join(os.path.dirname(gfootball.__file__), "scenarios")
SCENARIOS = sorted(
    f[:-3] for f in os.listdir(SCENARIO_DIR) if f.endswith(".py") and f != "__init__.py"
)


def test_scenarios_found():
    assert len(SCENARIOS) == 19
    assert "11_vs_11_stochastic" in SCENARIOS


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_resets_and_steps(scenario):
    env = football_env.create_environment(scenario, representation="simple115v2", render=False)
    try:
        obs = env.reset()
        assert obs.shape == (115,)
        for _ in range(20):
            obs, reward, done, _ = env.step(env.action_space.sample())
            assert np.isfinite(obs).all()
            assert np.isfinite(reward)
            if done:
                env.reset()
    finally:
        env.close()


def test_random_play_1000_steps():
    env = football_env.create_environment(
        "11_vs_11_easy_stochastic", representation="extracted", render=False
    )
    try:
        assert env.reset().shape == (72, 96, 4)
        assert env.action_space.n == 19
        for _ in range(1000):
            _, _, done, _ = env.step(env.action_space.sample())
            if done:
                env.reset()
    finally:
        env.close()


def test_multi_agent_shapes():
    env = football_env.create_environment(
        "academy_3_vs_1_with_keeper",
        representation="simple115v2",
        number_of_left_players_agent_controls=3,
        render=False,
    )
    try:
        assert env.reset().shape == (3, 115)
        obs, reward, _, _ = env.step([0, 0, 0])
        assert obs.shape == (3, 115)
        assert np.shape(reward) == (3,)
    finally:
        env.close()
