"""T2: same seed + same actions => identical game. T7: save/restore is exact."""

import numpy as np
import pytest

from grf_tools.trace import digest_obs, first_divergence, make_env, run_trace

# Extended with every patch switch turned on as patches land.
CONFIGS = {
    "upstream": {},
    "p1_attributes": {
        "player_overrides": {
            "left": {"*": {"stats": {"physical_velocity": 0.9, "technical_shot": 0.8},
                           "height": 1.9, "preferred_foot": "left"}},
            "right": {"*": {"stats": {"mental_defensivepositioning": 0.2}}},
        }
    },
    "p1b_physics": {"realistic_player_physics": True},
    "p2_ball": {"realistic_ball_physics": True, "pitch_condition": "wet"},
}


@pytest.mark.parametrize("config_name", CONFIGS)
@pytest.mark.parametrize(
    ("scenario", "max_steps"),
    [("11_vs_11_stochastic", 1500), ("academy_run_pass_and_shoot_with_keeper", 400)],
)
def test_same_seed_same_game(config_name, scenario, max_steps):
    config = CONFIGS[config_name]
    a = run_trace(scenario, seed=11, max_steps=max_steps, config=config)
    b = run_trace(scenario, seed=11, max_steps=max_steps, config=config)
    assert first_divergence(a.digests, b.digests) is None


def test_different_seed_different_game():
    a = run_trace("11_vs_11_stochastic", seed=11, max_steps=300)
    b = run_trace("11_vs_11_stochastic", seed=12, max_steps=300)
    assert first_divergence(a.digests, b.digests) is not None


@pytest.mark.parametrize("config_name", CONFIGS)
def test_save_restore_continues_identically(config_name):
    env = make_env("11_vs_11_stochastic", seed=21, config=CONFIGS[config_name])
    rng = np.random.default_rng(21)
    try:
        env.reset()
        for _ in range(400):
            env.step(int(rng.integers(19)))
        state = env.unwrapped.get_state()
        actions = rng.integers(19, size=300)

        def continue_from_here():
            return [digest_obs(env.step(int(a))[0][0]) for a in actions]

        first = continue_from_here()
        env.unwrapped.set_state(state)
        second = continue_from_here()
    finally:
        env.close()
    assert first == second


def test_mixing_tick_rates_in_one_process_is_refused():
    run_trace("academy_empty_goal_close", seed=1, max_steps=2)
    with pytest.raises(RuntimeError, match="physics_steps_per_frame"):
        make_env("academy_empty_goal_close", seed=1, config={"physics_steps_per_frame": 1})


def test_patch_configs_actually_change_the_game():
    # Otherwise the determinism checks above would only re-test upstream.
    base = run_trace("11_vs_11_stochastic", seed=11, max_steps=600)
    for name, config in CONFIGS.items():
        if name == "upstream":
            continue
        patched = run_trace("11_vs_11_stochastic", seed=11, max_steps=600, config=config)
        assert first_divergence(base.digests, patched.digests) is not None, name
