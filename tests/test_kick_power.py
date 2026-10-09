"""Phase 0: agent-controlled kick power and pass assists."""

import gym
import numpy as np
import pytest

from gfootball.env import wrappers
from grf_tools.measure import POWER_RANGES, power_sweep
from grf_tools.trace import make_env


@pytest.mark.parametrize("kick", sorted(POWER_RANGES))
def test_power_sets_the_distance(kick):
    rows = power_sweep(kick, (0.0, 0.25, 0.5, 0.75, 1.0))
    achieved = [a for _, _, a in rows]
    assert achieved == sorted(achieved), "more power must go further"
    for power, intended, got in rows:
        # Within 10% (or 1 m for the shortest taps) of the intended distance.
        assert abs(got - intended) <= max(0.1 * intended, 1.0), (kick, power, intended, got)


def test_full_assist_passes_to_the_teammate_whatever_the_power():
    # Teammate ~40 m straight ahead (drill_pass_long). With full assists the
    # engine picks the distance, as upstream; with none, power decides.
    assisted = {"realistic_ball_physics": True, "kick_power_control": True,
                "pass_direction_assist": 1.0, "pass_power_assist": 1.0}
    rows = power_sweep("high_pass", (0.1, 1.0), assisted, scenario="drill_pass_long")
    assert abs(rows[0][2] - rows[1][2]) < 4.0
    manual = {**assisted, "pass_power_assist": 0.0}
    rows = power_sweep("high_pass", (0.1, 1.0), manual, scenario="drill_pass_long")
    assert rows[1][2] - rows[0][2] > 20.0


def test_wrappers_build_engine_actions():
    env = make_env("academy_empty_goal_close", 1, {"kick_power_control": True})
    try:
        continuous = wrappers.KickPowerWrapper(env)
        assert isinstance(continuous.action_space, gym.spaces.Tuple)
        assert continuous._to_engine((np.array([11]), np.array([0.4]))) == (11, pytest.approx(0.4))
        binned = wrappers.KickPowerWrapper(env, bins=21)
        assert list(binned.action_space.nvec) == [19, 21]
        assert binned._to_engine(np.array([12, 20])) == (12, 1.0)
        binned.reset()
        binned.step(np.array([11, 10]))  # runs end to end
    finally:
        env.close()


def test_switch_off_ignores_power():
    # Without kick_power_control a power is accepted but has no effect.
    from grf_tools.trace import first_divergence, run_trace
    plain = run_trace("academy_pass_and_shoot_with_keeper", seed=4, max_steps=200)
    env = make_env("academy_pass_and_shoot_with_keeper", 4)
    rng = np.random.default_rng(4)
    from grf_tools.trace import digest_obs
    try:
        digests = [digest_obs(env.reset()[0])]
        for _ in range(plain.steps):
            obs, _, done, _ = env.step((int(rng.integers(19)), 0.9))
            digests.append(digest_obs(obs[0]))
            if done:
                break
    finally:
        env.close()
    assert first_divergence(plain.digests, np.array(digests, dtype=np.uint64)) is None
