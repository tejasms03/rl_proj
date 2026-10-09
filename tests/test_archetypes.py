"""Phase 0: player archetypes and mid-game stat swaps (for the manager agent)."""

import numpy as np
import pytest

from gfootball.env import archetypes, scenario_builder
from gfootball.env.football_env_core import PLAYER_ATTRIBUTE_NAMES
from grf_tools.sprint import _ACTIONS, X_SCALE
from grf_tools.trace import make_env

COL = {name: i for i, name in enumerate(PLAYER_ATTRIBUTE_NAMES)}


def all_archetypes():
    return [(p, n) for p, d in archetypes.ARCHETYPES.items() for n in d]


@pytest.mark.parametrize(("position", "name"), all_archetypes())
def test_archetypes_share_one_budget(position, name):
    a = archetypes.archetype(position, name)
    assert set(a["stats"]) == set(scenario_builder.PLAYER_STATS)
    assert np.mean(list(a["stats"].values())) == pytest.approx(archetypes.BUDGET, abs=0.005)
    assert all(0.05 <= v <= 1.0 for v in a["stats"].values())
    assert 1.6 <= a["height"] <= 2.0


def test_no_archetype_dominates_its_position():
    for position, group in archetypes.ARCHETYPES.items():
        for a, b in [(x, y) for x in group for y in group if x != y]:
            sa, sb = group[a]["stats"], group[b]["stats"]
            assert any(sa[k] > sb[k] + 0.05 for k in sa), (position, a, "never better than", b)


def test_archetype_applies_at_kickoff():
    target = archetypes.archetype("ST", "target_man")
    env = make_env("11_vs_11_stochastic", 1, {"player_overrides": {"left": {9: target}}})
    try:
        left = env.reset()[0]["left_team_attributes"]
    finally:
        env.close()
    assert left[9, COL["technical_header"]] == pytest.approx(target["stats"]["technical_header"], abs=1e-3)
    assert left[9, COL["height"]] == pytest.approx(target["height"])


def test_mid_game_swap_changes_stats_and_speed():
    env = make_env("drill_sprint", 0, {"realistic_player_physics": True, "player_overrides": {
        "left": {1: {"stats": {"physical_velocity": 0.0, "physical_acceleration": 0.5}}}}})
    try:
        obs = env.reset()[0]
        env.step(_ACTIONS["sprint"])
        for _ in range(40):  # 4 s: reach top speed
            obs = env.step(_ACTIONS["right"])[0][0]
        x0 = obs["left_team"][1][0]
        obs = env.step(_ACTIONS["right"])[0][0]
        slow = (obs["left_team"][1][0] - x0) * X_SCALE / 0.1

        env.unwrapped.set_player_stats("left", 1, {"physical_velocity": 1.0})
        for _ in range(30):
            obs = env.step(_ACTIONS["right"])[0][0]
        assert obs["left_team_attributes"][1, COL["physical_velocity"]] == pytest.approx(1.0)
        x0 = obs["left_team"][1][0]
        obs = env.step(_ACTIONS["right"])[0][0]
        fast = (obs["left_team"][1][0] - x0) * X_SCALE / 0.1
    finally:
        env.close()
    assert slow == pytest.approx(7.8, abs=0.3)
    assert fast == pytest.approx(9.8, abs=0.3)


def test_apply_archetype_and_bad_player_index():
    env = make_env("11_vs_11_stochastic", 1)
    try:
        env.reset()
        env.unwrapped.apply_archetype("right", 5, "CM", "playmaker")
        right = env.step(0)[0][0]["right_team_attributes"]
        expected = archetypes.archetype("CM", "playmaker")["stats"]["technical_shortpass"]
        assert right[5, COL["technical_shortpass"]] == pytest.approx(expected, abs=1e-3)
        with pytest.raises(ValueError, match="No player"):
            env.unwrapped.set_player_stats("left", 30, {"physical_velocity": 0.5})
    finally:
        env.close()
