"""P1: per-player stats, height and preferred foot."""

import numpy as np
import pytest

import gfootball_engine as libgame
from gfootball.env import scenario_builder
from gfootball.env.football_env_core import PLAYER_ATTRIBUTE_NAMES
from grf_tools.play import ai_episode
from grf_tools.trace import make_env

COL = {name: i for i, name in enumerate(PLAYER_ATTRIBUTE_NAMES)}
FAST = {"physical_velocity": 1.0, "physical_acceleration": 1.0}
SLOW = {"physical_velocity": 0.1, "physical_acceleration": 0.1}


def attributes(scenario, overrides=None, seed=1):
    config = {"player_overrides": overrides} if overrides else {}
    env = make_env(scenario, seed, config)
    try:
        obs = env.reset()[0]
    finally:
        env.close()
    return obs["left_team_attributes"], obs["right_team_attributes"]


def test_attribute_columns():
    assert len(PLAYER_ATTRIBUTE_NAMES) == 24
    assert PLAYER_ATTRIBUTE_NAMES[-2:] == ("height", "preferred_foot")
    assert set(PLAYER_ATTRIBUTE_NAMES[:-2]) == set(scenario_builder.PLAYER_STATS)


def test_defaults_are_the_built_in_profiles():
    left, right = attributes("11_vs_11_stochastic")
    assert left.shape == right.shape == (11, 24)
    np.testing.assert_array_equal(left, right)  # both teams share the 11 profiles
    # Profile 398 (slot 0) and 264 (slot 10), from data/playerdata.cpp.
    assert left[0, COL["height"]] == pytest.approx(1.87)
    assert left[0, COL["physical_velocity"]] == pytest.approx(0.6516, abs=1e-3)
    assert left[10, COL["height"]] == pytest.approx(1.69)
    assert left[10, COL["physical_velocity"]] == pytest.approx(1.0)
    assert (left[:, COL["preferred_foot"]] == 0).all()


def test_overrides_change_only_their_target():
    default_left, default_right = attributes("11_vs_11_stochastic")
    overrides = {
        "left": {
            "*": {"stats": {"technical_shot": 0.2}},
            3: {"stats": {"physical_velocity": 0.25}, "height": 1.95, "preferred_foot": "left"},
        },
    }
    left, right = attributes("11_vs_11_stochastic", overrides)
    np.testing.assert_array_equal(right, default_right)
    np.testing.assert_allclose(left[:, COL["technical_shot"]], 0.2, rtol=1e-6)
    assert left[3, COL["physical_velocity"]] == pytest.approx(0.25)
    assert left[3, COL["height"]] == pytest.approx(1.95)
    assert left[3, COL["preferred_foot"]] == 2
    untouched = [c for n, c in COL.items() if n not in ("technical_shot",)]
    others = [i for i in range(11) if i != 3]
    np.testing.assert_array_equal(left[np.ix_(others, untouched)],
                                  default_left[np.ix_(others, untouched)])


def test_set_player_attributes_on_entry():
    entry = libgame.FormationEntry(0.0, 0.0, libgame.e_PlayerRole.e_PlayerRole_CF, False, True)
    scenario_builder._SetPlayerAttributes(
        entry, stats={"technical_header": 0.75}, height=1.82, preferred_foot="right")
    assert entry.get_stat(scenario_builder.PLAYER_STATS["technical_header"]) == pytest.approx(0.75)
    assert entry.get_stat(scenario_builder.PLAYER_STATS["technical_shot"]) == -1.0
    assert entry.height == pytest.approx(1.82)
    assert entry.preferred_foot == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"left": {1: {"stats": {"speed": 0.5}}}}, "Unknown player stat"),
        ({"left": {1: {"stats": {"physical_velocity": 1.5}}}}, "outside"),
        ({"left": {1: {"height": 2.5}}}, "height"),
        ({"left": {1: {"preferred_foot": "both"}}}, "preferred_foot"),
        ({"middle": {1: {"height": 1.8}}}, "left/right"),
        ({"left": {11: {"height": 1.8}}}, "no slot"),
    ],
)
def test_invalid_overrides_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        attributes("11_vs_11_stochastic", overrides)


def test_faster_player_wins_the_footrace():
    def left_wins(left, right, seeds=range(15)):
        overrides = {"left": {1: {"stats": left}}, "right": {1: {"stats": right}}}
        return sum(ai_episode("drill_footrace", s, {"player_overrides": overrides}).first_owner == 0
                   for s in seeds)

    assert left_wins(FAST, SLOW) == 15
    assert left_wins(SLOW, FAST) == 0
