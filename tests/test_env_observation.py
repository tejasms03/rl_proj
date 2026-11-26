"""outfield.env observation: the checks listed in docs/observation_spec.md."""

import numpy as np
import pytest

from outfield.env import OutfieldEnv, random_actions
from outfield.env import observation as ob

RELATIVE = ("rel", "dist", "ball_dist", "opp_goal_rel", "opp_goal_dist", "own_goal_rel")


def _names():
    """Per-column block name of the 869 base numbers."""
    names = []
    for blk, sub in (("self", ob.SELF), ("ball", ob.BALL)):
        for n, s in sub.items():
            names += [n] * (s.stop - s.start)
    for _ in range(10):
        for n, s in ob.MATE.items():
            names += [n] * (s.stop - s.start)
    for _ in range(11):
        for n, s in ob.OPP.items():
            names += [n] * (s.stop - s.start)
    for n, s in ob.CONTEXT.items():
        names += [n] * (s.stop - s.start)
    return np.array(names)


@pytest.fixture(scope="module")
def match():
    env = OutfieldEnv("11_vs_11_easy_stochastic", seed=3)
    rng = np.random.default_rng(3)
    frames = [env.reset()]
    raws = [env.raw_list[0]]
    for _ in range(300):
        obs, *_ = env.step(random_actions(frames[-1]["masks"], rng))
        frames.append(obs)
        raws.append(env.raw_list[0])
    env.close()
    return frames, raws


def test_shapes_and_finite(match):
    frames, _ = match
    for obs in frames:
        assert obs["actor"].shape == (11, 960) and obs["actor"].dtype == np.float32
        assert obs["state"].ndim == 1 and np.isfinite(obs["state"]).all()
        assert np.isfinite(obs["actor"]).all()
    assert ob.ACTOR_SIZE == 869


def test_ranges(match):
    # Relative positions and distances span the whole pitch (up to 108 m / 55 ≈ 2),
    # so they get ±2.1; everything else stays within the spec's ±1.5.
    frames, _ = match
    names = _names()
    peak = np.max([np.abs(o["actor"][:, :869]).max(axis=0) for o in frames], axis=0)
    relative = np.isin(names, RELATIVE)
    assert peak[relative].max() <= 2.1
    assert peak[~relative].max() <= 1.5, names[~relative][peak[~relative] > 1.5]


def test_self_rows_match_raw(match):
    frames, raws = match
    obs, raw = frames[-1], raws[-1]
    pos = np.asarray(raw["left_team"])[:, :2] * ob.METRES / ob.POS_SCALE
    np.testing.assert_allclose(obs["actor"][:, ob.SELF["pos"]], pos, atol=1e-5)
    np.testing.assert_array_equal(obs["actor"][:, ob.SELF["active"]][:, 0], raw["left_team_active"])
    roles = obs["actor"][:, ob.SELF["role"]].argmax(axis=1)
    np.testing.assert_array_equal(roles, raw["left_team_roles"])
    # Opponent goal straight ahead of a player on the centre line is (+, 0).
    rel = obs["actor"][:, ob.ACTOR_SIZE - ob.CONTEXT_SIZE:][:, ob.CONTEXT["opp_goal_rel"]]
    expected = (ob.OPP_GOAL - np.asarray(raw["left_team"])[:, :2] * ob.METRES) / ob.POS_SCALE
    np.testing.assert_allclose(rel, expected, atol=1e-5)


def test_has_ball_agrees_everywhere(match):
    frames, raws = match
    checked = 0
    for obs, raw in zip(frames, raws):
        if raw["ball_owned_team"] != 0:
            continue
        j = int(raw["ball_owned_player"])
        a = obs["actor"]
        assert a[j, ob.SELF["has_ball"]][0] == 1
        assert a[j, ob.SELF_SIZE + ob.BALL["mine"].start] == 1
        assert a[:, ob.SELF["has_ball"]].sum() == 1
        for i in range(11):
            if i == j:
                continue
            row = j if j < i else j - 1  # teammate rows skip self
            start = ob.ACTOR_LAYOUT["teammates"].start + row * ob.MATE_SIZE
            assert a[i, start + ob.MATE["has_ball"].start] == 1
        checked += 1
    assert checked > 0


def test_padding_in_a_small_drill():
    env = OutfieldEnv("academy_3_vs_1_with_keeper", seed=1)
    try:
        obs = env.reset()
    finally:
        env.close()
    a = obs["actor"]
    assert obs["present"].tolist() == [True] * 4 + [False] * 7
    assert not a[4:].any(), "absent players' rows are zero"
    mates = a[0, ob.ACTOR_LAYOUT["teammates"]].reshape(10, ob.MATE_SIZE)
    assert mates[:3].any(axis=1).all() and not mates[3:].any()
    opps = a[0, ob.ACTOR_LAYOUT["opponents"]].reshape(11, ob.OPP_SIZE)
    assert opps[:2].any(axis=1).all() and not opps[2:].any()
    for name, m in obs["masks"].items():
        assert m[4:, 0].all() and not m[4:, 1:].any(), name
