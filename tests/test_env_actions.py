"""outfield.env actions: analog movement, sticky toggles, masks, wind-up, AI hand-off."""

import math

import numpy as np
import pytest

from outfield.env import OutfieldEnv
from outfield.env import actions as act
from outfield.env import observation as ob


def _idle():
    z = np.zeros(11, dtype=int)
    return {"move": z.copy(), "speed": z.copy(), "ball": z.copy(), "defend": z.copy(),
            "power": np.zeros(11, np.float32)}


def test_direction_round_trip():
    for k in range(17):
        assert act.direction_index(*act.move_vector(k)) == k


@pytest.fixture(scope="module")
def env():
    e = OutfieldEnv("outfield_3v3", seed=1, ai_on_ball=False)
    yield e
    e.close()


def _pos(env, i):
    return np.asarray(env.raw_list[0]["left_team"][i][:2]) * ob.METRES


@pytest.mark.parametrize("k", [1, 3, 5, 7, 9, 11, 13, 15])
def test_analog_moves_where_asked(env, k):
    # Player 3 (right wing, away from the ball) walks 1 s in direction k.
    env.reset()
    start = _pos(env, 3)
    a = _idle()
    a["move"][3] = k
    for _ in range(10):
        env.step(a)
    moved = _pos(env, 3) - start
    heading = math.degrees(math.atan2(moved[1], moved[0]))
    target = (k - 1) * 22.5
    assert np.linalg.norm(moved) > 1.5
    assert abs((heading - target + 180) % 360 - 180) <= 12


def test_sprint_and_pressure_are_toggled(env):
    env.reset()
    a = _idle()
    a["speed"][3] = 1  # sprint
    env.step(a)
    held = dict(zip(act.STICKY, env.raw_list[3]["sticky_actions"]))
    assert held["sprint"] == 1
    a["speed"][3] = 0
    env.step(a)
    held = dict(zip(act.STICKY, env.raw_list[3]["sticky_actions"]))
    assert held["sprint"] == 0


def test_masks():
    e = OutfieldEnv("outfield_3v3", seed=2)
    try:
        obs = e.reset()
        m = obs["masks"]
        # Keeper left to the AI: nothing to choose for slot 0.
        for name in m:
            assert m[name][0, 0] and not m[name][0, 1:].any(), name
        # Wingers are far from the ball: no kicks. We have it: no defending.
        assert not m["ball"][2, 1:].any() and not m["ball"][3, 1:].any()
        assert not m["defend"][1:4, 1:].any()
    finally:
        e.close()
    k = OutfieldEnv("outfield_3v3", seed=2, keeper="agent")
    try:
        obs = k.reset()
        m = obs["masks"]
        assert m["move"][0].all()
        # The keeper's set (agreed with G): walk / sprint, passes only, no defensive actions.
        assert m["speed"][0].tolist() == [True, True, False]
        assert not m["ball"][0, 4:].any(), "no shot, curl shot, chip or through ball"
        assert m["defend"][0].tolist() == [True, False, False, False, False]
        # Even while they have the ball and it is at his feet.
        raw =dict(k.raw_list[0], ball=np.asarray(k.raw_list[0]["left_team"][0]).tolist() + [0.1],
                   ball_owned_team=1, ball_owned_player=1)
        m2 = act.masks(raw, 1, k.present, keeper_agent=True)
        assert m2["ball"][0, 1:4].all() and not m2["ball"][0, 4:].any()
        assert m2["defend"][0].tolist() == [True, False, False, False, False]
        assert m2["defend"][1].all(), "outfield players keep every defensive action"
    finally:
        k.close()


def test_kick_waits_for_the_wind_up(env):
    # Our midfielder walks onto the ball, then the policy presses a short pass:
    # he stays on our control until the ball has left his foot.
    env.reset()
    a = _idle()
    a["move"][1] = 1  # toward their goal, onto the ball
    for _ in range(30):
        _, _, term, trunc, info = env.step(a)
        if env.raw_list[0]["ball_owned_team"] == 0 and env.raw_list[0]["ball_owned_player"] == 1:
            break
    else:
        pytest.fail("midfielder never reached the ball")
    a["ball"][1] = act.BALL.index("short_pass")
    a["power"][1] = 0.4
    _, _, _, _, info = env.step(a)
    assert any(e["event"] == "pressed" and e["slot"] == 1 for e in info["kicks"])
    a["ball"][1] = 0
    events, held = [], 0
    for _ in range(15):
        _, _, term, trunc, info = env.step(a)
        events += info["kicks"]
        held += int(info["overridden"][1])
        if term or trunc:
            break
    assert any(e["event"] == "left_foot" and e["slot"] == 1 for e in events)
    assert held >= 1, "the policy's next choices were held back during the wind-up"


def test_ai_plays_the_ball():
    e = OutfieldEnv("outfield_3v3", seed=1, ai_on_ball=True)
    try:
        # Only a player who *has* the ball is handed over: loose balls stay the
        # policy's job, so the midfielder walks onto it himself.
        e.reset()
        a = _idle()
        a["move"][1] = 1
        seen = 0
        for _ in range(60):
            raw = e.raw_list[0]  # the state the command is computed from
            carrier = int(raw["ball_owned_player"]) if raw["ball_owned_team"] == 0 else None
            _, _, term, trunc, info = e.step(a)
            if carrier:
                assert info["overridden"][carrier]
                seen += 1
            assert info["overridden"][0]  # keeper is the game's AI
            if term or trunc:
                break
        assert seen > 0
    finally:
        e.close()
