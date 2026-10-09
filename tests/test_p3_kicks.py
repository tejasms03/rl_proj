"""P3: curl shot, chip and through ball (action set v3)."""

import concurrent.futures
import multiprocessing

import numpy as np
import pytest

from gfootball.env import football_action_set

V3 = {a._name: i for i, a in enumerate(football_action_set.get_action_set({"action_set": "v3"}))}
CONFIG = {"realistic_ball_physics": True, "action_set": "v3", "physics_steps_per_frame": 1}
X_SCALE, Y_SCALE = 54.4, -83.6


def _kick_here(scenario, kick, seed, ticks):
    import warnings
    warnings.filterwarnings("ignore")
    from grf_tools.trace import make_env
    env = make_env(scenario, seed, CONFIG)
    pos, spin, score = [], [], (0, 0)
    try:
        env.reset()
        for t in range(ticks):
            action = V3["right"] if t < 30 else (V3[kick] if t < 35 else V3["release_direction"])
            obs, _, done, _ = env.step(action)
            o = obs[0]
            pos.append((o["ball"][0] * X_SCALE, o["ball"][1] * Y_SCALE, o["ball"][2]))
            r = o["ball_rotation"]
            spin.append((r[0] * X_SCALE * 1000, r[1] * Y_SCALE * 1000, r[2] * 1000))
            score = tuple(o["score"])
            if done:
                break
    finally:
        env.close()
    return np.array(pos), np.array(spin), score


def kick(scenario, action, seed=0, ticks=300):
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        return pool.submit(_kick_here, scenario, action, seed, ticks).result()


def flight(pos):
    """Trajectory from the kick: the first tick the ball jumps above 10 m/s
    (later collisions with the net can be faster than a soft chip)."""
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / 0.01
    k = int(np.argmax(speed > 10.0))
    return pos[k:], speed[k]


def bend(pos):
    """Largest sideways deviation from the launch direction (m)."""
    f, _ = flight(pos)
    d = f[3, :2] - f[0, :2]
    d /= np.linalg.norm(d)
    rel = f[:, :2] - f[0, :2]
    return float(np.max(np.abs(rel @ np.array([-d[1], d[0]]))))


def test_action_set_v3():
    v1 = football_action_set.get_action_set({"action_set": "default"})
    v3 = football_action_set.get_action_set({"action_set": "v3"})
    assert len(v1) == 19 and len(v3) == 22
    assert v3[:19] == v1
    assert [a._name for a in v3[19:]] == ["curl_shot", "chip", "through_ball"]


def test_curl_shot_bends_much_more_than_a_shot():
    curls = [bend(kick("drill_shot", "curl_shot", s)[0]) for s in range(3)]
    shots = [bend(kick("drill_shot", "shot", s)[0]) for s in range(3)]
    assert min(curls) > 1.5
    assert np.median(curls) > np.median(shots) + 1.0
    _, spin, _ = kick("drill_shot", "curl_shot", 0)
    assert np.abs(spin[:, 2]).max() == pytest.approx(55, abs=8)


def test_chip_clears_a_keeper_and_drops_under_the_bar():
    for seed in range(3):
        pos, _, score = kick("drill_shot", "chip", seed, ticks=500)  # ~3.5 s flight
        f, _ = flight(pos)
        goal_x = 55.0
        at_keeper = f[np.argmin(np.abs(f[:, 0] - (goal_x - 7)))]  # an advanced keeper
        assert at_keeper[2] > 2.6, f"seed {seed}: only {at_keeper[2]:.2f} m high 7 m out"
        crossing = f[np.argmax(f[:, 0] >= goal_x)] if (f[:, 0] >= goal_x).any() else None
        assert crossing is not None and crossing[2] < 2.44, f"seed {seed}: over the bar"
        assert score[0] == 1, f"seed {seed}: no goal"


def test_through_ball_goes_beyond_a_standing_receiver():
    # drill_through: receiver ~16 m ahead and to the side. A normal pass is
    # played to his feet; a through ball runs into the space beyond him.
    receiver_x = 0.1 * X_SCALE
    def furthest_x(action):
        f, _ = flight(kick("drill_through", action, 0)[0])
        return float(f[:, 0].max())
    assert furthest_x("short_pass") < receiver_x + 2.0
    assert furthest_x("through_ball") > receiver_x + 3.0
