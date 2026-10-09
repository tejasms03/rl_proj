"""P2: realistic ball flight, kicks and pitch conditions.

Flights are launched with exact velocity/spin through the launch_ball test
hook (grf_tools.ballistics); kicks are measured from real player touches
(grf_tools.measure).
"""

import numpy as np
import pytest

from grf_tools.ballistics import flights
from grf_tools.measure import measure
from grf_tools.play import ai_episode
from grf_tools.trace import make_env

REAL = {"realistic_ball_physics": True}
REALISTIC_PACE = {**REAL, "ball_pace": "realistic"}
AIR_K = 0.0541  # rho * A / (2 m) for a size-5 ball


def launch(speed, degrees, heading=0.0):
    r, h = np.radians(degrees), np.radians(heading)
    return (speed * np.cos(r) * np.cos(h), speed * np.cos(r) * np.sin(h), speed * np.sin(r))


def drag_coefficient(flight, tick):
    """Cd from the drop in speed over a few in-air ticks around `tick`."""
    v = flight.vel
    speed = np.linalg.norm(v[tick])
    horizontal = np.linalg.norm(v[tick - 2, :2]) - np.linalg.norm(v[tick + 2, :2])
    deceleration = horizontal / (4 * 0.01) * speed / np.linalg.norm(v[tick, :2])
    return deceleration / (AIR_K * speed * speed)


def test_drag_crisis():
    fast, slow = flights([(launch(26, 10), (0, 0, 0)), (launch(9, 60), (0, 0, 0))], REAL,
                         ticks=60, start=(-40.0, 0.0, 2.0))
    assert drag_coefficient(fast, 10) == pytest.approx(0.22, abs=0.04)
    assert drag_coefficient(slow, 10) == pytest.approx(0.45, abs=0.06)


def test_backspin_lifts_topspin_dips():
    # Ball along +x: backspin is negative rotation about +y.
    back, none, top = flights(
        [(launch(22, 32), (0, -45, 0)), (launch(22, 32), (0, 0, 0)), (launch(22, 32), (0, 45, 0))],
        REAL, ticks=400)
    carry = {name: f.landing() for name, f in zip(("back", "none", "top"), (back, none, top))}
    assert carry["back"][1] > carry["none"][1] + 1.5 > carry["top"][1] + 3.0  # peak height
    assert carry["back"][0] > carry["none"][0] > carry["top"][0]  # carry


@pytest.mark.parametrize("sign", [1, -1])
def test_sidespin_bends_toward_the_spin_side(sign):
    (flight,) = flights([(launch(25, 6), (0, 0, sign * 60))], REAL, ticks=150)
    x = flight.pos[:, 0] - flight.pos[0, 0]
    i = int(np.argmax(x >= 25))
    bend = sign * (flight.pos[i, 1] - flight.pos[0, 1])
    # Curled free kicks bend ~2-4 m over 25 m at ~60 rad/s.
    assert 2.0 <= bend <= 6.0


def test_spin_decays_exponentially_in_flight():
    (flight,) = flights([(launch(15, 70), (0, -50, 0))], REAL, ticks=110,
                        start=(-40.0, 0.0, 1.0))
    ratio = flight.spin[100, 1] / flight.spin[0, 1]
    assert ratio == pytest.approx(np.exp(-1.0 / 5.0), rel=0.02)


def test_lofted_passes_arrive_at_the_receiver():
    # drill_pass_long: receiver ~40 m away, standing still.
    for kick in ("high_pass", "long_pass"):
        rows = measure(kick, range(4), config=REALISTIC_PACE)
        for row in rows:
            assert 35.0 <= row["travel_m"] <= 42.0, (kick, row["travel_m"])
    high = measure("high_pass", range(4), config=REALISTIC_PACE)
    for row in high:
        assert 28.0 <= row["launch_angle_deg"] <= 40.0
    # Peak varies with the kicking animation (~5-10 m); the typical ball floats.
    assert np.median([row["peak_height"] for row in high]) >= 6.0


def test_game_pace_is_flatter_and_quicker():
    # Default pace under realistic_ball_physics: FIFA-like tempo.
    for kick, angles in (("high_pass", (25.0, 30.0)), ("long_pass", (12.0, 18.5))):
        for row in measure(kick, range(3), config=REAL):
            assert 35.0 <= row["travel_m"] <= 42.0, (kick, row["travel_m"])
            assert angles[0] <= row["launch_angle_deg"] <= angles[1], (kick, row)
    game = measure("short_pass", range(2), config=REAL)
    slow = measure("short_pass", range(2), config=REALISTIC_PACE)
    assert min(r["launch_speed"] for r in game) > max(r["launch_speed"] for r in slow) + 2.0


def test_ground_pass_arrives_at_the_receiver():
    for row in measure("short_pass", range(3), config=REAL):
        assert 11.0 <= row["travel_m"] <= 16.0


def rolls(config):
    f3, f12 = flights([((3, 0, 0), (0, 0, 0)), ((12, 0, 0), (0, 0, 0))], config, ticks=1500)
    return [(f.pos[:, 0] - f.pos[0, 0]).max() for f in (f3, f12)]


def test_realistic_grass_meets_fifa_roll_standard():
    normal3, _ = rolls(REAL)
    assert 4.0 <= normal3 <= 8.0  # FIFA Quality: 4-8 m from ~3 m/s


def test_pitch_conditions_order_roll_and_bounce():
    roll = {c: rolls({**REAL, "pitch_condition": c})[1] for c in ("normal", "wet", "long_grass")}
    assert roll["wet"] > roll["normal"] > roll["long_grass"]

    drops = {}
    for condition in ("dry", "normal", "wet"):
        (f,) = flights([((0.01, 0, 0), (0, 0, 0))], {"pitch_condition": condition}, ticks=200,
                       start=(-40.0, 0.0, 3.0))
        z = f.pos[:, 2]
        drops[condition] = z[int(np.argmax(z < 0.13)):].max()
    assert drops["dry"] > drops["normal"] > drops["wet"]


def test_unknown_pitch_condition_is_rejected():
    with pytest.raises(ValueError, match="pitch_condition"):
        make_env("academy_empty_goal_close", 1, {"pitch_condition": "icy"}).reset()


def test_full_matches_run_with_every_patch_on():
    # Regression: NaN spin once crashed full matches (assert in SelectAnim).
    config = {**REAL, "realistic_player_physics": True, "pitch_condition": "wet"}
    for seed in (0, 1):
        assert ai_episode("11_vs_11_stochastic", seed, config).steps >= 3000
