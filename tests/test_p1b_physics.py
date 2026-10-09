"""P1b: realistic_player_physics widens top speed and acceleration ranges.

Targets (standing start, timed from first movement, professional players):
top speed 7.8-9.8 m/s; an average player runs 10 m in ~1.78 s, 20 m in
~3.0 s and 40 m in ~5.2 s.
"""

import pytest

from grf_tools.sprint import sprint

REALISTIC = {"realistic_player_physics": True}


def run(velocity, acceleration, config=None):
    return sprint({"physical_velocity": velocity, "physical_acceleration": acceleration},
                  config=config)


@pytest.mark.parametrize(("velocity", "top"), [(0.0, 7.8), (0.5, 8.8), (1.0, 9.8)])
def test_top_speed_spans_real_range(velocity, top):
    assert run(velocity, 0.5, REALISTIC)["top_speed"] == pytest.approx(top, abs=0.1)


@pytest.mark.parametrize(("velocity", "top"), [(0.0, 7.2), (1.0, 8.0)])
def test_switch_off_keeps_upstream_speeds(velocity, top):
    assert run(velocity, 0.5)["top_speed"] == pytest.approx(top, abs=0.05)


def test_average_player_splits_match_professionals():
    r = run(0.5, 0.5, REALISTIC)
    assert 1.70 <= r["t10"] <= 1.90
    assert 2.90 <= r["t20"] <= 3.10
    assert 5.10 <= r["t40"] <= 5.40


def test_better_stats_are_faster_over_40m():
    slow, average, fast = run(0.0, 0.0, REALISTIC), run(0.5, 0.5, REALISTIC), run(1.0, 1.0, REALISTIC)
    assert slow["t40"] > average["t40"] + 0.4 > fast["t40"] + 0.8
    # Extremes stay inside what footballers run (~4.6 s to ~6.0 s for 40 m).
    assert 4.6 <= fast["t40"] and slow["t40"] <= 6.0

