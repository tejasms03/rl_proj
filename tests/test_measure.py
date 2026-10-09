"""The measurement tool records sensible kicks from the unpatched engine."""

import pytest

from grf_tools.measure import KICKS, measure

# Loose bounds: catch a broken drill or tool, not tune physics.
BOUNDS = {
    "launch_speed": (5.0, 45.0),
    "peak_height": (0.11, 30.0),
    "travel_m": (5.0, 80.0),
}


@pytest.mark.parametrize("kick", KICKS)
def test_kick_is_measured(kick):
    for row in measure(kick, range(2)):
        for field, (lo, hi) in BOUNDS.items():
            assert lo <= row[field] <= hi, (field, row[field])


def test_high_pass_goes_higher_than_shot():
    shot = measure("shot", range(3))
    lob = measure("high_pass", range(3))
    assert min(r["peak_height"] for r in lob) > max(r["peak_height"] for r in shot)
