"""T1: with every patch switched off, games match the unpatched engine exactly."""

import pytest

from grf_tools import reference
from grf_tools.trace import first_divergence, run_trace


@pytest.mark.parametrize(("scenario", "seed"), reference.CASES)
def test_matches_unpatched_reference(scenario, seed):
    expected, meta = reference.load(scenario, seed)
    trace = run_trace(scenario, seed)
    step = first_divergence(expected, trace.digests)
    assert step is None, f"diverged from the unpatched engine at step {step}"
    assert list(trace.score) == meta["score"]
