"""Reference games recorded from the unpatched engine (test T1).

    python -m grf_tools.reference record   # only on an unpatched build
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from grf_tools.trace import Trace, run_trace

REF_DIR = Path(__file__).resolve().parent.parent / "tests" / "reference"

# (scenario, seed): three full 11v11 matches plus every academy drill.
CASES: list[tuple[str, int]] = [
    ("11_vs_11_stochastic", 1),
    ("11_vs_11_stochastic", 2),
    ("11_vs_11_easy_stochastic", 3),
    ("academy_empty_goal_close", 1),
    ("academy_empty_goal", 1),
    ("academy_run_to_score", 1),
    ("academy_run_to_score_with_keeper", 1),
    ("academy_pass_and_shoot_with_keeper", 1),
    ("academy_run_pass_and_shoot_with_keeper", 1),
    ("academy_3_vs_1_with_keeper", 1),
    ("academy_corner", 1),
    ("academy_counterattack_easy", 1),
    ("academy_counterattack_hard", 1),
    ("academy_single_goal_versus_lazy", 1),
]


def ref_path(scenario: str, seed: int) -> Path:
    return REF_DIR / f"{scenario}__seed{seed}.npz"


def save(trace: Trace) -> None:
    np.savez_compressed(
        ref_path(trace.scenario, trace.seed),
        digests=trace.digests,
        meta=json.dumps({"score": trace.score, "steps": trace.steps}),
    )


def load(scenario: str, seed: int) -> tuple[np.ndarray, dict]:
    with np.load(ref_path(scenario, seed)) as data:
        return data["digests"], json.loads(str(data["meta"]))


def record() -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    for scenario, seed in CASES:
        trace = run_trace(scenario, seed)
        save(trace)
        print(f"{scenario:40s} seed={seed} steps={trace.steps:5d} score={trace.score}")


if __name__ == "__main__":
    if sys.argv[1:] == ["record"]:
        record()
    else:
        sys.exit(__doc__)
