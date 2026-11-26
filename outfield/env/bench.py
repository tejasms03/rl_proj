"""Steps per second of the wrapper, alone and in parallel.

    python -m outfield.env.bench                       # 11v11 and 3v3, 1 / 4 / 8 workers
    python -m outfield.env.bench --workers 1 16 --steps 2000 --scenario 11_vs_11_easy_stochastic

A "step" is one 0.1 s game step of one game; with N workers the figure is
the total over all games. Random valid actions, the AI-on-the-ball rule on.
The number decides how many seeds and how long 11v11 training can be
(docs/CONTEXT.md §9): give it to M1 with the machine's core count.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def measure(scenario: str, workers: int, steps: int) -> float:
    from outfield.env import VecEnv, random_actions
    rng = np.random.default_rng(0)
    with VecEnv(workers, scenario=scenario) as venv:
        obs = venv.reset()
        for _ in range(20):  # warm-up
            obs, *_ = venv.step(random_actions(obs["masks"], rng))
        start = time.perf_counter()
        for _ in range(steps):
            obs, *_ = venv.step(random_actions(obs["masks"], rng))
        return workers * steps / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", nargs="+", default=["11_vs_11_easy_stochastic", "outfield_3v3"])
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    print(f"machine: {os.cpu_count()} cores")
    print("| scenario | workers | steps/s (all games) | steps/s per game |\n|---|---|---|---|")
    for scenario in args.scenario:
        for w in args.workers:
            rate = measure(scenario, w, args.steps)
            print(f"| {scenario} | {w} | {rate:.0f} | {rate / w:.0f} |", flush=True)


if __name__ == "__main__":
    main()
