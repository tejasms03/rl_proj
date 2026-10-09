"""AI-vs-AI match statistics from raw observations (test T5).

    python -m grf_tools.matchstats --matches 12 [--config SWITCH ...]

Possession and passes are inferred from ball ownership: a change of owner
within the left team is a completed pass, a change to the other team a
turnover. Real top-league reference: ~2.7 goals/match, ~80-85% pass
completion.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import multiprocessing

from grf_tools.play import BUILTIN_AI
from grf_tools.trace import make_env


@dataclasses.dataclass
class MatchStats:
    goals: tuple[int, int]
    possession_left: float
    passes: int  # completed, both teams
    turnovers: int
    steps: int
    shots: tuple[int, int] = (0, 0)


def play_match(seed: int, config: dict | None = None,
               scenario: str = "11_vs_11_stochastic") -> MatchStats:
    env = make_env(scenario, seed, {"action_set": "v2", **(config or {})})
    owner = None  # (team, player) of the last owner
    owned = [0, 0]
    passes = turnovers = steps = 0
    shots = [0, 0]
    shot_open = False  # a counted shot is still in flight
    try:
        env.reset()
        done = False
        while not done:
            obs, _, done, _ = env.step(BUILTIN_AI)
            o = obs[0]
            steps += 1
            team = int(o["ball_owned_team"])
            if team == -1:
                if owner is not None and not shot_open and _is_shot(o, owner[0]):
                    shots[owner[0]] += 1
                    shot_open = True
                continue
            shot_open = False
            owned[team] += 1
            current = (team, int(o["ball_owned_player"]))
            if owner is not None and current != owner:
                if current[0] == owner[0]:
                    passes += 1
                else:
                    turnovers += 1
            owner = current
    finally:
        env.close()
    total = max(sum(owned), 1)
    return MatchStats(tuple(int(s) for s in o["score"]), owned[0] / total, passes, turnovers,
                      steps, tuple(shots))


def _is_shot(o, team: int) -> bool:
    """Loose ball moving fast toward the goal `team` attacks, aimed near it."""
    x, y = o["ball"][0] * 54.4, o["ball"][1] * 83.6
    vx = o["ball_direction"][0] * 54.4 * 10
    vy = o["ball_direction"][1] * 83.6 * 10
    goal_x = 55.0 if team == 0 else -55.0
    toward = vx * (goal_x - x) > 0
    if not toward or abs(goal_x - x) > 35 or (vx * vx + vy * vy) < 12 ** 2:
        return False
    y_at_line = y + vy * (goal_x - x) / vx
    return abs(y_at_line) < 12


def _play_many(seeds, config):
    return [play_match(s, config) for s in seeds]


def play_matches(n: int, config: dict | None = None, workers: int = 4) -> list[MatchStats]:
    ctx = multiprocessing.get_context("spawn")
    chunks = [list(range(i, n, workers)) for i in range(workers)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = [pool.submit(_play_many, c, config) for c in chunks if c]
        return [m for f in futures for m in f.result()]


def summarize(matches: list[MatchStats]) -> dict[str, float]:
    n = len(matches)
    passes = sum(m.passes for m in matches)
    turnovers = sum(m.turnovers for m in matches)
    return {
        "matches": n,
        "goals_per_match": sum(sum(m.goals) for m in matches) / n,
        "passes_per_match": passes / n,
        "pass_completion": passes / max(passes + turnovers, 1),
        "shots_per_match": sum(sum(m.shots) for m in matches) / n,
        "conversion": sum(sum(m.goals) for m in matches) / max(sum(sum(m.shots) for m in matches), 1),
        "left_goals": sum(m.goals[0] for m in matches) / n,
        "right_goals": sum(m.goals[1] for m in matches) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=12)
    parser.add_argument("--config", action="append", default=[])
    args = parser.parse_args()
    config = {name: True for name in args.config}
    stats = summarize(play_matches(args.matches, config))
    print(" ".join(f"{k}={v:.3g}" for k, v in stats.items()), "config", sorted(config))


if __name__ == "__main__":
    main()
