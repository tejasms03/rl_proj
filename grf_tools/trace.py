"""Record a compact, exact fingerprint of a game for equivalence tests.

Each step's raw observation (ball, players, score, mode, possession) is hashed
into a 64-bit digest. Two runs are identical if every per-step digest matches;
on a mismatch the first differing step points at where the engines diverged.
"""

from __future__ import annotations

import dataclasses
import hashlib
import warnings
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

import gfootball.env as football_env  # noqa: E402

N_ACTIONS_DEFAULT = 19
OBS_KEYS = (
    "ball",
    "ball_direction",
    "ball_rotation",
    "left_team",
    "left_team_direction",
    "right_team",
    "right_team_direction",
    "score",
    "ball_owned_team",
    "ball_owned_player",
    "game_mode",
)


@dataclasses.dataclass
class Trace:
    scenario: str
    seed: int
    digests: np.ndarray  # uint64, one per observation (step 0 = after reset)
    score: tuple[int, int]
    steps: int


def digest_obs(obs: dict[str, Any]) -> int:
    h = hashlib.blake2b(digest_size=8)
    for key in OBS_KEYS:
        h.update(np.ascontiguousarray(obs[key], dtype=np.float64).tobytes())
    return int.from_bytes(h.digest(), "little")


def make_env(scenario: str, seed: int, config: dict[str, Any] | None = None):
    options = {"game_engine_random_seed": seed}
    options.update(config or {})
    return football_env.create_environment(
        scenario, representation="raw", render=False, other_config_options=options
    )


def run_trace(
    scenario: str,
    seed: int,
    max_steps: int = 3000,
    config: dict[str, Any] | None = None,
    n_actions: int = N_ACTIONS_DEFAULT,
) -> Trace:
    """Play one episode with seeded random actions and fingerprint every step."""
    env = make_env(scenario, seed, config)
    rng = np.random.default_rng(seed)
    try:
        obs = env.reset()
        digests = [digest_obs(obs[0])]
        steps = 0
        for _ in range(max_steps):
            obs, _, done, _ = env.step(int(rng.integers(n_actions)))
            digests.append(digest_obs(obs[0]))
            steps += 1
            if done:
                break
        score = tuple(int(s) for s in obs[0]["score"])
    finally:
        env.close()
    return Trace(scenario, seed, np.array(digests, dtype=np.uint64), score, steps)


def first_divergence(a: np.ndarray, b: np.ndarray) -> int | None:
    """Index of the first differing step, or None if identical."""
    n = min(len(a), len(b))
    diff = np.nonzero(a[:n] != b[:n])[0]
    if len(diff):
        return int(diff[0])
    return None if len(a) == len(b) else n
