"""The outfield env wrapper (docs/env_wrapper.md).

    from outfield.env import OutfieldEnv, VecEnv
"""

from outfield.env.actions import HEADS
from outfield.env.env import OBS_SIZE, STATE_SIZE, OutfieldEnv

__all__ = ["OutfieldEnv", "VecEnv", "HEADS", "OBS_SIZE", "STATE_SIZE", "random_actions"]


def __getattr__(name):
    if name == "VecEnv":  # lazy: keeps multiprocessing out of plain imports
        from outfield.env.vec import VecEnv
        return VecEnv
    raise AttributeError(name)


def random_actions(masks, rng):
    """Uniform random valid actions for every head (smoke tests, benchmarks)."""
    import numpy as np
    out = {}
    for head, m in masks.items():
        # The largest of uniform noise over the allowed entries: one valid pick per row.
        out[head] = np.argmax(np.where(m, rng.random(m.shape), -1.0), axis=-1)
    out["power"] = rng.uniform(0, 1, size=masks["move"].shape[:-1]).astype(np.float32)
    return out
