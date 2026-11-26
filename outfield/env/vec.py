"""Many OutfieldEnv games at once, one per process.

    venv = VecEnv(8, scenario="11_vs_11_easy_stochastic", seeds=range(8))
    obs = venv.reset()                       # dict of arrays with a leading [8] axis
    obs, rewards, terminated, truncated, infos = venv.step(actions)   # actions: [8, 11] per head
    venv.close()

Each game runs in its own spawned process: the engine is CPU-only, one game
per core, and one engine setting per process (docs/CONTEXT.md §7.7). A
finished game resets by itself; its last observation is in
infos[k]["final_obs"] and the returned obs is the new game's first.
`instructions` (optional) is [n, 11, 85]; None leaves each game's sampler on.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback

import numpy as np


def _worker(conn, kwargs):
    import warnings
    warnings.filterwarnings("ignore")
    from outfield.env.env import OutfieldEnv
    env = None
    try:
        env = OutfieldEnv(**kwargs)
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                conn.send(("ok", env.reset(seed=data)))
            elif cmd == "step":
                actions, instructions = data
                obs, rewards, terminated, truncated, info = env.step(actions, instructions)
                info = {k: v for k, v in info.items() if k != "raw"}  # raw obs stay in the worker
                if terminated or truncated:
                    info["final_obs"] = obs
                    obs = env.reset()
                conn.send(("ok", (obs, rewards, terminated, truncated, info)))
            elif cmd == "close":
                break
    except Exception:  # noqa: BLE001  report, don't hang the parent
        conn.send(("error", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        conn.close()


def _stack(items):
    first = items[0]
    if isinstance(first, dict):
        return {k: _stack([it[k] for it in items]) for k in first}
    return np.stack(items)


class VecEnv:
    def __init__(self, n: int, scenario: str = "11_vs_11_easy_stochastic", seeds=None, **env_kwargs):
        seeds = list(range(n)) if seeds is None else list(seeds)
        assert len(seeds) == n
        ctx = mp.get_context("spawn")
        self.n = n
        self.conns, self.procs = [], []
        for k in range(n):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, daemon=True,
                               args=(child, dict(scenario=scenario, seed=seeds[k], **env_kwargs)))
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)

    def _gather(self):
        out = []
        for conn in self.conns:
            status, payload = conn.recv()
            if status == "error":
                self.close()
                raise RuntimeError(f"env worker failed:\n{payload}")
            out.append(payload)
        return out

    def reset(self, seeds=None) -> dict:
        seeds = [None] * self.n if seeds is None else list(seeds)
        for conn, seed in zip(self.conns, seeds):
            conn.send(("reset", seed))
        return _stack(self._gather())

    def step(self, actions: dict[str, np.ndarray], instructions: np.ndarray | None = None):
        for k, conn in enumerate(self.conns):
            per_env = {head: np.asarray(v)[k] for head, v in actions.items()}
            conn.send(("step", (per_env, None if instructions is None else instructions[k])))
        results = self._gather()
        obs = _stack([r[0] for r in results])
        rewards = _stack([r[1] for r in results])
        terminated = np.array([r[2] for r in results])
        truncated = np.array([r[3] for r in results])
        infos = [r[4] for r in results]
        return obs, rewards, terminated, truncated, infos

    def close(self):
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self.conns, self.procs = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
