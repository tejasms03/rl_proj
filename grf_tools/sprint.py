"""Measure sprint performance (P1b): split times and top speed from standing.

    python -m grf_tools.sprint
"""

from __future__ import annotations

import numpy as np

from grf_tools.trace import make_env

from gfootball.env import football_action_set  # isort: skip

STEP_S = 0.1  # default env step: 10 physics ticks of 10 ms
X_SCALE = 54.4
_ACTIONS = {a._name: i for i, a in enumerate(
    football_action_set.get_action_set({"action_set": "default"}))}


def sprint(stats: dict, seed: int = 0, seconds: float = 8.0, config: dict | None = None) -> dict:
    """Sprint straight ahead from standing; return 10/20/30 m splits and top speed."""
    overrides = {"left": {1: {"stats": stats}}}
    env = make_env("drill_sprint", seed, {"player_overrides": overrides, **(config or {})})
    try:
        obs = env.reset()[0]
        x0 = obs["left_team"][1][0]
        xs = [0.0]
        env.step(_ACTIONS["sprint"])
        xs.append((env.step(_ACTIONS["right"])[0][0]["left_team"][1][0] - x0) * X_SCALE)
        for _ in range(int(seconds / STEP_S) - 1):
            xs.append((env.step(_ACTIONS["right"])[0][0]["left_team"][1][0] - x0) * X_SCALE)
    finally:
        env.close()
    xs = np.array(xs)
    t = np.arange(len(xs)) * STEP_S

    def crossing(d):
        i = int(np.argmax(xs >= d))
        if xs[i] < d:
            return float("nan")
        # linear interpolation between samples
        return float(t[i - 1] + (d - xs[i - 1]) / (xs[i] - xs[i - 1]) * STEP_S)

    # Time from first movement, like timing gates (excludes input latency).
    start = crossing(0.05)

    def split(d):
        return crossing(d) - start

    speed = np.diff(xs) / STEP_S
    return {"t10": split(10), "t20": split(20), "t30": split(30), "t40": split(40),
            "top_speed": float(speed.max())}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--realistic", action="store_true",
                        help="turn on the P1b realistic_player_physics switch")
    args = parser.parse_args()
    config = {"realistic_player_physics": True} if args.realistic else {}
    for v in [0.0, 0.5, 1.0]:
        for a in [0.0, 0.5, 1.0]:
            r = sprint({"physical_velocity": v, "physical_acceleration": a}, config=config)
            print(f"velocity {v:.1f} accel {a:.1f}: 10m {r['t10']:.2f}s  20m {r['t20']:.2f}s  "
                  f"30m {r['t30']:.2f}s  40m {r['t40']:.2f}s  top {r['top_speed']:.2f} m/s")


if __name__ == "__main__":
    main()
