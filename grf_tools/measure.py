"""Measure ball flight from real kicks in the drill scenarios (test T3).

    python -m grf_tools.measure            # baseline table, all kick types
    python -m grf_tools.measure --csv out  # also write per-kick trajectories

Physics runs at 100 Hz; we step the env one physics tick at a time
(physics_steps_per_frame=1) so trajectories are sampled every 10 ms.
Positions are converted from observation units back to engine metres.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import multiprocessing
from pathlib import Path

import numpy as np

from grf_tools.trace import make_env

from gfootball.env import football_action_set  # isort: skip

DT = 0.01  # seconds per physics tick
X_SCALE, Y_SCALE = 54.4, -83.6  # obs units -> metres (engine defines.hpp)
BALL_RADIUS = 0.11

# kick name -> (drill scenario, action pressed to kick)
KICKS = {
    "shot": ("drill_shot", "shot"),
    "short_pass": ("drill_pass_short", "short_pass"),
    "long_pass": ("drill_pass_long", "long_pass"),
    "high_pass": ("drill_pass_long", "high_pass"),
}


@dataclasses.dataclass
class Trajectory:
    kick: str
    seed: int
    pos: np.ndarray  # (T, 3) metres
    rot: np.ndarray  # (T, 3) engine ball_rotation

    @property
    def vel(self) -> np.ndarray:
        return np.diff(self.pos, axis=0) / DT


def _action_index() -> dict[str, int]:
    actions = football_action_set.get_action_set({"action_set": "default"})
    return {a._name: i for i, a in enumerate(actions)}


def kick_trajectory(kick: str, seed: int, approach_ticks: int = 30, max_ticks: int = 400,
                    config: dict | None = None) -> Trajectory:
    """Run toward goal, press the kick action, then record the ball's flight."""
    scenario, action = KICKS[kick]
    idx = _action_index()
    plan = [idx["right"]] * approach_ticks + [idx[action]] * 5
    options = {"physics_steps_per_frame": 1, **(config or {})}
    env = make_env(scenario, seed, options)
    pos, rot = [], []
    try:
        env.reset()
        for t in range(max_ticks):
            a = plan[t] if t < len(plan) else idx["release_direction"]
            obs, _, done, _ = env.step(a)
            o = obs[0]
            b = o["ball"]
            pos.append((b[0] * X_SCALE, b[1] * Y_SCALE, b[2]))
            rot.append(o["ball_rotation"])
            if done:
                break
    finally:
        env.close()
    return Trajectory(kick, seed, np.array(pos), np.array(rot, dtype=float))


def summarize(traj: Trajectory) -> dict[str, float]:
    """Launch speed/angle, peak height, carry and sideways bend of one kick."""
    vel = traj.vel
    speed = np.linalg.norm(vel, axis=1)
    k = int(np.argmax(speed))  # tick of maximum speed ~= right after contact
    v0 = vel[k]
    horiz0 = v0[:2] / max(np.linalg.norm(v0[:2]), 1e-9)
    flight = traj.pos[k:]
    rel = flight[:, :2] - flight[0, :2]
    along = rel @ horiz0
    side = rel @ np.array([-horiz0[1], horiz0[0]])

    airborne = np.nonzero(flight[:, 2] > BALL_RADIUS + 0.05)[0]
    if len(airborne):
        land = airborne[-1] + 1 if airborne[-1] + 1 < len(flight) else airborne[-1]
        # first landing after the first airborne tick
        gaps = np.nonzero(flight[airborne[0]:, 2] <= BALL_RADIUS + 0.02)[0]
        land = airborne[0] + gaps[0] if len(gaps) else len(flight) - 1
        carry = float(along[land])
    else:
        carry = 0.0

    # bend: sideways offset after 20 m of travel (or at the end if shorter)
    i20 = np.nonzero(along >= 20.0)[0]
    j = int(i20[0]) if len(i20) else len(along) - 1
    return {
        "launch_speed": float(speed[k]),
        "launch_angle_deg": float(np.degrees(np.arctan2(v0[2], np.linalg.norm(v0[:2])))),
        "peak_height": float(flight[:, 2].max()),
        "carry_m": carry,
        "travel_m": float(along.max()),
        "bend_at_20m": float(side[j]),
        "bend_dist_m": float(along[j]),
        "speed_after_1s": float(speed[min(k + 100, len(speed) - 1)]),
    }


def _measure_here(kick: str, seeds: list[int], config: dict | None) -> list[dict[str, float]]:
    return [summarize(kick_trajectory(kick, s, config=config)) for s in seeds]


def measure(kick: str, seeds: range, config: dict | None = None) -> list[dict[str, float]]:
    """Summaries for several seeds, run in a fresh process at 100 Hz sampling.

    A separate process is required: the engine's tick rate is fixed by the
    first env created in a process (see trace.make_env).
    """
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        return pool.submit(_measure_here, kick, list(seeds), config).result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--config", action="append", default=[],
                        help="patch switch to turn on, e.g. realistic_ball_physics")
    args = parser.parse_args()
    config = {name: True for name in args.config}
    fields = ["launch_speed", "launch_angle_deg", "peak_height", "carry_m", "travel_m",
              "bend_at_20m", "speed_after_1s"]
    print(f"{'kick':12s}" + "".join(f"{f:>18s}" for f in fields))
    for kick in KICKS:
        trajs = [kick_trajectory(kick, s, config=config) for s in range(args.seeds)]
        rows = [summarize(t) for t in trajs]
        cells = []
        for f in fields:
            v = np.array([r[f] for r in rows])
            cells.append(f"{np.median(v):8.2f} ±{v.std():6.2f}  ")
        print(f"{kick:12s}" + "".join(f"{c:>18s}" for c in cells))
        if args.csv:
            args.csv.mkdir(parents=True, exist_ok=True)
            for t in trajs:
                np.savetxt(args.csv / f"{kick}_seed{t.seed}.csv",
                           np.hstack([t.pos, t.rot]), delimiter=",",
                           header="x_m,y_m,z_m,rot_x,rot_y,rot_z", comments="")


if __name__ == "__main__":
    main()
