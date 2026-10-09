"""Launch the ball with exact velocity/spin and record its flight (P2, T3).

Uses the launch_ball test hook on an empty pitch (drill_ball_lab), sampled
every physics tick (10 ms). Positions are engine metres. Run in a process
that uses physics_steps_per_frame=1 (see trace.make_env).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import multiprocessing

import numpy as np

from grf_tools.trace import make_env

DT = 0.01
X_SCALE, Y_SCALE = 54.4, -83.6
BALL_RADIUS = 0.11


@dataclasses.dataclass
class Flight:
    pos: np.ndarray  # (T, 3) metres, row 0 = launch point
    spin: np.ndarray  # (T, 3) angular velocity about +x, +y, +z, rad/s

    @property
    def vel(self) -> np.ndarray:
        return np.diff(self.pos, axis=0) / DT

    def landing(self) -> tuple[float, float, int]:
        """(horizontal distance, peak height, tick) of the first landing."""
        z = self.pos[:, 2]
        airborne = np.nonzero(z > BALL_RADIUS + 0.02)[0]
        if not len(airborne):
            return 0.0, float(z.max()), 0
        after = np.nonzero(z[airborne[0]:] <= BALL_RADIUS + 0.005)[0]
        i = airborne[0] + after[0] if len(after) else len(z) - 1
        dist = float(np.linalg.norm(self.pos[i, :2] - self.pos[0, :2]))
        return dist, float(z[:i + 1].max()), int(i)


def _flight_here(vel, spin, start, ticks, config):
    env = make_env("drill_ball_lab", 0, {"physics_steps_per_frame": 1, **(config or {})})
    try:
        env.reset()
        engine = env.unwrapped._env._env
        engine.launch_ball(*start, *vel, *spin)
        pos, rot = [], []
        for _ in range(ticks):
            o = env.step(0)[0][0]
            b = o["ball"]
            pos.append((b[0] * X_SCALE, b[1] * Y_SCALE, b[2]))
            # The observation scales rotation like positions; undo that to get
            # angular velocity about +x, +y, +z in rad/s.
            r = o["ball_rotation"]
            rot.append((r[0] * X_SCALE * 1000.0, r[1] * Y_SCALE * 1000.0, r[2] * 1000.0))
    finally:
        env.close()
    return Flight(np.vstack([np.asarray(start, float)[None], np.array(pos)]),
                  np.array(rot))


def flights(launches, config=None, ticks=400, start=(-40.0, 0.0, BALL_RADIUS)):
    """Run [(vel, spin), ...] in one fresh process; returns [Flight, ...].

    `spin` is passed to Ball::SetRotation, whose x axis points along -x: a
    physical angular velocity (wx, wy, wz) is spin=(-wx, wy, wz).
    """
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        return pool.submit(_run_many, list(launches), config, ticks, start).result()


def _run_many(launches, config, ticks, start):
    return [_flight_here(v, s, start, ticks, config) for v, s in launches]
