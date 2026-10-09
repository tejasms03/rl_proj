"""P4: shot/pass accuracy depends on pressure, composure and weak foot;
first touch on incoming ball speed and ball control."""

import concurrent.futures
import multiprocessing

import numpy as np

X_SCALE, Y_SCALE = 54.4, -83.6
ON = {"realistic_ball_physics": True, "situational_accuracy": True}


def _shots_here(scenario, config, seeds):
    import warnings
    warnings.filterwarnings("ignore")
    from grf_tools.measure import _action_index
    from grf_tools.trace import make_env
    idx = _action_index()
    out = []
    for seed in seeds:
        env = make_env(scenario, seed, {"physics_steps_per_frame": 1, **config})
        pos = []
        try:
            env.reset()
            for t in range(150):
                a = idx["right"] if t < 30 else (idx["shot"] if t < 35 else idx["release_direction"])
                obs, _, done, _ = env.step(a)
                o = obs[0]
                pos.append((o["ball"][0] * X_SCALE, o["ball"][1] * Y_SCALE))
                if done:
                    break
        finally:
            env.close()
        pos = np.array(pos)
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / 0.01
        k = int(np.argmax(speed > 10.0))
        d = pos[min(k + 8, len(pos) - 1)] - pos[k]
        out.append(float(np.degrees(np.arctan2(d[1], d[0]))))  # launch heading
    return out


def shot_headings(scenario, config, seeds):
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        return np.array(pool.submit(_shots_here, scenario, config, list(seeds)).result())


def nervous(config):
    # A low-composure shooter makes pressure matter most.
    return {**config, "player_overrides": {"left": {1: {"stats": {"mental_calmness": 0.1}}}}}


def test_pressure_spreads_shots():
    seeds = range(12)
    free = shot_headings("drill_shot", nervous(ON), seeds)
    pressed = shot_headings("drill_pressure", nervous(ON), seeds)
    assert np.std(pressed) > np.std(free) * 1.3


def test_weak_foot_changes_kicks_only_with_the_switch():
    seeds = range(8)
    def headings(config, foot):
        cfg = {**config, "player_overrides": {"left": {1: {"preferred_foot": foot}}}}
        return shot_headings("drill_pressure", cfg, seeds)
    on_right, on_left = headings(ON, "right"), headings(ON, "left")
    assert np.mean(np.abs(on_right - on_left) > 0.05) >= 0.75
    off = {"realistic_ball_physics": True}
    np.testing.assert_array_equal(headings(off, "right"), headings(off, "left"))


def _touch_here(control, config):
    import warnings
    warnings.filterwarnings("ignore")
    from grf_tools.measure import _action_index
    from grf_tools.trace import make_env
    idx = _action_index()
    cfg = {"physics_steps_per_frame": 1, **config,
           "player_overrides": {"left": {2: {"stats": {"technical_ballcontrol": control}}}}}
    worst = []
    for seed in range(6):
        env = make_env("drill_receive", seed, cfg)
        try:
            env.reset()
            # A firm ground ball straight at the receiver (~29 m, x), arriving
            # at ~12 m/s; he is AI-controlled, so he actually traps it (lazy
            # players don't).
            env.unwrapped._env._env.launch_ball(14.0, 0.0, 0.11, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            gap, touched = 0.0, None
            for t in range(300):
                obs, _, done, _ = env.step(idx["idle"])
                o = obs[0]
                if done:
                    break
                if touched is None and o["ball_owned_team"] == 0 and o["ball_owned_player"] == 2:
                    touched = t
                if touched is not None and t <= touched + 100:
                    ball = np.array(o["ball"][:2]) * [X_SCALE, Y_SCALE]
                    me = np.array(o["left_team"][2]) * [X_SCALE, Y_SCALE]
                    gap = max(gap, float(np.linalg.norm(ball - me)))
        finally:
            env.close()
        worst.append(gap)
    return float(np.mean(worst))


def first_touch_gap(control, config):
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        return pool.submit(_touch_here, control, config).result()


def test_poor_ball_control_gives_heavier_first_touch():
    on_gap = first_touch_gap(0.1, ON) - first_touch_gap(1.0, ON)
    off = {"realistic_ball_physics": True}
    off_gap = first_touch_gap(0.1, off) - first_touch_gap(1.0, off)
    assert on_gap > off_gap + 0.3
