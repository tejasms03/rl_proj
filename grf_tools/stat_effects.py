"""How much does each player stat change what happens? (2026-11-24)

Each test is a small scene where one stat (or one group of stats) is set low
(0.3) or high (1.0) for one player; every other stat of every player is 0.7
(the archetypes' average) and every height 1.82 m unless the test is about
height. Same seeds for both levels, so the only difference is the stat.

    python -m grf_tools.stat_effects                 # all tests
    python -m grf_tools.stat_effects --tests shot tackle --seeds 40

Training physics is on: realistic player and ball physics, situational
accuracy. The game's AI plays every player except where a test scripts a kick.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import multiprocessing
import sys
import types

import numpy as np

STATS = ('physical_balance', 'physical_reaction', 'physical_acceleration', 'physical_velocity',
         'physical_stamina', 'physical_agility', 'physical_shotpower', 'technical_standingtackle',
         'technical_slidingtackle', 'technical_ballcontrol', 'technical_dribble',
         'technical_shortpass', 'technical_highpass', 'technical_header', 'technical_shot',
         'technical_volley', 'mental_calmness', 'mental_workrate', 'mental_resilience',
         'mental_defensivepositioning', 'mental_offensivepositioning', 'mental_vision')
BASE = 0.7
HEIGHT = 1.82
LOW, HIGH = 0.3, 1.0
PHYSICS = {"realistic_player_physics": True, "realistic_ball_physics": True,
           "situational_accuracy": True}
X_M, Y_M = 54.4, 83.6


# ---------------------------------------------------------------- scenes

def _scene(build):
    def build_scenario(builder):
        config = builder.config()
        config.game_duration = 300
        config.deterministic = False
        config.offsides = False
        config.end_episode_on_score = True
        config.end_episode_on_out_of_play = True
        config.end_episode_on_possession_change = False
        build(builder)
    return build_scenario


def _register():
    from gfootball.scenarios import (Team, e_PlayerRole_CB, e_PlayerRole_CF, e_PlayerRole_CM,
                                     e_PlayerRole_GK)

    def shot(b):
        # Our striker 21 m out, central, their keeper in goal.
        b.SetBallPosition(0.62, 0.0)
        b.SetTeam(Team.e_Left)
        b.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=False)
        b.AddPlayer(0.605, 0.0, e_PlayerRole_CF)
        b.SetTeam(Team.e_Right)
        b.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)

    def aerial(side):
        # Our centre-back and their striker side by side, 1 m apart across the
        # ball's line; a high ball drops between them. `side` swaps who is on
        # which side (seed parity), so neither has a built-in head start.
        def build(b):
            b.SetBallPosition(-0.46, 0.0)
            b.SetTeam(Team.e_Left)
            b.AddPlayer(-1.0, 0.42, e_PlayerRole_GK, lazy=True, controllable=False)
            b.AddPlayer(0.0, 0.006 * side, e_PlayerRole_CB)
            b.SetTeam(Team.e_Right)
            b.AddPlayer(-1.0, 0.42, e_PlayerRole_GK, lazy=True, controllable=False)
            b.AddPlayer(0.0, 0.006 * side, e_PlayerRole_CF)  # own frame: screen (0, -0.006 * side)
        return build

    def aerial_solo(b):
        # Our centre-back alone; high balls fly over him at different heights.
        b.SetBallPosition(-0.46, 0.0)
        b.SetTeam(Team.e_Left)
        b.AddPlayer(-1.0, 0.42, e_PlayerRole_GK, lazy=True, controllable=False)
        b.AddPlayer(0.0, 0.0, e_PlayerRole_CB)
        b.SetTeam(Team.e_Right)
        b.AddPlayer(-1.0, 0.42, e_PlayerRole_GK, lazy=True, controllable=False)

    def header(b):
        # Our striker 10 m out, central; their keeper; crosses come from the right.
        b.SetBallPosition(0.81, 0.26)
        b.SetTeam(Team.e_Left)
        b.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=False)
        b.AddPlayer(0.82, 0.0, e_PlayerRole_CF)
        b.SetTeam(Team.e_Right)
        b.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)

    for name, build in (("shot", shot), ("aerial_a", aerial(1)), ("aerial_b", aerial(-1)),
                        ("aerial_solo", aerial_solo), ("header", header)):
        module = types.ModuleType(f"gfootball.scenarios.stat_{name}")
        module.build_scenario = _scene(build)
        sys.modules[module.__name__] = module


def _overrides(team: str, slot: int, stats: dict, height: float | None = None) -> dict:
    base = {"stats": {s: BASE for s in STATS}, "height": HEIGHT}
    target = {"stats": {**base["stats"], **stats}, "height": height or HEIGHT}
    other = "right" if team == "left" else "left"
    return {team: {"*": base, slot: target}, other: {"*": base}}


def _env(scenario: str, seed: int, overrides: dict, agents: int = 1, extra: dict | None = None):
    import warnings
    warnings.filterwarnings("ignore")
    from grf_tools.trace import make_env
    _register()
    config = {**PHYSICS, "player_overrides": overrides, "action_set": "v4", **(extra or {})}
    if agents != 1:
        config["number_of_left_players_agent_controls"] = agents
    return make_env(scenario, seed, config)


def _names():
    from gfootball.env import football_action_set
    return {a._name: i for i, a in enumerate(football_action_set.get_action_set({"action_set": "v4"}))}


# ---------------------------------------------------------------- tests
# Each returns one number per episode (or a dict); the runner averages them.

def sprint_test(stats: dict, seed: int) -> dict:
    """Sprint from standing (drill_sprint): time to 20 m and top speed."""
    from grf_tools.sprint import sprint
    r = sprint({s: BASE for s in STATS} | stats, seed=seed, config=PHYSICS)
    return {"time_20m_s": r["t20"], "top_speed_ms": r["top_speed"]}


def shot_test(stats: dict, seed: int) -> dict:
    """Our striker shoots once from 21 m at power 0.7 (scripted): goal, keeper, off target; ball speed."""
    a = _names()
    env = _env("stat_shot", seed, _overrides("left", 1, stats), extra={"kick_power_control": True})
    try:
        env.reset()
        plan = [a["right"]] * 2 + [(a["shot"], 0.7)] + [a["right"]] * 3
        goal = keeper = False
        top = 0.0
        prev = None
        for t in range(60):
            obs, _, done, info = env.step(plan[t] if t < len(plan) else a["idle"])
            o = obs[0]
            b = np.array(o["ball"][:2]) * [X_M, Y_M]
            if prev is not None and not done:
                speed = float(np.linalg.norm(b - prev)) / 0.1
                if speed < 50.0:  # larger = the ball being reset, not a kick
                    top = max(top, speed)
            prev = b
            goal |= info["score_reward"] > 0
            keeper |= o["ball_owned_team"] == 1
            if done:
                break
    finally:
        env.close()
    return {"goal": float(goal), "keeper_has_it": float(keeper and not goal),
            "off_target": float(not goal and not keeper), "ball_speed_ms": top}


def pass_test(kind: str, scenario: str = "drill_pass_long"):
    def run(stats: dict, seed: int) -> dict:
        """A harder pass: run with the ball, turn 45 degrees and pass at once to a
        still teammate (drill_pass_long, 40 m ahead). How close does it come?
        (Standing passes have almost no built-in difficulty, so the stat can't
        show there.)"""
        a = _names()
        env = _env(scenario, seed, _overrides("left", 1, stats))
        try:
            obs = env.reset()
            target = np.array(obs[0]["left_team"][2]) * [X_M, Y_M]
            plan = [a["sprint"]] + [a["right"]] * 8 + [a["top_right"], a[kind]] + [a["idle"]] * 2
            best = math.inf
            reached = False
            for t in range(70):
                obs, _, done, _ = env.step(plan[t] if t < len(plan) else a["idle"])
                o = obs[0]
                b = np.array(o["ball"][:2]) * [X_M, Y_M]
                if o["ball"][2] < 2.0 and t > 5:
                    best = min(best, float(np.linalg.norm(b - target)))
                reached |= o["ball_owned_team"] == 0 and o["ball_owned_player"] == 2
                if done:
                    break
        finally:
            env.close()
        return {"miss_m": best, "reached": float(reached)}
    return run


_DIRS = ["right", "bottom_right", "bottom", "bottom_left", "left", "top_left", "top", "top_right"]


def _toward(a, me, target) -> int:
    d = (np.array(target) - np.array(me)) * [X_M, Y_M]
    return a[_DIRS[int(round(math.atan2(d[1], d[0]) / (math.pi / 4))) % 8]]


def duel_test(side: str):
    def run(stats: dict, seed: int) -> dict:
        """1 v 1 (drill_defend): the game's AI attacks; our defender (scripted)
        holds the press button and runs at the ball, as our AI players will.
        Does the defender win the ball, or does the attacker score?"""
        a = _names()
        team, slot = ("left", 1) if side == "defender" else ("right", 1)
        env = _env("drill_defend", seed, _overrides(team, slot, stats))
        try:
            obs = env.reset()
            won = scored = False
            for t in range(300):
                o = obs[0]
                act = a["pressure"] if t == 0 else _toward(a, o["left_team"][1], o["ball"][:2])
                obs, _, done, info = env.step(act)
                o = obs[0]
                won |= o["ball_owned_team"] == 0 and o["ball_owned_player"] == 1
                scored |= info["score_reward"] < 0
                if done or won:
                    break
        finally:
            env.close()
        return {"defender_wins": float(won), "attacker_scores": float(scored and not won)}
    return run


def aerial_test(height_only: bool):
    def run(stats: dict, seed: int) -> dict:
        """A high ball drops on our centre-back and their striker: who touches it first?"""
        a = _names()
        height = stats.pop("_height", None)
        env = _env("stat_aerial_a" if seed % 2 == 0 else "stat_aerial_b", seed,
                   _overrides("left", 1, stats, height))
        try:
            env.reset()
            vz = 9.0 + 0.2 * ((seed // 2) % 4)  # arrives 2.0-2.7 m high
            env.unwrapped._env._env.launch_ball(-25.0, 0.0, 0.11, 22.0, 0.0, vz, 0, 0, 0)
            prev = None
            winner = None
            for _ in range(40):
                obs, _, done, _ = env.step(a["builtin_ai"])
                o = obs[0]
                v = np.array(o["ball_direction"]) * [X_M, Y_M, 1.0] * 10
                if prev is not None and np.linalg.norm(v - prev) > 3.0:
                    ball = np.array(o["ball"][:2]) * [X_M, Y_M]
                    ours = np.linalg.norm(np.array(o["left_team"][1]) * [X_M, Y_M] - ball)
                    theirs = np.linalg.norm(np.array(o["right_team"][1]) * [X_M, Y_M] - ball)
                    if min(ours, theirs) < 2.5:
                        winner = 0 if ours < theirs else 1
                        break
                prev = v
                if done:
                    break
        finally:
            env.close()
        return {"we_win_it": float(winner == 0), "nobody": float(winner is None)}
    return run


def reach_test(stats: dict, seed: int) -> dict:
    """Our centre-back alone (game's AI); a ball flies over him 2.0-2.9 m high.
    Does he get his head to it?"""
    a = _names()
    height = stats.pop("_height", None)
    env = _env("stat_aerial_solo", seed, _overrides("left", 1, stats, height))
    try:
        env.reset()
        vz = 9.0 + 0.1 * (seed % 9)  # arrives about 2.0-2.9 m high
        env.unwrapped._env._env.launch_ball(-25.0, 0.0, 0.11, 22.0, 0.0, vz, 0, 0, 0)
        prev = None
        touched = False
        for _ in range(30):
            obs, _, done, _ = env.step(a["builtin_ai"])
            o = obs[0]
            v = np.array(o["ball_direction"]) * [X_M, Y_M, 1.0] * 10
            ball = np.array(o["ball"][:2]) * [X_M, Y_M]
            me = np.array(o["left_team"][1]) * [X_M, Y_M]
            if prev is not None and np.linalg.norm(v - prev) > 3.0 and np.linalg.norm(me - ball) < 2.5 \
                    and o["ball"][2] > 1.0:
                touched = True
                break
            prev = v
            if done:
                break
    finally:
        env.close()
    return {"reaches_it": float(touched)}


def header_test(stats: dict, seed: int) -> dict:
    """Our striker (scripted) heads a cross at goal from about 10 m: goal, keeper
    or off target. The cross is launched from the right wing at 2.0-2.6 m high
    where he stands; he presses shot when it is 3 m away."""
    a = _names()
    height = stats.pop("_height", None)
    env = _env("stat_header", seed, _overrides("left", 1, stats, height),
               extra={"kick_power_control": True})
    try:
        obs = env.reset()
        vz = 8.6 + 0.1 * (seed % 5)
        env.unwrapped._env._env.launch_ball(44.0, 22.0, 0.11, 0.0, -22.0, vz, 0, 0, 0)
        pressed = False
        goal = keeper = headed = False
        for _ in range(40):
            o = obs[0]
            ball = np.array(o["ball"][:2]) * [X_M, Y_M]
            me = np.array(o["left_team"][1]) * [X_M, Y_M]
            act = a["right"]
            if not pressed and np.linalg.norm(ball - me) < 3.0:
                act, pressed = (a["shot"], 0.5), True
            obs, _, done, info = env.step(act)
            o = obs[0]
            headed |= pressed and o["ball"][0] * X_M > me[0] + 1.0
            goal |= info["score_reward"] > 0
            keeper |= o["ball_owned_team"] == 1
            if done:
                break
    finally:
        env.close()
    return {"goal": float(goal), "keeper_has_it": float(keeper and not goal),
            "missed": float(not goal and not keeper)}


# name -> (what is varied, test, stat dict for one level)
def _group(names, level):
    return {n: level for n in names}


TESTS = {
    "pace": ("top speed (physical_velocity)", sprint_test, lambda v: _group(["physical_velocity"], v)),
    "acceleration": ("acceleration (physical_acceleration)", sprint_test,
                     lambda v: _group(["physical_acceleration"], v)),
    "shot": ("finishing (technical_shot)", shot_test, lambda v: _group(["technical_shot"], v)),
    "shot_power": ("shot power (physical_shotpower)", shot_test,
                   lambda v: _group(["physical_shotpower"], v)),
    "short_pass": ("short pass on the run (technical_shortpass)", pass_test("short_pass", "drill_pass_short"),
                   lambda v: _group(["technical_shortpass"], v)),
    "long_pass": ("long pass (technical_shortpass: the engine uses it for long passes too)",
                  pass_test("long_pass"), lambda v: _group(["technical_shortpass"], v)),
    "high_pass": ("high pass (technical_highpass)", pass_test("high_pass"),
                  lambda v: _group(["technical_highpass"], v)),
    "tackle": ("defender tackling (standing + sliding tackle)", duel_test("defender"),
               lambda v: _group(["technical_standingtackle", "technical_slidingtackle"], v)),
    "defender_balance": ("defender strength (physical_balance)", duel_test("defender"),
                         lambda v: _group(["physical_balance"], v)),
    "defender_reaction": ("defender reactions (physical_reaction)", duel_test("defender"),
                          lambda v: _group(["physical_reaction"], v)),
    "dribbling": ("attacker dribbling (dribble + ball control + agility)", duel_test("attacker"),
                  lambda v: _group(["technical_dribble", "technical_ballcontrol", "physical_agility"], v)),
    "attacker_balance": ("attacker strength (physical_balance)", duel_test("attacker"),
                         lambda v: _group(["physical_balance"], v)),
    "header": ("headed finish from a cross (technical_header)", header_test,
               lambda v: _group(["technical_header"], v)),
    "header_height": ("headed finish from a cross (height 1.70 m vs 1.95 m)", header_test,
                      lambda v: {"_height": 1.70 if v == LOW else 1.95}),
    "height": ("heading reach (height 1.70 m vs 1.95 m), ball 2.0-2.9 m high", reach_test,
               lambda v: {"_height": 1.70 if v == LOW else 1.95}),
}


def _run_chunk(name: str, level: float, seeds: list[int], switches: dict) -> list[dict]:
    PHYSICS.update(switches)
    _, test, stats_for = TESTS[name]
    return [test(dict(stats_for(level)), s) for s in seeds]


def run(name: str, seeds: int, workers: int, switches: dict | None = None) -> dict:
    ctx = multiprocessing.get_context("spawn")
    out = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        jobs = {}
        for level in (LOW, HIGH):
            for w in range(workers):
                chunk = list(range(w, seeds, workers))
                if chunk:
                    jobs[pool.submit(_run_chunk, name, level, chunk, switches or {})] = level
        rows = {LOW: [], HIGH: []}
        for f in concurrent.futures.as_completed(jobs):
            rows[jobs[f]].extend(f.result())
    for level, rs in rows.items():
        keys = rs[0].keys()
        out[level] = {k: (float(np.nanmean([r[k] for r in rs])),
                          float(np.nanstd([r[k] for r in rs]) / math.sqrt(len(rs)))) for k in keys}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", nargs="+", default=list(TESTS))
    parser.add_argument("--seeds", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--switch", action="append", default=[],
                        help="extra engine switch to turn on, e.g. --switch skill_stats")
    args = parser.parse_args()
    for name in args.tests:
        if name not in TESTS:
            sys.exit(f"unknown test {name}; choose from {', '.join(TESTS)}")
    for name in args.tests:
        result = run(name, args.seeds, args.workers, {s: True for s in args.switch})
        print(f"\n{name}: {TESTS[name][0]}, {args.seeds} runs per level, switches {args.switch}", flush=True)
        for k in result[LOW]:
            lo, lo_se = result[LOW][k]
            hi, hi_se = result[HIGH][k]
            print(f"  {k:16s} low {lo:7.3f} (±{lo_se:.3f})   high {hi:7.3f} (±{hi_se:.3f})", flush=True)


if __name__ == "__main__":
    main()
