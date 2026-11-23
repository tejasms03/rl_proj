"""Run lab scenes before/after an instruction and write the evidence.

    python -m grf_tools.duty_lab.run [press mark ...] [--seeds 1 2 3 4] [--out out/duty_lab]

Each test is full 11v11 from the same seed, twice: before (the tested players
are left to the game's AI) and after (they follow the instruction). "Base"
instructions apply in both runs (e.g. the presser a coverer backs up), so the
only difference is the tested instruction. Per test: compare.mp4 (before on
top), trace.png (top-down path of the focus player) and metrics.json; plus a
summary.md over all tests.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import shutil
import subprocess
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")

TICKS_PER_STEP = 4  # 40 ms of game per step: real-time video at 25 fps
FPS = 25
SECONDS = 15
SIZE = (960, 540)  # each half of the video
PASS_PCT = 70.0  # duty true this share of the time it applies (the drills' bar)
ENGAGED_M = 3.0  # a spell counts as contested once one of the defenders is this close
WON_WINDOW_S = 0.5  # the spell ended with us on the ball within this

_BACK_FOUR = [1, 2, 3, 4]
_LANES = dict(zip(_BACK_FOUR, [-0.27, -0.09, 0.09, 0.27]))


def _line(x, players=_BACK_FOUR):
    return {i: ("hold_line", {"line_x": x, "lane_y": _LANES[i], "group": players})
            for i in players}


# test -> scene, base (both runs) and tested (after only) duties as
# {player: (duty, params)}, and the focus player whose duty is measured.
# Duties that go after the ball list their "outcome" defenders: they are
# scored on how the opponent's spells on the ball end (see outcomes()), not on
# time in position, because their success ends the duty. Duties that deny
# one opponent the ball name him in "deny" and are scored on his receptions.
# In "attack" their midfielder brings the ball down our right-centre channel
# (y = +0.09), facing our centre-back 3 and midfielder 7.
TESTS = {
    "press": dict(scene="build_up", tested={9: ("press", {})}, focus=9, outcome=[9]),
    "contain": dict(scene="attack", opening="run_at_goal", tested={3: ("contain", {})}, focus=3,
                    outcome=[3]),
    "mark": dict(scene="attack", tested={3: ("mark", {"opp": 9})}, focus=3),
    "cover": dict(scene="attack", base={3: ("press", {})},
                  tested={2: ("cover", {"mate": 3})}, focus=2),
    # Their centre-back (2) on the ball, every other forward option marked, so
    # their midfielder 6 is the way out; scored on how often 6 gets the ball.
    "screen_lane": dict(scene="build_up",
                        base={9: ("press", {}), 8: ("mark", {"opp": 1}), 7: ("mark", {"opp": 5}),
                              6: ("mark", {"opp": 7}), 5: ("mark", {"opp": 4})},
                        tested={10: ("screen_lane", {"opp": 6})}, focus=10, deny=6),
    "hold_line": dict(scene="attack", tested=_line(-0.45), focus=2),
    "hold_zone": dict(scene="attack", tested={6: ("hold_zone", {"anchor": (-0.3, -0.05)})},
                      focus=6),
    "sweep": dict(scene="high_line", base=_line(-0.25),
                  tested={0: ("sweep", {"line_x": -0.25})}, focus=0),
    "double_team": dict(scene="attack", opening="run_at_goal",
                        tested={3: ("double_team", {"partner": 7, "role": "goal"}),
                                7: ("double_team", {"partner": 3, "role": "side"})}, focus=3,
                        outcome=[3, 7]),
    "shadow_press": dict(scene="build_up", tested={9: ("shadow_press", {"opp": 6})}, focus=9,
                         outcome=[9]),
    # Attacking duties: we have the ball; our player on it plays as the game's AI.
    # "receive" counts how often the tested player gets the ball (shown, not scored).
    "hold_shape": dict(scene="our_attack", opening="we_walk",
                       tested={7: ("hold_shape", {"anchor": (0.15, 0.09)})}, focus=7),
    # "serve": a scripted passer (see _Serve) gives the focus player the ball
    # when his duty's moment comes, in both runs; scored on passes reaching him.
    "support": dict(scene="our_build", opening="we_walk", tested={6: ("support", {})}, focus=6,
                    receive=6, serve="short_pass"),
    "width": dict(scene="our_attack", opening="we_walk", tested={8: ("width", {"side": "right"})},
                  focus=8, receive=8, serve="auto"),
    "run_in_behind": dict(scene="our_attack", opening="we_walk",
                          tested={10: ("run_in_behind", {"channel": 0.07})}, focus=10, receive=10,
                          serve="through_ball"),
    "drop_to_receive": dict(scene="our_attack", opening="we_walk",
                            tested={9: ("drop_to_receive", {})}, focus=9, receive=9,
                            serve="short_pass"),
    "rest_defence": dict(scene="our_attack", opening="we_walk",
                         tested={1: ("rest_defence", {"zone": (-0.2, -0.2)})}, focus=1),
    # Batch 2. Our winger (8) crosses from near their byline to the penalty spot.
    "box_arrival": dict(scene="our_wing", opening="to_byline",
                        tested={9: ("attack_zone", {"zone": (0.80, 0.03)})}, focus=9, receive=9,
                        serve=[{"to": 9, "kind": "high_pass", "from": 8, "ready": "cross",
                                "aim": (0.80, 0.03)}]),
    "overlap": dict(scene="our_wide", opening="hold_ball", tested={4: ("overlap", {"mate": 8})},
                    focus=4, receive=4, serve=[{"to": 4, "kind": "short_pass", "from": 8}]),
    "pin_fullback": dict(scene="our_attack", opening="we_walk",
                         tested={8: ("pin_fullback", {"opp": 1})}, focus=8),
    "inverted_fullback": dict(scene="our_attack", opening="we_walk",
                              tested={1: ("inverted_fullback", {"side": -1})}, focus=1),
    # Our midfielder (6) finds the striker coming short (9, a base duty), who
    # finds the second striker (10) running beyond him.
    "third_man_run": dict(scene="our_attack", opening="we_walk",
                          base={9: ("drop_to_receive", {})},
                          tested={10: ("third_man_run", {"mate": 9, "channel": 0.07})},
                          focus=10, receive=10,
                          serve=[{"to": 9, "kind": "short_pass", "ready": 9},
                                 {"to": 10, "kind": "through_ball", "from": 9}]),
}

# Opening rules, the same in both runs, then the game's AI plays on:
# - "walk" (default): their player nearest the ball walks it forward (patient
#   build-up) for up to HOLD_S, until one of ours is within HOLD_FREE_M.
#   Without it the AI clears long at once and there is nothing to defend.
# - "run_at_goal": their carrier runs at our goal for up to RUN_S, however
#   close we get (the situation contain and double team are for).
# - "we_walk": as "walk", for our player nearest the ball (attacking scenes),
#   so the off-ball runs have time to happen.
# - "to_byline": our player on the ball carries it up the pitch toward their
#   byline (up to CROSS_X or RUN_S), however close they get: the winger in the
#   crossing scene.
# - "hold_ball": our player on the ball stands and shields it for up to
#   HOLD_UP_S, however close they get: the winger holding up for an overlap.
HOLD_S = 4.0
HOLD_FREE_M = 5.0
HOLD_REACH_M = 3.0
RUN_S = 8.0
HOLD_UP_S = 4.0


def _opening(kind, step, view, duties):
    """(side, player, action) while the opening rule holds, else None."""
    ours_first = kind in ("we_walk", "to_byline", "hold_ball")
    holders, chasers = (("left_team", "right_team") if ours_first else ("right_team", "left_team"))
    gaps = [duties.metres(p, view["ball"]) for p in view[holders]]
    who = int(np.argmin(gaps))
    if view["ball_owned_team"] == (1 if ours_first else 0) or gaps[who] > HOLD_REACH_M:
        return None
    side = "left" if ours_first else "right"
    if kind == "hold_ball":
        if step >= HOLD_UP_S * FPS:
            return None
        return "left", who, duties.ACTION["release_direction"]
    if kind == "to_byline":
        if view["left_team"][who][0] >= CROSS_X or step >= RUN_S * FPS:
            return None
        return "left", who, duties.ACTION["right"]
    if kind == "run_at_goal":
        if step >= RUN_S * FPS:
            return None
        return side, who, duties.ACTION[duties.mirrored(
            duties.direction_to(view["right_team"][who], duties.OUR_GOAL))]
    nearest = min(duties.metres(p, view["ball"]) for p in view[chasers])
    if step >= HOLD_S * FPS or nearest <= HOLD_FREE_M:
        return None
    return side, who, duties.ACTION["right"]  # forward for both: theirs is mirrored


class _Serve:
    """A willing passer for the attacking tests. Each plan is a dict: "to"
    (receiver), "kind" (pass button, or "auto": ground pass up to SHORT_M,
    long ball beyond), optional "from" (only this passer), "aim" (a spot to
    play it toward instead of the receiver, e.g. a cross) and "ready" (a
    function of the view: is it the moment). When a plan's moment comes, our
    player on the ball turns toward the target and plays it; otherwise he
    stays the game's AI. At most one pass per plan per SERVE_GAP_S.

    The kick has a wind-up of several frames, and handing a player to the
    game's AI resets his controller, which cancels a pending kick: so the
    passer stays on manual ("idle") until the ball has left his foot."""

    SERVE_GAP_S = 3.0
    SHORT_M = 20.0
    WINDUP_S = 1.0  # give up on a kick that has not happened by then

    def __init__(self, plans, duties):
        self.plans, self.d = plans, duties
        self.passer, self.state, self.since, self.current = None, None, 0, None
        self.last = [-10_000] * len(plans)
        self.kicks: list[tuple[int, int, int]] = []  # (step, passer, receiver)

    def __call__(self, step, view, raw_owner, raw_player, left):
        d = self.d
        if self.state == "turn":  # facing the target: press the kick
            left[self.passer] = d.ACTION[self.chosen]
            self.kicks.append((step, self.passer, self.current["to"]))
            self.state, self.since = "windup", 0
            return
        if self.state == "windup":
            self.since += 1
            still_on = raw_owner == 0 and raw_player == self.passer
            if still_on and self.since < self.WINDUP_S * FPS:
                left[self.passer] = d.ACTION["idle"]  # stay manual until it is gone
            else:
                left[self.passer] = d.ACTION["release_direction"]
                self.state = None
            return
        if raw_owner != 0:
            return
        for n, plan in enumerate(self.plans):
            if (raw_player != plan["to"] and plan.get("from", raw_player) == raw_player
                    and step - self.last[n] >= self.SERVE_GAP_S * FPS and plan["ready"](view)):
                me = view["left_team"][raw_player]
                aim = plan.get("aim")
                target = view["left_team"][plan["to"]] if aim is None else np.asarray(aim)
                self.chosen = plan["kind"]
                if self.chosen == "auto":
                    self.chosen = "short_pass" if d.metres(me, target) <= self.SHORT_M else "long_pass"
                left[raw_player] = d.ACTION[d.direction_to(me, target)]
                self.passer, self.state, self.current, self.last[n] = raw_player, "turn", plan, step
                return


LOOSE_S = 1.0  # possession memory, see _Possession


class _Possession:
    """Who has the ball, as a player would say it. The engine reports nobody
    between a dribbler's touches; without memory every duty that follows the
    ball switches off and on several times a second. A loose ball counts as
    the last owner's for up to LOOSE_S."""

    def __init__(self):
        self.owner, self.loose = (-1, -1), 0

    def __call__(self, view) -> tuple[int, int]:
        if view["ball_owned_team"] != -1:
            self.owner, self.loose = (int(view["ball_owned_team"]), int(view["ball_owned_player"])), 0
        else:
            self.loose += 1
            if self.loose > LOOSE_S * FPS:
                self.owner = (-1, -1)
        return self.owner


def _play(test: str, instructed: bool, seed: int, record: bool, keep_frames: bool) -> dict:
    import gfootball.env as football_env

    from grf_tools.duty_lab import duties, scenes  # noqa: F401  (registers scenes)

    spec = TESTS[test]
    orders = dict(spec.get("base", {}))
    if instructed:
        orders.update(spec["tested"])
    # The focus duty is always built, so "before" is measured the same way.
    focus_name, focus_params = spec["tested"][spec["focus"]]
    focus = duties.DUTIES[focus_name](spec["focus"], **focus_params)
    players = {i: (focus if i == spec["focus"] and instructed else duties.DUTIES[n](i, **p))
               for i, (n, p) in orders.items()}

    env = football_env.create_environment(
        f"lab_{spec['scene']}", representation="raw", render=False,
        number_of_left_players_agent_controls=11, number_of_right_players_agent_controls=11,
        other_config_options={"action_set": "full", "physics_steps_per_frame": TICKS_PER_STEP,
                              "game_engine_random_seed": seed})
    log = {k: [] for k in ("left", "right", "ball", "owner", "raw_player", "carrier",
                           "our_carrier", "duty", "strict", "measure", "refs", "frames")}
    holding, last_held = True, None
    possession = _Possession()
    serve = None
    if spec.get("serve"):
        plans = spec["serve"]
        if isinstance(plans, str):  # shorthand: that pass to the focus player
            plans = [{"to": spec["focus"], "kind": plans}]
        resolved = []
        for plan in plans:
            plan = dict(plan)
            ready = plan.get("ready")
            if ready is None:  # the focus duty decides
                plan["ready"] = focus.serve_ready
            elif ready == "cross":  # the crosser is in the crossing area
                passer = plan["from"]
                plan["ready"] = lambda v, i=passer: v["left_team"][i][0] >= CROSS_X
            else:  # another player's duty decides (a base duty: in both runs)
                plan["ready"] = players[ready].serve_ready
            resolved.append(plan)
        serve = _Serve(resolved, duties)
    try:
        obs = env.reset()
        if record:
            env.render(mode="rgb_array")  # rendering changes the game: both runs must match
        for step in range(SECONDS * FPS):
            team, player = possession(obs[0])
            seen = [dict(o, ball_owned_team=team, ball_owned_player=player,
                         raw_ball_owned_team=o["ball_owned_team"]) for o in obs[:11]]
            view = seen[0]  # left-team view: absolute coordinates
            left = [duties.ACTION["builtin_ai"]] * 11
            for i, duty in players.items():
                left[i] = duty.control(seen[i])  # own view: sticky buttons are per player
            right = [duties.ACTION["builtin_ai"]] * 11
            if holding:
                acts = {"left": left, "right": right}
                rule = _opening(spec.get("opening", "walk"), step, view, duties)
                if last_held is not None and (rule is None or rule[:2] != last_held):
                    acts[last_held[0]][last_held[1]] = duties.ACTION["release_direction"]
                if rule is None:
                    holding = False
                else:
                    last_held = rule[:2]
                    acts[rule[0]][rule[1]] = rule[2]
            if serve is not None:  # after the opening rule: a pass ends it
                before = list(left)
                serve(step, view, int(obs[0]["ball_owned_team"]), int(obs[0]["ball_owned_player"]),
                      left)
                if left != before:
                    holding = False
            log["left"].append(view["left_team"].tolist())
            log["right"].append(view["right_team"].tolist())
            log["ball"].append(view["ball"][:2].tolist())
            log["owner"].append(int(obs[0]["ball_owned_team"]))  # raw, for regains
            log["raw_player"].append(int(obs[0]["ball_owned_player"]))
            log["carrier"].append(player if team == 1 else None)  # with possession memory
            log["our_carrier"].append(player if team == 0 else None)
            log["duty"].append(focus.check(view))
            log["strict"].append(focus.strict(view))
            log["measure"].append(focus.measure(view))
            log["refs"].append([np.asarray(r)[:2].tolist() for r in focus.refs(view)])
            obs, _, done, _ = env.step(left + right)
            if record:
                frame = env.render(mode="rgb_array")
                if keep_frames:
                    log["frames"].append(cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA))
            if done:
                break
        log["score"] = [int(s) for s in obs[0]["score"]]
    finally:
        env.close()
    log["kicks"] = serve.kicks if serve is not None else []
    log["focus"] = spec["focus"]
    log["instructed"] = sorted(orders) if instructed else sorted(spec.get("base", {}))
    return log


def outcomes(log: dict, defenders: list[int]) -> list[dict]:
    """Each spell of one of their players on the ball that one of our
    defenders got within ENGAGED_M of: did it end with us winning the ball
    (on it within WON_WINDOW_S), and how long after first contact."""
    from grf_tools.duty_lab.duties import metres

    found, start, close = [], None, None
    carriers = log["carrier"] + [None]
    for t, who in enumerate(carriers):
        prev = carriers[t - 1] if t else None
        if start is not None and who != prev:
            if close is not None:
                window = log["owner"][t:t + int(WON_WINDOW_S * FPS)]
                won = who is None and 0 in window
                found.append({"won": won, "to_win_s": (t - close) / FPS if won else None})
            start = close = None
        if who is not None and who != prev:
            start = t
        if start is not None and close is None and who is not None and min(
                metres(log["left"][t][d], log["right"][t][who]) for d in defenders) <= ENGAGED_M:
            close = t
    return found


SERVE_WINDOW_S = 2.5  # a served pass counts as reaching him if he has it within this
CROSS_X = 0.74  # "cross" plans: the crosser plays it once he is this far up (about 14 m out)


def served(log: dict, receiver: int) -> dict:
    """What happened to each scripted pass: reached him, went to another of
    ours, they won it, or the scene ended (out of play / offside)."""
    tally = {"played": 0, "reached": 0, "other": 0, "theirs": 0, "dead": 0}
    for k, passer, to in log["kicks"]:
        if to != receiver:
            continue
        # The kick has a wind-up: skip while the passer still has it.
        t0 = k
        while t0 < len(log["owner"]) and log["owner"][t0] == 0 and log["raw_player"][t0] == passer:
            t0 += 1
        if t0 - k > FPS:  # never let go: no pass happened
            continue
        tally["played"] += 1
        result = "dead"
        for t in range(t0, min(t0 + int(SERVE_WINDOW_S * FPS), len(log["owner"]))):
            if log["owner"][t] == 1:
                result = "theirs"
                break
            if log["owner"][t] == 0:
                result = "reached" if log["raw_player"][t] == receiver else "other"
                break
        tally[result] += 1
    return tally


def receptions(log: dict, player: int, key: str = "carrier") -> int:
    """Times a player got the ball (with possession memory): theirs by
    default, ours with key="our_carrier"."""
    c = log[key]
    return sum(1 for t in range(1, len(c)) if c[t] == player and c[t - 1] != player)


def metrics(log: dict, defenders: list[int] | None = None, deny: int | None = None,
            receive: int | None = None) -> dict:
    applies = [d for d in log["duty"] if d is not None]
    placed = [d for d in log["strict"] if d is not None]
    values = [m for m in log["measure"] if m is not None]
    regains, last = 0, None
    for o in log["owner"]:
        if o != -1:
            regains += last == 1 and o == 0
            last = o
    return {
        "seconds": round(len(log["owner"]) / FPS, 1),
        "applies_s": round(len(applies) / FPS, 1),
        "duty_true_pct": round(100 * sum(applies) / len(applies), 1) if applies else None,
        "in_position_pct": round(100 * sum(placed) / len(placed), 1) if placed else None,
        "measure_mean": round(float(np.mean(values)), 1) if values else None,
        "regains": regains,
        "goals_against": log["score"][1],
        **({"spells": outcomes(log, defenders)} if defenders else {}),
        **({"receptions": receptions(log, deny)} if deny is not None else {}),
        **({"got_ball": receptions(log, receive, "our_carrier")} if receive is not None else {}),
        **({"served": served(log, receive)} if log.get("kicks") is not None and receive is not None
           else {}),
    }


# --- pictures --------------------------------------------------------------

_PX = 8  # pixels per metre
_W, _H = int(108.8 * _PX), int(70.2 * _PX)
_BGR = dict(on=(30, 140, 255), off=(255, 200, 60), idle=(170, 170, 170), ref=(40, 40, 220))


def _to_px(p, w=_W, h=_H) -> tuple[int, int]:
    return int((p[0] + 1) / 2 * w), int((p[1] + 0.42) / 0.84 * h)


def _pitch() -> np.ndarray:
    img = np.full((_H + 60, _W, 3), (60, 120, 50), np.uint8)
    g = img[60:]
    white = (230, 230, 230)
    cv2.rectangle(g, (0, 0), (_W - 1, _H - 1), white, 2)
    cv2.line(g, (_W // 2, 0), (_W // 2, _H), white, 2)
    cv2.circle(g, (_W // 2, _H // 2), int(9.15 * _PX), white, 2)
    for x0 in (0, _W - int(16.5 * _PX)):
        cv2.rectangle(g, (x0, _H // 2 - int(20.15 * _PX)),
                      (x0 + int(16.5 * _PX), _H // 2 + int(20.15 * _PX)), white, 2)
    return img


def trace_panel(log: dict, title: str, stats: dict, unit: str) -> np.ndarray:
    """Focus player's path (orange = duty met, blue = not met, grey = doesn't
    apply), the ball (thin white) and what the duty is about (red dots)."""
    img = _pitch()
    g = img[60:]
    ball = [_to_px(b) for b in log["ball"]]
    for a, b in zip(ball, ball[1:]):
        cv2.line(g, a, b, (200, 200, 200), 1, cv2.LINE_AA)
    for refs in log["refs"][::3]:
        for r in refs:
            cv2.circle(g, _to_px(r), 2, _BGR["ref"], -1)
    me = [_to_px(p[log["focus"]]) for p in log["left"]]
    for i, (a, b) in enumerate(zip(me, me[1:])):
        on = log["duty"][i]
        colour = _BGR["on"] if on else _BGR["off"] if on is False else _BGR["idle"]
        cv2.line(g, a, b, colour, 3, cv2.LINE_AA)
    cv2.circle(g, me[0], 7, (255, 255, 255), -1)
    text = (f"{title}   duty met {stats['duty_true_pct']}% of {stats['applies_s']} s"
            f" | {unit}: {stats['measure_mean']} | regains {stats['regains']}"
            f" | goals against {stats['goals_against']}")
    cv2.putText(img, text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


_MAP = (320, 207)  # inset top-down map, bottom-right of each video frame


def _inset(out: np.ndarray, log: dict, i: int) -> None:
    w, h = _MAP
    x0, y0 = SIZE[0] - w - 12, SIZE[1] - h - 12
    box = out[y0:y0 + h, x0:x0 + w]
    box[:] = (box * 0.25 + np.array([40, 110, 40]) * 0.75).astype(np.uint8)
    px = lambda p: _to_px(p, w, h)  # noqa: E731
    cv2.rectangle(box, (0, 0), (w - 1, h - 1), (230, 230, 230), 1)
    cv2.line(box, (w // 2, 0), (w // 2, h), (230, 230, 230), 1)
    for p in log["right"][i]:
        cv2.circle(box, px(p), 4, (200, 40, 40), -1)
    for j, p in enumerate(log["left"][i]):
        big = j in log["instructed"] or j == log["focus"]
        cv2.circle(box, px(p), 7 if big else 4, (255, 150, 30) if big else (240, 240, 240), -1)
    for r in log["refs"][i]:
        cv2.circle(box, px(r), 9, (255, 60, 60), 2)
    cv2.circle(box, px(log["ball"][i]), 3, (255, 255, 0), -1)


def _caption(frame: np.ndarray, text: str, duty) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (SIZE[0], 36), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    if duty is not None:
        label, colour = ("DUTY MET", (60, 220, 60)) if duty else ("NOT MET", (230, 60, 60))
        cv2.putText(out, label, (SIZE[0] - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2,
                    cv2.LINE_AA)
    return out


def write_video(before: dict, after: dict, label: str, path: Path) -> None:
    n = max(len(before["frames"]), len(after["frames"]))
    ffmpeg = subprocess.Popen(
        [shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1] * 2}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)],
        stdin=subprocess.PIPE)
    for i in range(n):
        halves = []
        for log, text in ((before, "BEFORE: game AI decides"), (after, f"AFTER: {label}")):
            j = min(i, len(log["frames"]) - 1)
            half = _caption(log["frames"][j], text, log["duty"][j])
            _inset(half, log, j)
            halves.append(half)
        ffmpeg.stdin.write(np.ascontiguousarray(np.vstack(halves), dtype=np.uint8).tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def run(test: str, seeds: list[int], out: Path, pool, video: bool = True) -> dict:
    from grf_tools.duty_lab import duties

    spec = TESTS[test]
    duty = duties.DUTIES[spec["tested"][spec["focus"]][0]]
    out = out / test
    out.mkdir(parents=True, exist_ok=True)
    # Seeds[0] is rendered (for the video) and repeated: the repeat must match.
    # Rendering needs the display awake (macOS pauses it when the screen
    # sleeps); --no-video skips it, numbers only.
    first = video
    jobs = [(test, instructed, seed, first and seed == seeds[0], first and seed == seeds[0])
            for seed in seeds for instructed in (0, 1)]
    jobs.append((test, 1, seeds[0], first, False))
    logs = list(pool.map(_play, *zip(*jobs)))
    repeat = logs.pop()
    by_seed = {s: (logs[2 * i], logs[2 * i + 1]) for i, s in enumerate(seeds)}
    before, after = by_seed[seeds[0]]

    if video:
        write_video(before, after, duty.label, out / "compare.mp4")
    trace = np.vstack([trace_panel(before, "BEFORE", metrics(before), duty.unit),
                       trace_panel(after, "AFTER", metrics(after), duty.unit)])
    cv2.imwrite(str(out / "trace.png"), trace)  # colours above are BGR

    defenders, deny, receive = spec.get("outcome"), spec.get("deny"), spec.get("receive")
    per_seed = {s: {"before": metrics(b, defenders, deny, receive),
                    "after": metrics(a, defenders, deny, receive)}
                for s, (b, a) in by_seed.items()}

    def median(side, key):
        vals = [v[side][key] for v in per_seed.values() if v[side][key] is not None]
        return round(float(np.median(vals)), 1) if vals else None

    after_pct, before_pct = median("after", "duty_true_pct"), median("before", "duty_true_pct")
    report = {
        "test": test, "instruction": duty.label, "unit": duty.unit,
        "deterministic_repeat": repeat["left"] == after["left"] and repeat["ball"] == after["ball"],
        "median": {side: {k: median(side, k) for k in ("duty_true_pct", "in_position_pct",
                                                      "measure_mean", "regains", "goals_against")}
                   for side in ("before", "after")},
        "passes": bool(after_pct is not None and after_pct >= PASS_PCT
                       and (before_pct is None or after_pct > before_pct)),
        "seeds": per_seed,
    }
    if receive is not None:
        report["got_ball"] = {side: sum(v[side]["got_ball"] for v in per_seed.values())
                              for side in ("before", "after")}
    if spec.get("serve"):
        report["served"] = {side: {k: sum(v[side]["served"][k] for v in per_seed.values())
                                   for k in ("played", "reached", "other", "theirs", "dead")}
                            for side in ("before", "after")}
        b, a = report["served"]["before"], report["served"]["after"]
        rate = lambda t: t["reached"] / t["played"] if t["played"] else 0.0  # noqa: E731
        # Pass: the willing passer finds him more often than the game's AI
        # player in the same slot (or as often, at a better rate).
        report["passes"] = bool(a["reached"] > b["reached"] or (
            a["reached"] == b["reached"] > 0 and rate(a) > rate(b)))
    if deny is not None:
        report["denied"] = {side: {"receptions": sum(v[side]["receptions"] for v in per_seed.values()),
                                   "never": sum(v[side]["receptions"] == 0 for v in per_seed.values()),
                                   "seeds": len(per_seed)}
                            for side in ("before", "after")}
        # Pass: he gets the ball less often than when the game's AI plays.
        report["passes"] = (report["denied"]["after"]["receptions"]
                            < report["denied"]["before"]["receptions"])
    if defenders:
        report["outcome"] = {side: _totals([sp for v in per_seed.values() for sp in v[side]["spells"]])
                             for side in ("before", "after")}
        b, a = report["outcome"]["before"], report["outcome"]["after"]
        # Pass: wins the ball in at least as many contested spells as the
        # game's AI, and faster.
        report["passes"] = bool(a["won"] > 0 and a["won"] >= b["won"] and (
            b["to_win_s"] is None or (a["to_win_s"] is not None and a["to_win_s"] < b["to_win_s"])))
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def _totals(spells: list[dict]) -> dict:
    times = [sp["to_win_s"] for sp in spells if sp["won"]]
    return {"contested": len(spells), "won": len(times),
            "to_win_s": round(float(np.median(times)), 2) if times else None}


def summary(reports: list[dict], seeds: list[int]) -> str:
    rows = [f"Duty lab, 11v11, seeds {seeds}. Every test replays exactly from its seed "
            f"(last column).", ""]
    positional = [r for r in reports if "outcome" not in r and "denied" not in r
                  and "served" not in r]
    serving = [r for r in reports if "served" in r]
    winning = [r for r in reports if "outcome" in r]
    denying = [r for r in reports if "denied" in r]
    if positional:
        rows += [f"Positional duties, medians over seeds. \"Met\" = in position or closing on "
                 f"it at 1 m/s or more; \"in position\" alone is also shown. Pass: met at least "
                 f"{PASS_PCT:.0f}% of the time the duty applies, and more than before.", "",
                 "| Test | Instruction | Met: before → after | In position: before → after "
                 "| Measure: before → after | Got the ball (total) | Pass | Same replay |",
                 "|---|---|---|---|---|---|---|---|"]
        for r in positional:
            b, a = r["median"]["before"], r["median"]["after"]
            rows.append(f"| {r['test']} | {r['instruction']} | {b['duty_true_pct']}% → "
                        f"{a['duty_true_pct']}% | {b['in_position_pct']}% → "
                        f"{a['in_position_pct']}% | {b['measure_mean']} → {a['measure_mean']} "
                        f"{r['unit']} | "
                        + (f"{r['got_ball']['before']} → {r['got_ball']['after']}" if "got_ball" in r
                           else "–")
                        + f" | {'yes' if r['passes'] else 'NO'} | "
                        f"{'yes' if r['deterministic_repeat'] else 'NO'} |")
        rows.append("")
    if winning:
        rows += [f"Ball-winning duties, totals over seeds. A spell is contested once one of the "
                 f"defenders is within {ENGAGED_M:.0f} m of their player on the ball; won = we "
                 f"have it within {WON_WINDOW_S} s of the spell ending; time to win is from first "
                 f"contact (median). Pass: wins at least as many as the game's AI, and faster.", "",
                 "| Test | Instruction | Contested: before → after | Won: before → after "
                 "| Time to win: before → after | Pass | Same replay |",
                 "|---|---|---|---|---|---|---|"]
        for r in winning:
            b, a = r["outcome"]["before"], r["outcome"]["after"]
            rows.append(f"| {r['test']} | {r['instruction']} | {b['contested']} → {a['contested']} "
                        f"| {b['won']} → {a['won']} | {b['to_win_s']} → {a['to_win_s']} s | "
                        f"{'yes' if r['passes'] else 'NO'} | "
                        f"{'yes' if r['deterministic_repeat'] else 'NO'} |")
    if serving:
        rows += ["", "Attacking duties with a willing passer, totals over seeds: our player on "
                 "the ball plays it to the tested player when his duty's moment comes, the same "
                 "rule in both runs. Pass: more passes reach him than reach the game's AI player "
                 "in that slot.", "",
                 "| Test | Instruction | Passes played: before → after | Reached him "
                 "| Lost to them | Out / offside | In position: before → after | Pass "
                 "| Same replay |", "|---|---|---|---|---|---|---|---|---|"]
        for r in serving:
            b, a = r["served"]["before"], r["served"]["after"]
            mb, ma = r["median"]["before"], r["median"]["after"]
            rows.append(f"| {r['test']} | {r['instruction']} | {b['played']} → {a['played']} | "
                        f"{b['reached']} → {a['reached']} | {b['theirs']} → {a['theirs']} | "
                        f"{b['dead']} → {a['dead']} | {mb['in_position_pct']}% → "
                        f"{ma['in_position_pct']}% | {'yes' if r['passes'] else 'NO'} | "
                        f"{'yes' if r['deterministic_repeat'] else 'NO'} |")
    if denying:
        rows += ["", "Denying duties, totals over seeds: how often the screened player got the "
                 "ball. Pass: less often than when the game's AI plays.", "",
                 "| Test | Instruction | His receptions: before → after "
                 "| Seeds he never got it: before → after | In position: before → after "
                 "| Pass | Same replay |", "|---|---|---|---|---|---|---|"]
        for r in denying:
            b, a = r["denied"]["before"], r["denied"]["after"]
            mb, ma = r["median"]["before"], r["median"]["after"]
            rows.append(f"| {r['test']} | {r['instruction']} | {b['receptions']} → "
                        f"{a['receptions']} | {b['never']} → {a['never']} of {a['seeds']} | "
                        f"{mb['in_position_pct']}% → {ma['in_position_pct']}% | "
                        f"{'yes' if r['passes'] else 'NO'} | "
                        f"{'yes' if r['deterministic_repeat'] else 'NO'} |")
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tests", nargs="*", help=f"default: all of {list(TESTS)}")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--out", type=Path, default=Path("out/duty_lab"))
    parser.add_argument("--no-video", action="store_true",
                        help="numbers only: no rendering, works with the screen asleep")
    args = parser.parse_args()
    unknown = set(args.tests) - set(TESTS)
    if unknown:
        parser.error(f"unknown tests {sorted(unknown)}; choose from {list(TESTS)}")
    ctx = multiprocessing.get_context("spawn")
    reports = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6, mp_context=ctx) as pool:
        for test in args.tests or list(TESTS):
            reports.append(run(test, args.seeds, args.out, pool, video=not args.no_video))
            print(summary(reports[-1:], args.seeds).splitlines()[-1], flush=True)
    # The summary covers every test with results on disk, so rerunning one
    # test updates its row instead of replacing the table.
    done = {r["test"]: r for r in reports}
    for test in TESTS:
        saved = args.out / test / "metrics.json"
        if test not in done and saved.exists():
            done[test] = json.loads(saved.read_text())
    text = summary([done[t] for t in TESTS if t in done], args.seeds)
    (args.out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
