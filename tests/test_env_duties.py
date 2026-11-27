"""outfield.env duty checks: each v1 duty, T2 recover, the keeper's duties and
plug-in, the NaN rules, the closing-in rule, and the checks through the wrapper."""

import math

import numpy as np
import pytest

from grf_tools.duty_lab import duties as lab
from outfield.env import OutfieldEnv
from outfield.env import duties as du
from outfield.env import instructions as ins

M = np.array([54.4, 83.6])  # raw -> metres (y down, the lab's frame)


def view(us=None, them=None, ball=(0.0, 0.0), owner=(1, 1), us_dir=None, them_dir=None,
         ball_dir=(0.0, 0.0)):
    """A left-team view in raw units. us / them: {slot: (x, y)}; the rest of
    our players stand far away on the halfway line edge."""
    left = np.tile([0.0, 0.41], (11, 1))
    right = np.tile([0.9, 0.41], (11, 1))
    for k, p in (us or {}).items():
        left[k] = p
    for k, p in (them or {}).items():
        right[k] = p
    ld, rd = np.zeros((11, 3)), np.zeros((11, 3))
    for k, d in (us_dir or {}).items():
        ld[k, :2] = d
    for k, d in (them_dir or {}).items():
        rd[k, :2] = d
    return {"left_team": left, "right_team": right, "left_team_direction": ld,
            "right_team_direction": rd, "ball": np.array([*ball, 0.1]),
            "ball_direction": np.array([*ball_dir, 0.0]),
            "ball_owned_team": owner[0], "ball_owned_player": owner[1]}


def blocks(*players, present=None):
    """[11, 85] from Instructions placed at their slots."""
    row = [ins.Instruction(role=5, slot=i) for i in range(11)]
    for p in players:
        row[p.slot] = p
    present = np.ones(11, bool) if present is None else present
    return ins.encode(ins.Plan(row, {}), present), present


FORMATION = np.zeros((11, 2))


def judge(v, b, present, checker=None, formation=FORMATION):
    checker = checker or du.DutyChecker()
    return checker(v, b, present, formation)


def ms_per_step(mps):
    """A *_direction entry for a speed in m/s along +x (raw units per 0.1 s step)."""
    return mps * 0.1 / M[0]


# --- one duty at a time --------------------------------------------------------------------

def test_hold_zone_uses_the_blocks_anchor_and_radius():
    p = ins.Instruction(role=1, slot=3, defending=1, anchor=(-30.0, 10.0), radius_m=6.0)
    b, pres = blocks(p)
    # Ball at the anchor: no shift, so the centre is the anchor itself.
    a_raw = du.to_raw(p.anchor)
    ok, strict, active, target = judge(view(us={3: a_raw + [0.05, 0]}, ball=a_raw), b, pres)
    assert ok[3] == 1.0 and du.DUTY_CODES[active[3]] == "D1"
    np.testing.assert_allclose(target[3], p.anchor, atol=1e-3)
    far = judge(view(us={3: a_raw + [0.2, 0]}, ball=a_raw), b, pres)[0]
    assert far[3] == 0.0, "10.9 m out of a 6 m zone"


def test_press_range_grows_with_intensity():
    me, him = np.array([0.0, 0.0]), np.array([20.0 / M[0], 0.0])  # 20 m apart
    for intensity, applies in ((0.0, False), (0.4, False), (0.6, True), (1.0, True)):
        p = ins.Instruction(role=5, slot=4, defending=2, press_intensity=intensity)
        b, pres = blocks(p)
        ok = judge(view(us={4: me}, them={1: him}, ball=him, owner=(1, 1)), b, pres)[0]
        assert np.isnan(ok[4]) != applies, (intensity, 10 + 20 * intensity)


def test_mark_and_cover_targets_come_from_the_block():
    man, mate = np.array([-0.5, 0.0]), np.array([-0.3, 0.1])
    mark = ins.Instruction(role=1, slot=2, defending=5, opp=7)
    cover = ins.Instruction(role=1, slot=3, defending=4, mate=5)
    b, pres = blocks(mark, cover)
    v = view(us={2: man - [2.0 / M[0], 0], 3: lab.goal_side_of(mate, 7.0), 5: mate},
             them={7: man, 1: (0.2, 0.0)}, ball=(0.2, 0.0))
    ok, *_ = judge(v, b, pres)
    assert ok[2] == 1.0 and ok[3] == 1.0
    v["left_team"][2] = man + [2.0 / M[0], 0]  # 2 m on the wrong side of his man
    assert judge(v, b, pres)[0][2] == 0.0


def test_mark_leads_a_running_man_with_the_x10_velocity():
    mark = ins.Instruction(role=1, slot=2, defending=5, opp=7)
    b, pres = blocks(mark)
    man = np.array([0.0, 0.0])
    v = view(us={2: (-10.0 / M[0], 0.0)}, them={7: man, 1: (0.3, 0.0)}, ball=(0.3, 0.0),
             them_dir={7: (-ms_per_step(5.0), 0.0)})  # running at our goal at 5 m/s
    target = judge(v, b, pres)[3][2]
    # 10 m away -> lead of min(10 / 7.5, 1.0) = 1 s -> 5 m, then 1.5 m goal-side.
    assert target[0] == pytest.approx(-6.5, abs=0.2)


def test_hold_line_height_maps_to_12_to_52_m_from_goal():
    assert du.line_x(0.0) * M[0] == pytest.approx(-54.4 + 12)
    assert du.line_x(1.0) * M[0] == pytest.approx(-54.4 + 52)
    lh = 0.45  # 30 m from goal, the lab's tested line
    line = [ins.Instruction(role=1, slot=k, defending=7, line_height=lh) for k in (2, 3, 4)]
    b, pres = blocks(*line)
    x = du.line_x(lh)
    ok = judge(view(us={2: (x, -0.1), 3: (x, 0.0), 4: (x + 0.5 / M[0], 0.1)}, ball=(0.3, 0)), b, pres)[0]
    assert ok[2] == ok[3] == ok[4] == 1.0
    ok = judge(view(us={2: (x, -0.1), 3: (x, 0.0), 4: (x + 6 / M[0], 0.1)}, ball=(0.3, 0)), b, pres)[0]
    assert ok[4] == 0.0, "6 m ahead of the line"


def test_attacking_duties():
    supp = ins.Instruction(role=6, slot=6, attacking=2)
    wide_l = ins.Instruction(role=6, slot=7, attacking=3, anchor=(0.0, 30.0))   # +y up = left
    wide_r = ins.Instruction(role=6, slot=8, attacking=3, anchor=(0.0, -30.0))
    run = ins.Instruction(role=9, slot=9, attacking=4, anchor=(20.0, 0.0))
    b, pres = blocks(supp, wide_l, wide_r, run)
    carrier = np.array([0.0, 0.0])
    them = {k: (0.5 - 0.02 * k, 0.41) for k in range(1, 11)}  # their last line at x = 0.48
    them.update({0: (1.0, 0.0), 1: (0.5, 0.2), 2: (0.5, -0.2)})
    v = view(us={5: carrier, 6: carrier + [11 / M[0], 0], 7: (0.1, -0.40), 8: (0.1, 0.40),
                 9: (0.5 - 2 / M[0], 0.0)}, them=them, ball=carrier, owner=(0, 5))
    ok, _, active, _ = judge(v, b, pres)
    assert [du.DUTY_CODES[active[k]] for k in (6, 7, 8, 9)] == ["A2", "A3", "A3", "A4"]
    assert ok[6] == ok[7] == ok[8] == ok[9] == 1.0
    assert np.isnan(ok[5]), "the player on the ball has no duty"
    v["left_team"][7] = (0.1, 0.40)  # left winger on the right touchline
    assert judge(v, b, pres)[0][7] == 0.0
    # Beyond the line: only a forward run of > 3 m/s counts (x10 velocity).
    v["left_team"][9] = (0.5 + 3 / M[0], 0.0)
    assert judge(v, b, pres)[0][9] == 0.0
    v["left_team_direction"][9, 0] = ms_per_step(4.0)
    assert judge(v, b, pres)[0][9] == 1.0


def test_recover_is_strict():
    rec = ins.Instruction(role=1, slot=3, defending=1, transition=2, anchor=(-30.0, 0.0), radius_m=6)
    b, pres = blocks(rec)
    checker = du.DutyChecker()
    me = np.array([0.2, 0.0])
    ours = view(us={3: me}, ball=(0.3, 0.0), owner=(0, 5))
    judge(ours, b, pres, checker)                     # we have it...
    lost = view(us={3: me}, ball=(0.3, 0.0), owner=(1, 1))
    ok, _, active, _ = judge(lost, b, pres, checker)  # ...and lose it
    assert du.DUTY_CODES[active[3]] == "T2" and ok[3] == 0.0, "standing still"
    lost["left_team_direction"][3, 0] = -ms_per_step(3.0)
    assert judge(lost, b, pres, checker)[0][3] == 0.0, "a jog back fails"
    lost["left_team_direction"][3, 0] = -ms_per_step(6.5)
    assert judge(lost, b, pres, checker)[0][3] == 1.0, "a sprint back passes"
    lost["left_team_direction"][3, 0] = ms_per_step(6.5)
    assert judge(lost, b, pres, checker)[0][3] == 0.0, "sprinting the wrong way"
    for _ in range(50):  # after 5 s he falls into his defending duty
        _, _, active, _ = judge(lost, b, pres, checker)
    assert du.DUTY_CODES[active[3]] == "D1"


# --- the keeper ------------------------------------------------------------------------------

def test_keeper_hold_zone_between_the_posts():
    gk = ins.Instruction(role=0, slot=0, defending=1, anchor=ins.KEEPER_ZONE,
                         radius_m=ins.KEEPER_ZONE_RADIUS_M)
    b, pres = blocks(gk)
    ok = judge(view(us={0: (-0.98, 0.03)}, ball=(0.5, 0.3)), b, pres)[0]
    assert ok[0] == 1.0, "no shift toward the ball: 1.1 m off the line, 2.5 m off centre"
    ok = judge(view(us={0: (-0.98, 0.06)}, ball=(0.5, 0.3)), b, pres)[0]
    assert ok[0] == 0.0, "5 m wide of the centre: outside the posts"


def test_keeper_sweep_and_support():
    sweep = ins.Instruction(role=0, slot=0, defending=8, line_height=0.6)
    b, pres = blocks(sweep)
    line = du.line_x(0.6)
    ok = judge(view(us={0: (-1 + 12 / M[0], 0.0)}, ball=(0.0, 0.0)), b, pres)[0]
    assert ok[0] == 1.0 and -1 + 12 / M[0] < line
    assert judge(view(us={0: (-1 + 3 / M[0], 0.0)}, ball=(0.0, 0.0)), b, pres)[0][0] == 0.0
    supp = ins.Instruction(role=0, slot=0, attacking=2)
    b, pres = blocks(supp)
    carrier = np.array([-0.55, 0.1])
    v = view(us={0: (-0.85, 0.0), 2: carrier}, ball=carrier, owner=(0, 2))
    ok, _, active, _ = judge(v, b, pres)
    assert du.DUTY_CODES[active[0]] == "A2" and ok[0] == 1.0
    v["right_team"][3] = lab.lane_point((-0.7, 0.05), carrier, (-0.85, 0.0))  # someone in the lane
    assert judge(v, b, pres)[0][0] == 0.0
    v["left_team"][2] = v["ball"][:2] = (0.3, 0.0)  # carrier in their half: no back pass
    assert np.isnan(judge(v, b, pres)[0][0])


def test_keeper_plug_in_and_the_true_false_contract():
    gk = ins.Instruction(role=0, slot=0, defending=0)
    b, pres = blocks(gk)
    calls = []

    def stub(answer):
        def fn(v, slot, instruction):
            calls.append((slot, instruction.defending))
            return answer
        return fn

    v = view(us={0: (-0.98, 0.0)}, ball=(-0.5, 0.0), owner=(1, 1))
    for answer, want in ((True, 1.0), (False, 0.0), (None, np.nan)):
        ok = judge(v, b, pres, du.DutyChecker(keeper_check=stub(answer)))[0]
        np.testing.assert_equal(ok[0], np.float32(want))
    assert calls and all(c == (0, 0) for c in calls)
    calls.clear()
    judge(dict(v, ball_owned_team=0, ball_owned_player=2), b, pres, du.DutyChecker(keeper_check=stub(True)))
    assert not calls, "only while they have the ball"
    # G's reward reads duty_ok[0] == True: only 1.0 passes, NaN and 0.0 don't.
    assert np.float32(1.0) == True  # noqa: E712
    assert not (np.float32(0.0) == True) and not (np.float32(np.nan) == True)  # noqa: E712
    assert bool(np.float32(np.nan)), "why `if duty_ok[0]:` is wrong: NaN is truthy"


# --- the rules ------------------------------------------------------------------------------

def test_nan_rules_and_closing_in():
    p = ins.Instruction(role=1, slot=3, defending=1, anchor=(-30.0, 0.0), radius_m=5.0)
    b, pres = blocks(p, present=np.array([True] * 4 + [False] * 7))
    v = view(us={3: (0.0, 0.0)}, ball=du.to_raw((-30.0, 0.0)))
    checker = du.DutyChecker()
    ok, strict, active, target = judge(v, b, pres, checker)
    assert np.isnan(ok[4:]).all() and (active[4:] == -1).all() and np.isnan(target[4:]).all()
    assert np.isnan(ok[1]) and np.isnan(ok[2]), "D0: nothing to judge"
    assert ok[3] == 0.0
    for k in range(1, 7):  # walking at the zone at 1.5 m/s: closing in counts once 0.5 s is seen
        v["left_team"][3, 0] -= ms_per_step(1.5)
        ok, strict, *_ = judge(v, b, pres, checker)
    assert ok[3] == 1.0 and strict[3] == 0.0, "closing in: met, but not in position"
    p2 = ins.Instruction(role=1, slot=3, defending=1, anchor=(-40.0, 0.0), radius_m=5.0)
    b2, _ = blocks(p2, present=pres)
    ok = judge(v, b2, pres, checker)[0]
    assert ok[3] == 0.0, "a new anchor restarts the closing history"
    v["ball_owned_team"] = 0
    assert np.isnan(judge(v, b, pres, checker)[0][3]), "no attacking duty given"


def test_the_ported_checks_agree_with_the_lab():
    rng = np.random.default_rng(0)
    for _ in range(300):
        us = {k: rng.uniform([-1, -0.42], [1, 0.42]) for k in range(11)}
        them = {k: rng.uniform([-1, -0.42], [1, 0.42]) for k in range(11)}
        ball = rng.uniform([-1, -0.42], [1, 0.42])
        v = view(us=us, them=them, ball=ball, owner=(1, 4))
        anchor = rng.uniform([-40, -25], [20, 25])
        p = ins.Instruction(role=1, slot=3, defending=1, anchor=tuple(anchor), radius_m=7.0)
        assert du.HoldZone().ok(v, 3, p) == lab.HoldZone(3, anchor=du.to_raw(anchor), radius_m=7.0).ok(v)
        p = ins.Instruction(role=1, slot=3, defending=5, opp=6)
        assert du.Mark().ok(v, 3, p) == lab.Mark(3, opp=6).ok(v)
        p = ins.Instruction(role=1, slot=3, defending=4, mate=5)
        assert du.Cover().ok(v, 3, p) == lab.Cover(3, mate=5).ok(v)
        assert du.Press().ok(v, 3, p) == lab.Press(3).ok(v)
        p = ins.Instruction(role=0, slot=0, defending=8, line_height=0.5)
        assert du.Sweep().ok(v, 0, p) == lab.Sweep(0, line_x=du.line_x(0.5)).ok(v)
        vo = dict(v, ball_owned_team=0, ball_owned_player=5)
        assert du.Support().ok(vo, 3, p) == lab.Support(3).ok(vo)


# --- through the wrapper ---------------------------------------------------------------------

def _go_to(info, present, sprint_m=4.0):
    from outfield.env import actions as act
    z = np.zeros(11, dtype=int)
    a = {"move": z.copy(), "speed": z.copy(), "ball": z.copy(), "defend": z.copy(),
         "power": np.zeros(11, np.float32)}
    if info is None:
        return a
    for i in range(1, 11):
        t = info["duty_target"][i]
        if not present[i] or np.isnan(t).any():
            continue
        me = np.asarray(info["raw"][0]["left_team"][i][:2]) * np.array([54.4, -83.6])
        d = t - me
        if np.linalg.norm(d) > 0.8:
            a["move"][i] = act.direction_index(*d)
            a["speed"][i] = int(np.linalg.norm(d) > sprint_m)
    return a


def _share(policy):
    env = OutfieldEnv("11_vs_11_easy_stochastic", seed=5, sampler=None)
    try:
        env.reset()
        plan = ins.Plan([ins.Instruction(role=0, slot=0)] + [
            ins.Instruction(role=5, slot=i, defending=1, attacking=1,
                            anchor=tuple(env.formation[i] + [8.0, 0.0]), radius_m=6.0)
            for i in range(1, 11)], {})
        given = ins.encode(plan, env.present)
        info, met = None, []
        for _ in range(600):
            obs, _, term, trunc, info = env.step(_go_to(info if policy == "go_to" else None, env.present),
                                                 instructions=given)
            met.extend(v for v in info["duty_ok"][1:] if not math.isnan(v))
            if term or trunc:
                break
        return float(np.mean(met))
    finally:
        env.close()


def test_hold_zone_and_shape_are_carried_out_through_the_wrapper():
    # Anchors 8 m ahead of kick-off: going to the target meets the duty (stage-0
    # bar, 70 %), standing still clearly doesn't.
    go, idle = _share("go_to"), _share("idle")
    assert go >= 0.7, go
    assert idle < go - 0.15, (idle, go)


def test_info_fields_and_keeper_plug_in_in_the_env():
    seen = []

    env = OutfieldEnv("outfield_3v3", seed=1, keeper_check=_always_true)
    try:
        obs = env.reset()
        z = np.zeros(11, dtype=int)
        idle = {"move": z, "speed": z, "ball": z, "defend": z, "power": np.zeros(11, np.float32)}
        _, _, _, _, info = env.step(idle)
        assert info["duty_ok"].dtype == np.float32 and info["duty_ok"].shape == (11,)
        assert info["duty_strict"].shape == (11,) and info["duty_active"].shape == (11,)
        assert info["duty_target"].shape == (11, 2)
        assert np.isnan(info["duty_ok"][4:]).all(), "padded slots"
        seen.append(info["duty_active"])
    finally:
        env.close()


def _always_true(view, slot, instruction):
    return True
