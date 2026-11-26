"""outfield.env wrapper: end to end, replay, instructions, episode ends, parallel games."""

import hashlib

import numpy as np

from outfield.env import OutfieldEnv, VecEnv, random_actions
from outfield.env import instructions as ins


def _idle():
    z = np.zeros(11, dtype=int)
    return {"move": z.copy(), "speed": z.copy(), "ball": z.copy(), "defend": z.copy(),
            "power": np.zeros(11, np.float32)}


def _digest(env, steps, seed):
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)
    h = hashlib.blake2b(digest_size=8)
    for _ in range(steps):
        obs, _, term, trunc, _ = env.step(random_actions(obs["masks"], rng))
        h.update(np.round(env.raw_list[0]["ball"], 4).tobytes())
        h.update(obs["actor"].tobytes())
        if term or trunc:
            break
    return h.hexdigest()


def test_full_match_runs_and_replays():
    env = OutfieldEnv("11_vs_11_easy_stochastic", seed=4)
    try:
        rng = np.random.default_rng(0)
        obs = env.reset()
        for _ in range(3100):  # 3000 steps of play; the whistle comes on the step after
            obs, rewards, term, trunc, info = env.step(random_actions(obs["masks"], rng))
            assert set(rewards) == {"score"} and rewards["score"].shape == (11,)
            if term or trunc:
                break
        assert term and not trunc, "the final whistle of a full match is terminal"
        assert info["outcome"] in ("time", "goal_for", "goal_against")
        assert _digest(env, 300, seed=9) == _digest(env, 300, seed=9), "same seed must replay"
    finally:
        env.close()


def test_given_instructions_replace_the_sampler():
    env = OutfieldEnv("outfield_3v3", seed=1)
    try:
        env.reset()
        given = np.zeros((11, ins.MANAGER_SIZE), np.float32)
        plan = ins.Plan([ins.Instruction(role=5, slot=i, defending=2, attacking=2) for i in range(11)],
                        {"line_height": 0.7})
        present = np.array([True] * 4 + [False] * 7)
        given[:] = ins.encode(plan, present)
        obs, *_, info = env.step(_idle(), instructions=given)
        block = obs["actor"][:, 869:869 + ins.MANAGER_SIZE]
        np.testing.assert_allclose(block[:4], given[:4])
        assert not block[4:].any()
        assert info["plan"] == "given"
        assert ins.decode(block[1]).defending == 2
    finally:
        env.close()


def test_sampler_draws_v1_duties_and_presets():
    env = OutfieldEnv("11_vs_11_easy_stochastic", seed=1)
    try:
        env.reset()
        names = set()
        for seed in range(12):
            env.sampler.reset(seed)
            plan = env.sampler(0, env.raw_list[0]["left_team_roles"], env.formation,
                               env.formation, env.formation, env.present, env.present)
            names.add(plan.name)
            for i, p in enumerate(plan.players):
                if i == 0:
                    if plan.name == "random":
                        assert p.defending in ins.KEEPER_DEFENDING
                        assert p.attacking in ins.KEEPER_ATTACKING
                        if p.defending == 8:
                            assert p.line_height > 0, "sweep needs a line height"
                        if p.attacking == 1:
                            assert p.anchor[0] >= -52.0, "his spot is in front of the goal line"
                    continue
                assert p.defending in ins.V1_DEFENDING + (0,)
                assert p.attacking in ins.V1_ATTACKING + (0,)
                if p.defending == 5:
                    assert p.opp is not None
                if p.defending == 4:
                    assert p.mate is not None and p.mate != i
        assert "random" in names
    finally:
        env.close()


def test_drill_ends_terminated_or_truncated():
    # Our player walks onto the ball; the game's AI then plays it and scores
    # into the empty goal (terminal, +1). Loose balls stay the policy's job.
    env = OutfieldEnv("academy_empty_goal_close", seed=1, ai_on_ball=True)
    try:
        env.reset()
        a = _idle()
        a["move"][1] = 1
        for _ in range(400):
            _, rewards, term, trunc, info = env.step(a)
            if term or trunc:
                break
        assert term and info["outcome"] == "goal_for" and rewards["score"][1] == 1.0
    finally:
        env.close()
    # A drill only the clock can end: its time limit is a truncation.
    _register_quiet_drill()
    env = OutfieldEnv("outfield_test_quiet", seed=1, ai_on_ball=False)
    try:
        env.reset()
        for _ in range(100):
            _, _, term, trunc, info = env.step(_idle())
            if term or trunc:
                break
        assert trunc and not term and info["outcome"] == "time"
    finally:
        env.close()


def _register_quiet_drill():
    """Our keeper and one player on the ball at halfway, their keeper in goal,
    nobody moves, 3 s long."""
    import sys
    import types

    from gfootball.scenarios import Team, e_PlayerRole_CM, e_PlayerRole_GK

    def build(builder):
        config = builder.config()
        config.game_duration = 30
        config.deterministic = False
        config.end_episode_on_score = True
        config.end_episode_on_out_of_play = True
        config.end_episode_on_possession_change = True
        builder.SetBallPosition(0.0, 0.0)
        builder.SetTeam(Team.e_Left)
        builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
        builder.AddPlayer(-0.01, 0.0, e_PlayerRole_CM)
        builder.SetTeam(Team.e_Right)
        builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)

    module = types.ModuleType("gfootball.scenarios.outfield_test_quiet")
    module.build_scenario = build
    sys.modules[module.__name__] = module


def test_parallel_games():
    rng = np.random.default_rng(0)
    with VecEnv(2, scenario="outfield_3v3", seeds=[1, 2]) as venv:
        obs = venv.reset()
        assert obs["actor"].shape == (2, 11, 960)
        resets = 0
        for _ in range(300):
            obs, rewards, term, trunc, infos = venv.step(random_actions(obs["masks"], rng))
            assert rewards["score"].shape == (2, 11)
            resets += sum("final_obs" in i for i in infos)
        assert resets > 0, "finished games reset by themselves"
