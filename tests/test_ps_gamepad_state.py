"""PlayStation mapping: which face-button set (kicks or defending) is live."""

import numpy as np

from gfootball.env.players import ps_gamepad as ps


def _player():
    player = object.__new__(ps.Player)  # no controller needed for _state
    player._last_owner = -1
    player._since_ours = ps.CONTEST_MEMORY_STEPS + 1
    return player


def _obs(owner, me, opponent, ball):
    return {"ball_owned_team": owner, "active": 0, "ball": np.array([*ball, 0.1]),
            "left_team": np.array([me]), "right_team": np.array([opponent])}


def test_their_dribble_stays_defend_between_touches():
    # Ownership blinks to -1 between an opponent's touches: ○ must stay
    # "pressure", not turn into a shot (that made holding ○ stop-start).
    player = _player()
    far = ((-0.5, 0.0), (-0.3, 0.0), (-0.3, 0.0))
    for owner in (1, -1, 1, -1, -1, 1):
        assert player._state(_obs(owner, *far)) == ps.DEFEND
    near = ((-0.31, 0.0), (-0.3, 0.0), (-0.3, 0.0))  # right on the dribbler
    for owner in (1, -1, 1):
        assert player._state(_obs(owner, *near)) == ps.DEFEND


def test_duel_after_our_possession_keeps_kicks():
    # Attacking in their box: a defender pokes at the ball (owner 1, then
    # loose). You can still shoot through the challenge.
    player = _player()
    duel = ((0.8, 0.0), (0.81, 0.0), (0.805, 0.0))
    assert player._state(_obs(0, *duel)) == ps.ATTACK
    assert player._state(_obs(1, *duel)) == ps.CONTESTED
    assert player._state(_obs(-1, *duel)) == ps.CONTESTED


def test_loose_ball_you_reach_first_is_yours():
    player = _player()
    player._state(_obs(1, (-0.5, 0.0), (-0.2, 0.0), (-0.2, 0.0)))
    # Their touch ran away from them, right to you.
    assert player._state(_obs(-1, (-0.5, 0.0), (-0.2, 0.0), (-0.48, 0.0))) == ps.ATTACK
    # Our own pass in flight: kicks (first-time shot / pass).
    player._state(_obs(0, (-0.5, 0.0), (-0.2, 0.0), (-0.5, 0.0)))
    assert player._state(_obs(-1, (-0.5, 0.0), (-0.2, 0.0), (-0.3, 0.0))) == ps.ATTACK
