"""outfield/scripted: the 3v3 scene and the scripted play run, and replay exactly."""

from outfield.scripted import play
from outfield.scripted.run import OUTCOMES, episode


def test_play_uses_v4_actions():
    for name in ("builtin_ai", "short_pass", "shot", "release_direction", "release_sprint"):
        assert name in play.ACTION
    assert len(play.STICKY) == 13  # v4: v3's 10 + keeper_rush, pressure, team_pressure


def test_scripted_episode_runs_and_replays():
    first = episode("scripted", seed=12)
    again = episode("scripted", seed=12)
    assert first["outcome"] in OUTCOMES
    assert first["digest"] == again["digest"], "the same seed must replay exactly"
