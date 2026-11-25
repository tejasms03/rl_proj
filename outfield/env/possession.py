"""Who has the ball, as a player would say it.

Between a dribbler's touches the engine reports nobody on the ball; without
a short memory every rule that follows the ball (masks, duty checks, the
memory block) switches off and on several times a second (docs/CONTEXT.md
§7.1). A loose ball counts as the last owner's for up to LOOSE_S.
"""

from __future__ import annotations

from outfield.config import STEP_S

LOOSE_S = 1.0


class Possession:
    def __init__(self):
        self.owner, self.loose = (-1, -1), 0
        self.changed_steps = 0  # steps since the owning team last changed

    def reset(self) -> None:
        self.owner, self.loose, self.changed_steps = (-1, -1), 0, 0

    def __call__(self, raw) -> tuple[int, int]:
        """Update from one raw observation; return (team, player): team 0 us,
        1 them, -1 nobody (loose for longer than LOOSE_S)."""
        before = self.owner[0]
        if raw["ball_owned_team"] != -1:
            self.owner, self.loose = (int(raw["ball_owned_team"]), int(raw["ball_owned_player"])), 0
        else:
            self.loose += 1
            if self.loose * STEP_S > LOOSE_S:
                self.owner = (-1, -1)
        team = self.owner[0]
        if team != -1 and team != before:
            self.changed_steps = 0
        else:
            self.changed_steps += 1
        return self.owner

    @property
    def team(self) -> int:
        return self.owner[0]
