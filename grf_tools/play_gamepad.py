"""Play against the built-in AI with a PlayStation controller.

    ./play.sh                                  # 11 vs 11 against the AI
    ./play.sh --level 11_vs_11_hard_stochastic
    ./play.sh --level academy_3_vs_1_with_keeper
    ./play.sh --test                           # show which buttons your pad reports

You control the active player (switches automatically, or L1 when
defending); the AI plays everyone else. Analog movement, input read 50x a
second. FIFA-style layout; hold a kick to charge power (aim while charging):
  attacking (also in a duel for the ball at your feet):
              X short pass   O shot   [] lofted pass   /\\ through ball
              R1+O curl shot  L1+O chip  R1+[] driven long ball
  defending:  O (hold) tackle   [] slide tackle   X (hold) teammate press
              /\\ keeper rush   L1 switch player
  any time:   R2 sprint   L2 close control   left stick / D-pad move
              SHARE instant replay (last 8 s); goals replay automatically
              (OPTIONS skips a replay)
"""

from __future__ import annotations

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import pygame  # noqa: E402

import collections  # noqa: E402

from gfootball.env import config as gconfig  # noqa: E402
from gfootball.env import football_env  # noqa: E402

V1 = {
    "realistic_player_physics": True,
    "realistic_ball_physics": True,
    "ball_pace": "game",
    "situational_accuracy": True,
    "kick_power_control": True,
    "pass_direction_assist": 1.0,  # aim snaps toward the teammate you point at
}


def test_controller() -> None:
    """Print button / axis numbers as you press them (Ctrl-C to stop)."""
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No controller found. Connect it (USB or Bluetooth) and try again.")
        return
    pad = pygame.joystick.Joystick(0)
    pad.init()
    print(f"{pad.get_name()}: {pad.get_numbuttons()} buttons, {pad.get_numaxes()} axes")
    print("Press buttons / move sticks and triggers. Ctrl-C to stop.")
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    print(f"button {event.button}")
                elif event.type == pygame.JOYAXISMOTION and abs(event.value) > 0.6:
                    print(f"axis {event.axis}: {event.value:+.2f}")
            pygame.time.wait(10)
    except KeyboardInterrupt:
        pass


class Replays:
    """Goal and instant replays. Every second the match state is snapshotted
    (about 0.2 ms each) and every step the controller's action is logged; a
    replay restores a snapshot, re-plays the logged actions (the engine is
    deterministic given the state and inputs), then restores live play."""

    SNAPSHOT_EVERY = 50   # steps (1 s at 50 Hz)
    KEEP_SECONDS = 15
    SHARE, OPTIONS = 4, 6  # PS controller buttons

    def __init__(self, env, step_seconds, length_s=8.0, slow=0.75):
        self.env, self.step_seconds = env, step_seconds
        self.length_steps = int(length_s * self.SNAPSHOT_EVERY)
        self.slow = slow
        # No players when spectating (the AI runs both teams): nothing to log.
        self.player = env._players[0] if env._players else None
        if self.player:
            self._take_action = self.player.take_action
            self.player.take_action = self._recording
        self.t = 0
        keep = self.KEEP_SECONDS * self.SNAPSHOT_EVERY
        self.actions = collections.deque(maxlen=keep)       # (t, action)
        self.snapshots = collections.deque(maxlen=self.KEEP_SECONDS)  # (t, state)
        self._share_was_down = False

    def _recording(self, observations):
        action = self._take_action(observations)
        self.actions.append((self.t, action))
        return action

    def _button(self, index):
        joystick = getattr(self.player, "_joystick", None)
        return bool(joystick and index < joystick.get_numbuttons() and joystick.get_button(index))

    def before_step(self):
        if self.t % self.SNAPSHOT_EVERY == 0:
            self.snapshots.append((self.t, self.env.get_state()))

    def after_step(self, done):
        self.t += 1
        if done:  # new match: old snapshots belong to the previous one
            self.actions.clear()
            self.snapshots.clear()

    def share_pressed(self):
        down = self._button(self.SHARE)
        pressed = down and not self._share_was_down
        self._share_was_down = down
        return pressed

    def play(self, end_t=None, label="REPLAY"):
        """Replay the length_steps before end_t (default: now)."""
        end_t = self.t if end_t is None else end_t
        start = [(t, st) for t, st in self.snapshots if t <= end_t - self.length_steps]
        start = start[-1] if start else (self.snapshots[0] if self.snapshots else None)
        if start is None:
            return
        logged = dict(self.actions)
        if self.player is None:  # spectating: the AI replays itself
            logged = {t: None for t in range(start[0], self.t)}
        if start[0] not in logged:
            return
        print(f"{label} (OPTIONS to skip)", flush=True)
        live = self.env.get_state()
        self.env.set_state(start[1])
        t = start[0]
        if self.player:
            self.player.take_action = lambda observations: logged[t]
        next_step = time.perf_counter()
        try:
            while t < min(end_t, self.t) and t in logged:
                if self._button(self.OPTIONS):
                    break
                self.env.step([])
                t += 1
                next_step += self.step_seconds / self.slow
                delay = next_step - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_step = time.perf_counter()
        finally:
            self.env.set_state(live)
            if self.player:
                self.player.take_action = self._recording
            while self._button(self.OPTIONS):  # don't let the skip press leak into play
                time.sleep(0.01)
                pygame.event.pump()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", default="11_vs_11_stochastic")
    parser.add_argument("--pitch", default="normal", choices=["normal", "dry", "wet", "long_grass"])
    parser.add_argument("--power-assist", type=float, default=0.5,
                        help="0 = your charge sets the distance, 1 = fully automatic")
    parser.add_argument("--upstream", action="store_true", help="original GRF physics")
    parser.add_argument("--windup", type=float, default=2.0,
                        help="kick wind-up speed (v1.1): 1 = original GRF timing, 2 = twice as snappy")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed: 1.0 = real time, 0.8 = 20%% slower "
                             "(display only; physics unchanged)")
    parser.add_argument("--no-magnet", action="store_true",
                        help="no assisted steering toward the ball: pure stick control")
    parser.add_argument("--home", default="FC Barcelona", help="your team's name")
    parser.add_argument("--away", default="Real Madrid", help="the AI team's name")
    parser.add_argument("--home-short", default="FCB", help="3-letter scoreboard name")
    parser.add_argument("--away-short", default="RMA")
    parser.add_argument("--home-players", help="comma-separated names in formation order "
                        "(goalkeeper first), e.g. \"Ter Stegen,Kounde,...\"; blanks keep defaults")
    parser.add_argument("--away-players")
    parser.add_argument("--no-injury-time", action="store_true",
                        help="no half time / added time: one continuous 90 minutes")
    parser.add_argument("--no-replays", action="store_true", help="no goal / instant replays")
    parser.add_argument("--test", action="store_true", help="check controller buttons")
    args = parser.parse_args()
    if args.test:
        test_controller()
        return

    pygame.joystick.init()
    spectate = pygame.joystick.get_count() == 0
    if spectate:
        print("No controller found (pair or plug it in, then restart).\n"
              "Spectator mode: the AI plays both teams.", flush=True)
    values = {
        "action_set": "full",
        "players": [] if spectate else ["ps_gamepad:left_players=1"],
        "real_time": True,
        "level": args.level,
        # Read the controller 50 times a second (default 10); physics is
        # unchanged, the engine still runs at 100 ticks per second.
        "physics_steps_per_frame": 2,
        "keep_match_length": True,  # full match, not 1/5 of one at 50 Hz
    }
    if args.no_magnet:
        values["use_magnet"] = False
    if not args.no_injury_time:
        values["injury_time"] = True
    for side, name, short, players in (("left", args.home, args.home_short, args.home_players),
                                       ("right", args.away, args.away_short, args.away_players)):
        if name:
            values[f"{side}_team_name"] = name
        if short:
            values[f"{side}_team_short_name"] = short
        if players:
            names = {str(i): {"name": n.strip()} for i, n in enumerate(players.split(",")) if n.strip()}
            values.setdefault("player_overrides", {})[side] = names
    if not args.upstream:
        values.update(V1, pitch_condition=args.pitch, pass_power_assist=args.power_assist,
                      kick_windup_speed=args.windup)
    env = football_env.FootballEnv(gconfig.Config(values))
    env.render()
    env.reset()
    # GRF's real_time mode only adjusts how many frames it draws and never
    # waits, so on a fast Mac the game runs faster than real time. Pace it:
    # each step is physics_steps_per_frame ticks of 10 ms game time.
    step_seconds = values["physics_steps_per_frame"] * 0.01 / max(args.speed, 0.1)
    replays = None if args.no_replays else Replays(env, step_seconds)
    goal_replay_at = None  # step at which to start a pending goal replay
    score = None
    next_step = time.perf_counter()
    try:
        while True:
            if replays:
                replays.before_step()
            observation, _, done, _ = env.step([])
            if replays:
                replays.after_step(done)
                new_score = tuple(observation["score"])  # no agent: one raw dict
                if score is not None and new_score != score and not done:
                    goal_replay_at = replays.t + 100  # let the celebration start (2 s)
                score = new_score
                if goal_replay_at is not None and replays.t >= goal_replay_at:
                    replays.play(end_t=goal_replay_at - 50, label="GOAL REPLAY")
                    goal_replay_at = None
                    next_step = time.perf_counter()
                elif replays.share_pressed():
                    replays.play()
                    next_step = time.perf_counter()
            if done:
                env.reset()
                score, goal_replay_at = None, None
            next_step += step_seconds
            delay = next_step - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_step = time.perf_counter()  # running behind: don't catch up
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
