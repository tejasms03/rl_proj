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
"""

from __future__ import annotations

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import pygame  # noqa: E402

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", default="11_vs_11_stochastic")
    parser.add_argument("--pitch", default="normal", choices=["normal", "dry", "wet", "long_grass"])
    parser.add_argument("--power-assist", type=float, default=0.5,
                        help="0 = your charge sets the distance, 1 = fully automatic")
    parser.add_argument("--upstream", action="store_true", help="original GRF physics")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed: 1.0 = real time, 0.8 = 20%% slower "
                             "(display only; physics unchanged)")
    parser.add_argument("--no-magnet", action="store_true",
                        help="no assisted steering toward the ball: pure stick control")
    parser.add_argument("--test", action="store_true", help="check controller buttons")
    args = parser.parse_args()
    if args.test:
        test_controller()
        return

    values = {
        "action_set": "full",
        "players": ["ps_gamepad:left_players=1"],
        "real_time": True,
        "level": args.level,
        # Read the controller 50 times a second (default 10); physics is
        # unchanged, the engine still runs at 100 ticks per second.
        "physics_steps_per_frame": 2,
    }
    if args.no_magnet:
        values["use_magnet"] = False
    if not args.upstream:
        values.update(V1, pitch_condition=args.pitch, pass_power_assist=args.power_assist)
    env = football_env.FootballEnv(gconfig.Config(values))
    env.render()
    env.reset()
    # GRF's real_time mode only adjusts how many frames it draws and never
    # waits, so on a fast Mac the game runs faster than real time. Pace it:
    # each step is physics_steps_per_frame ticks of 10 ms game time.
    step_seconds = values["physics_steps_per_frame"] * 0.01 / max(args.speed, 0.1)
    next_step = time.perf_counter()
    try:
        while True:
            _, _, done, _ = env.step([])
            if done:
                env.reset()
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
