"""Play against the built-in AI with a PlayStation controller.

    ./play.sh                                  # 11 vs 11 against the AI
    ./play.sh --level 11_vs_11_hard_stochastic
    ./play.sh --level academy_3_vs_1_with_keeper
    ./play.sh --test                           # show which buttons your pad reports

You control the active player (switches automatically, or L1); the AI plays
everyone else. FIFA-style layout, hold kicks to charge power:
  X short pass / pressure     O shot / slide tackle
  [] lofted pass / team press /\\ through ball / keeper rush
  R1 curl shot   L3 chip   R3 driven long ball   L1 switch
  R2 sprint      L2 close control (dribble)    left stick move
"""

from __future__ import annotations

import argparse
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
    }
    if not args.upstream:
        values.update(V1, pitch_condition=args.pitch, pass_power_assist=args.power_assist)
    env = football_env.FootballEnv(gconfig.Config(values))
    env.render()
    env.reset()
    try:
        while True:
            _, _, done, _ = env.step([])
            if done:
                env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
