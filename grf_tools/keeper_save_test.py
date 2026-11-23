"""Can an agent-controlled goalkeeper make saves?

In the engine, the keeper's save (dive / deflect) logic only runs for the
built-in AI controller. This test checks that: the same shots are taken at a
keeper controlled three ways, and we count goals conceded.

    python -m grf_tools.keeper_save_test               # 30 shots per mode
    python -m grf_tools.keeper_save_test --episodes 10 --watch

Modes:
    builtin_ai   the built-in AI controls the keeper (normal GRF keeper)
    agent_idle   our agent controls the keeper and sends no input
    agent_track  our agent controls the keeper and only slides along the line
                 to stay level with the ball (positioning, no dive command)

If agent_idle and agent_track concede far more than builtin_ai, an
agent-controlled keeper cannot save on its own.
"""

from __future__ import annotations

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import gfootball.env as football_env  # noqa: E402

from gfootball.env import football_action_set  # noqa: E402  isort: skip

V1 = {
    "realistic_player_physics": True,
    "realistic_ball_physics": True,
    "ball_pace": "game",
    "situational_accuracy": True,
}
MODES = ["builtin_ai", "agent_idle", "agent_track"]


def keeper_action(mode: str, obs: dict, names: list[str]) -> int:
    """The action index our agent sends for the keeper this step."""
    if mode == "builtin_ai":
        return names.index("builtin_ai")
    if mode == "agent_idle":
        return names.index("idle")
    # agent_track: our team is 'left_team' in our own (mirrored) frame; keeper = 0
    keeper_y = obs["left_team"][0][1]
    ball_y = obs["ball"][1]
    if ball_y < keeper_y - 0.01:
        return names.index("top")
    if ball_y > keeper_y + 0.01:
        return names.index("bottom")
    return names.index("idle")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30, help="shots per mode")
    parser.add_argument("--watch", action="store_true", help="show the game window in real time")
    parser.add_argument("--upstream", action="store_true", help="original GRF, no patches")
    args = parser.parse_args()

    config = {"action_set": "v2", "real_time": args.watch}
    if not args.upstream:
        config.update(V1)
    env = football_env.create_environment(
        "drill_keeper_save", representation="raw", render=args.watch,
        number_of_left_players_agent_controls=0,
        number_of_right_players_agent_controls=1,
        other_config_options=config)
    names = [a._name for a in football_action_set.get_action_set({"action_set": "v2"})]

    conceded = {mode: 0 for mode in MODES}
    try:
        for mode in MODES:
            for episode in range(args.episodes):
                obs = env.reset()
                done = False
                while not done:
                    action = keeper_action(mode, obs[0], names)
                    obs, _, done, info = env.step([action])
                    if info["score_reward"] > 0:  # the attacking (left) team scored
                        conceded[mode] += 1
                    if args.watch:
                        time.sleep(0.1)
            print(f"{mode:12s} conceded {conceded[mode]:3d} / {args.episodes}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

    print("\nGoals conceded (lower is better):")
    for mode in MODES:
        print(f"  {mode:12s} {conceded[mode]:3d} / {args.episodes}"
              f"  ({100 * conceded[mode] / args.episodes:.0f}%)")


if __name__ == "__main__":
    main()
