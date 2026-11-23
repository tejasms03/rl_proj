"""What does keeper_rush do for an agent-controlled keeper?

A built-in-AI striker dribbles in on goal (drill_keeper_save) and our agent
controls the keeper three ways. We count goals conceded and how far the keeper
comes off his line (furthest distance per shot, averaged).

    python -m grf_tools.keeper_rush_test               # 30 shots per mode
    python -m grf_tools.keeper_rush_test --episodes 5 --watch

Modes:
    idle             no input: the keeper stays where he is
    rush_every_step  presses keeper_rush every step
    rush_once        presses keeper_rush on the first step only, then idle
                     (is keeper_rush a toggle that stays on?)
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
MODES = ["idle", "rush_every_step", "rush_once"]

def keeper_action(mode: str, step: int, names: list[str]) -> int:
    """The action index our agent sends for the keeper this step."""
    if mode == "idle":
        return names.index("idle")
    if mode == "rush_every_step":
        return names.index("keeper_rush")
    # rush_once: rush on the first step only
    if step == 0:
        return names.index("keeper_rush")
    return names.index("idle")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30, help="shots per mode")
    parser.add_argument("--watch", action="store_true", help="show the game window in real time")
    parser.add_argument("--upstream", action="store_true", help="original GRF, no patches")
    args = parser.parse_args()

    config = {"action_set": "full", "real_time": args.watch}
    if not args.upstream:
        config.update(V1)
    env = football_env.create_environment(
        "drill_keeper_save", representation="raw", render=args.watch,
        number_of_left_players_agent_controls=0,
        number_of_right_players_agent_controls=1,
        other_config_options=config)
    names = [a._name for a in football_action_set.get_action_set({"action_set": "full"})]

    conceded = {mode: 0 for mode in MODES}
    try:
        furthest = {mode: [] for mode in MODES}
        for mode in MODES:
            for episode in range(args.episodes):
                obs = env.reset()
                done = False
                step = 0
                max_dist = 0
                while not done:
                    action = keeper_action(mode, step, names)
                    obs, _, done, info = env.step([action])
                    keeper_x = obs[0]["left_team"][0][0]
                    dist = (keeper_x + 1) * 54.4
                    max_dist = max(dist, max_dist)
                    if info["score_reward"] > 0:  # the attacking (left) team scored
                        conceded[mode] += 1
                    if args.watch:
                        time.sleep(0.1)
                    step+=1

                furthest[mode].append(max_dist)

            avg = sum(furthest[mode]) / len(furthest[mode])
            print(f"  {mode:16s} furthest off line: {avg:.1f} m")
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
