"""Watch a scenario live in a game window, played by the built-in AI.

    python -m grf_tools.watch academy_3_vs_1_with_keeper
    python -m grf_tools.watch 11_vs_11_easy_stochastic --upstream
    python -m grf_tools.watch academy_pass_and_shoot_with_keeper --episodes 10 --pitch wet

Runs in real time with the v1.0 patch switches on (unless --upstream).
Close the window or press Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", nargs="?", default="academy_3_vs_1_with_keeper")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--pitch", default="normal", choices=["normal", "dry", "wet", "long_grass"])
    parser.add_argument("--upstream", action="store_true", help="original GRF, no patches")
    parser.add_argument("--fast", action="store_true", help="run as fast as possible")
    args = parser.parse_args()

    config = {"action_set": "v2", "real_time": not args.fast}
    if not args.upstream:
        config.update(V1, pitch_condition=args.pitch)
    env = football_env.create_environment(
        args.scenario, representation="raw", render=True, other_config_options=config)
    builtin_ai = [a._name for a in football_action_set.get_action_set(
        {"action_set": "v2"})].index("builtin_ai")
    try:
        for episode in range(args.episodes):
            env.reset()
            done = False
            while not done:
                obs, _, done, _ = env.step(builtin_ai)
            print(f"episode {episode + 1}: score {obs[0]['score']}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
