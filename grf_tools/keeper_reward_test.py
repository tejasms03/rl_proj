from __future__ import annotations
from grf_tools.keeper import keeper_reward
import argparse
import json
import math
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

MODES = ["positioned", "off_post"]
V1 = {
    "realistic_player_physics": True,
    "realistic_ball_physics": True,
    "ball_pace": "game",
    "situational_accuracy": True,
}
BALL_RADIUS = 0.11
SHOT_FROM = (35.0, 0.0)  # engine metres; the keeper's goal line is x = 55
# (goal-line y in metres, shot speed m/s): corners, inside the posts, centre
SHOTS = [(-2.8, 25.0), (2.8, 25.0), (-1.6, 22.0), (1.6, 22.0), (0.0, 20.0)]


def launch_shot(env, episode: int) -> None:
    """Fire one shot at the right team's goal from 20 m out."""
    target_y, speed = SHOTS[episode % len(SHOTS)]
    dx, dy = 55.0 - SHOT_FROM[0], target_y - SHOT_FROM[1]
    d = math.hypot(dx, dy)
    vx, vy = speed * dx / d, speed * dy / d
    vz = 4.0  # low shot that stays under the bar
    engine = env.unwrapped._env._env
    engine.launch_ball(SHOT_FROM[0], SHOT_FROM[1], BALL_RADIUS, vx, vy, vz, 0.0, 0.0, 0.0)


def run_mode(mode: str, episodes: int, watch: bool, pause: bool = False) -> int:
    import gfootball.env as football_env
    from gfootball.env import football_action_set

    config = {"action_set": "v4", "real_time": watch, **V1, "keeper_auto_save": True, "use_magnet": False}
    keeper_agent = mode != "ai_keeper"
    env = football_env.create_environment(
        "drill_keeper_shot", representation="raw", render=watch,
        number_of_left_players_agent_controls=1,  # the parked left player, always idle
        number_of_right_players_agent_controls=1 if keeper_agent else 0,
        other_config_options=config)
    names = [a._name for a in football_action_set.get_action_set({"action_set": "v4"})]
    idle = names.index("idle")

    conceded = 0
    try:
        for episode in range(episodes):
            obs = env.reset()
            done = False
            goal = False
            step = 0
            memory = {"pass_in_air": False}
            while not done:
                step += 1
                if step == 30:
                    launch_shot(env, episode)
                actions = [idle]
                if keeper_agent:
                    actions.append(keeper_action(mode, obs[1], names))
                prev_obs = obs
                obs, _, done, info = env.step(actions)
                reward = keeper_reward(prev_obs[1], obs[1], False, memory)
                print(step, reward, file=sys.stderr)
                if info["score_reward"] > 0:
                    conceded += 1
                    goal = True
                if watch:
                    time.sleep(0.1)
            if pause:
                result = "GOAL" if goal else "no goal"
                print(f"[{mode}] shot {episode + 1} of {episodes}: {result}. "
                      "Press Enter for the next shot...", file=sys.stderr, flush=True)
                input()
            elif watch:
                time.sleep(1.0)  # short pause between shots
    finally:
        env.close()
    return conceded


def keeper_action(mode: str, obs: dict, names: list[str]) -> int:
    keeper_y = obs["left_team"][0][1]  # our own frame: our team is 'left_team'
    if mode == "off_post":
        if keeper_y > -0.1:
            return names.index("top")
        return names.index("release_direction")
    ball_y = obs["ball"][1]
    if ball_y < keeper_y - 0.01:
        return names.index("top")
    if ball_y > keeper_y + 0.01:
        return names.index("bottom")
    return names.index("release_direction")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20, help="shots per mode")
    parser.add_argument("--watch", action="store_true", help="show the game window in real time")
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--pause", action="store_true",
                        help="wait for Enter after each shot (use with --watch)")
    parser.add_argument("--child", choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:  # one mode, in this process
        print(json.dumps({"conceded": run_mode(args.child, args.episodes, args.watch, args.pause)}))
        return

    results = {}
    for mode in args.modes:
        print(f"running {mode} ...", flush=True)
        cmd = [sys.executable, "-m", "grf_tools.keeper_reward_test", "--child", mode,
               "--episodes", str(args.episodes)]
        cmd += ["--watch"] if args.watch else []
        cmd += ["--pause"] if args.pause else []
        # stdout carries the result; errors and pause prompts go straight to the terminal
        out = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
        if not lines:
            print(out.stdout[-2000:])
            sys.exit(f"{mode} failed (see the error above)")
        results[mode] = json.loads(lines[-1])["conceded"]

    print(f"\nGoals conceded out of {args.episodes} shots on target (lower is better):")
    for mode, n in results.items():
        print(f"  {mode:12s} {n:3d} / {args.episodes}  ({100 * n / args.episodes:.0f}%)")


if __name__ == "__main__":
    main()