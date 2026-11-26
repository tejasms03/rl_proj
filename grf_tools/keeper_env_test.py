"""Smoke test: our keeper (KeeperNet, untrained) plugged into O2's env wrapper.

The 10 outfield players press random allowed buttons; the keeper (slot 0) is
driven by KeeperNet, with O2's masks applied to its scores before it picks.
Every step prints the keeper's choices and keeper_reward. Goal: no crash, the
keeper only ever picks allowed buttons, and the reward numbers look sensible.

    PYTHONPATH=. $GRF_ENV_PREFIX/bin/python -m grf_tools.keeper_env_test
    PYTHONPATH=. $GRF_ENV_PREFIX/bin/python -m grf_tools.keeper_env_test --steps 300 --watch
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from grf_tools.keeper import KeeperNet, keeper_reward
from outfield.env import OutfieldEnv, random_actions

HEADS = ("move", "speed", "ball", "defend")  # the wrapper's names for heads A-D


def keeper_choice(net: KeeperNet, obs: dict) -> tuple[dict, float]:
    """KeeperNet's pick for slot 0: mask the scores, then sample each head."""
    x = torch.as_tensor(obs["actor"][0], dtype=torch.float32)  # the keeper's 960 numbers
    with torch.no_grad():
        heads, value = net(x)
    *scores, power = heads  # move, speed, ball, defend scores; then the 2 Beta numbers
    choice = {}
    for name, s in zip(HEADS, scores):
        allowed = torch.as_tensor(obs["masks"][name][0])
        s = s.masked_fill(~allowed, float("-inf"))  # a blocked button can never be picked
        choice[name] = int(torch.distributions.Categorical(logits=s).sample())
    choice["power"] = float(torch.distributions.Beta(power[0], power[1]).sample())
    return choice, float(value)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--scenario", default="11_vs_11_easy_stochastic")
    p.add_argument("--watch", action="store_true", help="show the game window")
    args = p.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = KeeperNet()
    env = OutfieldEnv(args.scenario, seed=1, keeper="agent", render=args.watch)
    blocked_picks = 0
    total = 0.0
    try:
        obs = env.reset()
        memory = {"pass_in_air": False}
        for step in range(1, args.steps + 1):
            actions = random_actions(obs["masks"], rng)  # all 11 players: random allowed buttons
            choice, value = keeper_choice(net, obs)
            for name in HEADS:  # check the mask really held, then overwrite slot 0
                blocked_picks += not obs["masks"][name][0, choice[name]]
                actions[name][0] = choice[name]
            actions["power"][0] = choice["power"]

            prev_raw = env.raw_list[0]  # the photo before the step (our team = left)
            obs, rewards, terminated, truncated, info = env.step(actions)
            reward = keeper_reward(prev_raw, info["raw"][0], info["duty_ok"][0] == 1, memory)
            total += reward
            print(f"{step:4d}  move {choice['move']:2d}  speed {choice['speed']}  "
                  f"ball {choice['ball']}  defend {choice['defend']}  power {choice['power']:.2f}  "
                  f"value {value:+.3f}  reward {reward:+.3f}"
                  + ("  (overridden)" if info["overridden"][0] else ""))
            if terminated or truncated:
                print("episode ended:", info["outcome"])
                obs = env.reset()
                memory = {"pass_in_air": False}
    finally:
        env.close()

    print(f"\n{args.steps} steps, no crash. Keeper picks of a blocked button: {blocked_picks} "
          f"(should be 0). Total keeper reward: {total:+.3f}")


if __name__ == "__main__":
    main()
