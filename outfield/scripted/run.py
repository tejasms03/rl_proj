"""Run the scripted 3v3 play against the game's AI and compare it with the
game's AI playing our side, seed for seed.

    python -m outfield.scripted.run                    # 20 seeds, both modes, numbers only
    python -m outfield.scripted.run --seeds 1 2 3 4
    python -m outfield.scripted.run --watch            # watch the scripted play live
    python -m outfield.scripted.run --watch --mode ai  # watch the game's AI in the same scene
    python -m outfield.scripted.run --record --seeds 12   # MP4 with every player's command on screen

While watching, the terminal prints what each player is doing whenever
that changes. The live game window can't show our text, so --record writes a
video (out/outfield/videos/) with the commands drawn on every frame.

Each episode is one attack (scene.py): it ends on a goal, the ball going out,
them winning the ball, or after 30 s. Both modes start from the same seed;
the only difference is who decides for our three outfield players. Our keeper
is the game's AI in both.

Numbers come from headless runs only: rendering changes the game, so a
watched episode does not replay a headless one (docs/CONTEXT.md §7.5).
Every headless game runs in its own process (one engine per process, §7.7).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

MODES = ("ai", "scripted")
OUTCOMES = ("goal", "keeper", "lost", "out", "timeout")


def episode(mode: str, seed: int, render: bool = False, real_time: bool = False,
            show: bool = False, video: Path | None = None) -> dict:
    """Play one attack; return its outcome and a few numbers. show: print each
    player's command when it changes. video: write an MP4 with the commands
    drawn on screen (rendered, so it plays differently from a headless run)."""
    import numpy as np

    import gfootball.env as football_env

    from outfield.config import STEP_S, TRAINING_CONFIG
    from outfield.scripted import play, scene

    env = football_env.create_environment(
        scene.NAME, representation="raw", render=render,
        number_of_left_players_agent_controls=4,
        other_config_options={**TRAINING_CONFIG, "game_engine_random_seed": seed})
    script = play.TrianglePlay() if mode == "scripted" else None
    ai = [play.ACTION["builtin_ai"]] * 4
    digest = hashlib.blake2b(digest_size=8)
    passes, steps, last_ours, last_owner = 0, 0, None, (-1, -1)
    writer, shown = None, {}
    try:
        obs = env.reset()
        assert [int(o["active"]) for o in obs] == [0, 1, 2, 3], "agent i must control player i"
        if video is not None:
            from outfield.scripted import overlay
            env.render(mode="rgb_array")  # as the duty lab: rendering on from the first step
            writer = overlay.VideoWriter(video)
        done = False
        next_step = time.perf_counter()
        while not done:
            actions = script.actions(obs) if script else ai
            intents = script.intent if script else {i: "game AI" for i in range(4)}
            sent = {i: play.action_name(a) for i, a in enumerate(actions)}
            if show:
                now = {i: intents.get(i, "") for i in range(1, 4)}
                if now != shown:  # intents only: buttons change nearly every step
                    print(f"{steps * STEP_S:5.1f} s  " + "  |  ".join(
                        f"#{i} {intent}" for i, intent in now.items()), flush=True)
                    shown = now
            obs, _, done, _ = env.step(actions)
            steps += 1
            view = obs[0]
            if writer is not None:
                writer.add(env.render(mode="rgb_array"), view, intents, sent, steps * STEP_S, mode)
            digest.update(np.round(view["ball"], 4).tobytes())
            owner = (int(view["ball_owned_team"]), int(view["ball_owned_player"]))
            if owner[0] == 0:
                if last_ours is not None and owner[1] != last_ours:
                    passes += 1  # a different one of ours has it, and they never did
                last_ours = owner[1]
            elif owner[0] == 1:
                last_ours = None
            if owner[0] != -1:
                last_owner = owner
            if real_time:
                next_step += STEP_S
                delay = next_step - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_step = time.perf_counter()
    finally:
        if writer is not None:
            writer.close()
        env.close()

    if view["score"][0] > 0:
        outcome = "goal"
    elif last_owner[0] == 1 and view["game_mode"] == 0:
        outcome = "keeper" if last_owner[1] == 0 else "lost"
    elif view["game_mode"] != 0:
        outcome = "out"
    else:
        outcome = "timeout"
    kicks = Counter(k for _, _, k, _ in script.log) if script else {}
    return {"mode": mode, "seed": seed, "outcome": outcome, "seconds": round(steps * STEP_S, 1),
            "passes": passes, "shots": kicks.get("shot"), "pass_attempts": kicks.get("short_pass"),
            "digest": digest.hexdigest()}


def summarise(results: list[dict]) -> str:
    lines = ["| mode | n | " + " | ".join(OUTCOMES) + " | passes / ep | seconds / ep |",
             "|---|---|" + "---|" * len(OUTCOMES) + "---|---|"]
    for mode in MODES:
        rows = [r for r in results if r["mode"] == mode]
        if not rows:
            continue
        count = Counter(r["outcome"] for r in rows)
        n = len(rows)
        lines.append(f"| {mode} | {n} | " + " | ".join(str(count[o]) for o in OUTCOMES)
                     + f" | {sum(r['passes'] for r in rows) / n:.1f}"
                     + f" | {sum(r['seconds'] for r in rows) / n:.1f} |")
    return "\n".join(lines)


def watch(mode: str, seeds: list[int]) -> None:
    for seed in seeds:
        print(f"--- seed {seed} ({mode}) ---", flush=True)
        r = episode(mode, seed, render=True, real_time=True, show=True)
        print(f"seed {seed}: {r['outcome']} after {r['seconds']} s, {r['passes']} passes\n", flush=True)


def record(mode: str, seeds: list[int], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        path = folder / f"3v3_{mode}_seed{seed}.mp4"
        r = episode(mode, seed, video=path)
        print(f"seed {seed}: {r['outcome']} after {r['seconds']} s -> {path}", flush=True)


def compare(seeds: list[int], out: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    jobs = [(mode, seed) for seed in seeds for mode in MODES]
    jobs.append(("scripted", seeds[0]))  # a repeat: the same seed must replay exactly
    with concurrent.futures.ProcessPoolExecutor(max_workers=6, mp_context=ctx) as pool:
        results = list(pool.map(episode, *zip(*jobs)))
    repeat = results.pop()
    first = next(r for r in results if r["mode"] == "scripted" and r["seed"] == seeds[0])
    replays = repeat["digest"] == first["digest"]

    print("| seed | ai | scripted |\n|---|---|---|")
    for seed in seeds:
        a, s = (next(r for r in results if r["mode"] == m and r["seed"] == seed) for m in MODES)
        print(f"| {seed} | {a['outcome']} ({a['seconds']} s) | {s['outcome']} ({s['seconds']} s, "
              f"{s['passes']} passes, {s['shots'] or 0} shots) |")
    print()
    print(summarise(results))
    print(f"\nsame seed replays exactly: {'yes' if replays else 'NO, results are not comparable'}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": seeds, "replays": replays, "results": results}, indent=1))
    print(f"saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 21)))
    parser.add_argument("--watch", action="store_true", help="render live instead of measuring")
    parser.add_argument("--record", action="store_true",
                        help="write an MP4 per seed with each player's command on screen")
    parser.add_argument("--mode", choices=MODES, default="scripted",
                        help="with --watch / --record: who plays our side")
    parser.add_argument("--out", type=Path, default=Path("out/outfield/scripted_3v3.json"))
    args = parser.parse_args()
    if args.record:
        record(args.mode, args.seeds, Path("out/outfield/videos"))
    elif args.watch:
        watch(args.mode, args.seeds)
    else:
        compare(args.seeds, args.out)


if __name__ == "__main__":
    main()
