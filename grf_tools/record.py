"""Record side-by-side videos of the patched mechanics.

    python -m grf_tools.record [--out out/videos] [--only NAME ...]

Each clip is rendered twice, upstream (top) and patched (bottom), in fresh
processes, and encoded to H.264 MP4 with ffmpeg at 25 fps.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from grf_tools.trace import make_env

from gfootball.env import football_action_set  # isort: skip

TICKS_PER_FRAME = 4  # 40 ms of game per frame -> real-time at 25 fps
FPS = 25
HALF = (1280, 720)  # each half of the stacked video

_DEFAULT = {a._name: i for i, a in enumerate(
    football_action_set.get_action_set({"action_set": "default"}))}
_V2 = {a._name: i for i, a in enumerate(football_action_set.get_action_set({"action_set": "v2"}))}
_V3 = {a._name: i for i, a in enumerate(football_action_set.get_action_set({"action_set": "v3"}))}
_P3 = {"realistic_ball_physics": True, "action_set": "v3"}


def _pass_plan(kick):
    # 30 ticks of running right ~ 8 frames, then press the kick.
    return lambda f: _DEFAULT["right"] if f < 8 else (_DEFAULT[kick] if f < 10 else 0)


def _v3_plan(kick):
    # ~1.3 s run toward goal, then the kick.
    return lambda f: _V3["right"] if f < 8 else (_V3[kick] if f < 10 else _V3["release_direction"])


def _curl_launch(sign):
    # From drill_shot's ball spot (22 m out, central): 25 m/s, 6 degrees,
    # aimed wide of the post, 60 rad/s sidespin bending it back in.
    def launch(engine):
        engine.launch_ball(32.6, 0.0, 0.11, 24.7, -3.0 * sign, 2.6, 0.0, 0.0, 60.0 * sign)
    return launch


# name -> (scenario, frames, action plan(frame) -> action, upstream cfg, patched cfg, extras)
CLIPS = {
    "high_pass": dict(
        scenario="drill_pass_long", frames=150, plan=_pass_plan("high_pass"),
        left={}, right={"realistic_ball_physics": True},
        caption=("upstream: flat 32 m/s lob", "game pace: 22 m/s, 28 deg, backspin")),
    "long_pass": dict(
        scenario="drill_pass_long", frames=120, plan=_pass_plan("long_pass"),
        left={}, right={"realistic_ball_physics": True},
        caption=("upstream long pass", "game pace: driven 26 m/s, 17 deg")),
    "ground_pass": dict(
        scenario="drill_pass_short", frames=90, plan=_pass_plan("short_pass"),
        left={}, right={"realistic_ball_physics": True},
        caption=("upstream: 18 m/s, heavy grass", "game pace: 14 m/s, FIFA-standard grass")),
    "curl": dict(
        scenario="drill_shot", frames=40, plan=lambda f: 0,
        left={"realistic_ball_physics": True}, right={"realistic_ball_physics": True},
        launch=(_curl_launch(1), _curl_launch(-1)),
        caption=("sidespin one way", "sidespin the other way")),
    "curl_shot": dict(
        scenario="drill_shot", frames=45, plans=(_v3_plan("shot"), _v3_plan("curl_shot")),
        left=_P3, right=_P3,
        caption=("normal shot", "curl shot: starts wide, bends back in")),
    "chip": dict(
        scenario="drill_shot", frames=110, plans=(_v3_plan("shot"), _v3_plan("chip")),
        left=_P3, right=_P3,
        caption=("normal shot", "chip: 45 deg lob with backspin")),
    "through_ball": dict(
        scenario="drill_through", frames=75, plans=(_v3_plan("short_pass"), _v3_plan("through_ball")),
        left=_P3, right=_P3,
        caption=("ground pass: to his feet", "through ball: into space ahead of him")),
    "wet_pitch": dict(
        scenario="drill_pass_short", frames=90, plan=_pass_plan("short_pass"),
        left={"realistic_ball_physics": True},
        right={"realistic_ball_physics": True, "pitch_condition": "wet"},
        caption=("realistic, normal pitch", "realistic, wet pitch")),
}


def _render(clip_name: str, side: int) -> list[np.ndarray]:
    clip = CLIPS[clip_name]
    config = {"physics_steps_per_frame": TICKS_PER_FRAME, **(clip["left"], clip["right"])[side]}
    env = make_env(clip["scenario"], seed=3, config=config)
    frames = []
    try:
        env.reset()
        env.render(mode="rgb_array")  # switches the engine to rendering mode
        if "launch" in clip:
            clip["launch"][side](env.unwrapped._env._env)
        for f in range(clip["frames"]):
            plan = clip["plans"][side] if "plans" in clip else clip["plan"]
            obs, _, done, _ = env.step(plan(f))
            frame = env.render(mode="rgb_array")
            if frame.shape[1::-1] != HALF:
                frame = cv2.resize(frame, HALF, interpolation=cv2.INTER_AREA)
            frames.append(frame)
            if done:
                break
    finally:
        env.close()
    return frames


def _caption(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (HALF[0], 44), (0, 0, 0), -1)
    cv2.putText(out, text, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                cv2.LINE_AA)
    return out


def record(clip_name: str, out_dir: Path) -> Path:
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
        left, right = [f.result() for f in (pool.submit(_render, clip_name, 0),
                                             pool.submit(_render, clip_name, 1))]
    n = max(len(left), len(right))
    left += [left[-1]] * (n - len(left))
    right += [right[-1]] * (n - len(right))
    captions = CLIPS[clip_name]["caption"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{clip_name}.mp4"
    ffmpeg = subprocess.Popen(
        [shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{HALF[0]}x{HALF[1] * 2}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)],
        stdin=subprocess.PIPE)
    for a, b in zip(left, right):
        frame = np.vstack([_caption(a, captions[0]), _caption(b, captions[1])])
        ffmpeg.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {clip_name}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("out/videos"))
    parser.add_argument("--only", nargs="*", choices=sorted(CLIPS))
    args = parser.parse_args()
    for name in args.only or CLIPS:
        print(record(name, args.out))


if __name__ == "__main__":
    main()
