"""Draw each player's command on the game frame and write the MP4.

The live game window has no hook for our text (custom_display_stats only
reaches GRF's own dump videos), so commands go on recorded frames instead.
Per frame: a top bar (clock, who has the ball), a panel with every player's
intent and the button actually sent, and a mini-pitch with numbered players.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from outfield.config import STEP_S

SIZE = (1280, 720)
FPS = round(1 / STEP_S)  # one frame per env step: 10 fps
FONT = cv2.FONT_HERSHEY_SIMPLEX
# RGB: frames go to ffmpeg as rgb24.
WHITE, GREY, OURS, THEIRS, BALL = (240, 240, 240), (170, 170, 170), (255, 150, 30), (90, 160, 255), (255, 230, 0)
ROLES = {0: "GK", 1: "CM", 2: "LW", 3: "RW"}
MAP = (300, 194)  # mini-pitch, bottom right
PANEL_W = 525  # command panel, bottom left: clear of the game's radar (bottom centre)


def _shade(img, x0, y0, x1, y1, alpha=0.6):
    box = img[y0:y1, x0:x1]
    box[:] = (box * (1 - alpha)).astype(np.uint8)


def _text(img, text, xy, scale=0.55, colour=WHITE, thick=1):
    cv2.putText(img, text, xy, FONT, scale, colour, thick, cv2.LINE_AA)


def _owner(view) -> str:
    team, player = int(view["ball_owned_team"]), int(view["ball_owned_player"])
    if team == 0:
        return f"ours (#{player})"
    if team == 1:
        return f"theirs (#{player})"
    return "loose"


def _map(img, view):
    w, h = MAP
    x0, y0 = SIZE[0] - w - 12, SIZE[1] - h - 12
    box = img[y0:y0 + h, x0:x0 + w]
    box[:] = (box * 0.25 + np.array([40, 110, 40]) * 0.75).astype(np.uint8)
    px = lambda p: (int((p[0] + 1) / 2 * w), int((p[1] + 0.42) / 0.84 * h))  # noqa: E731
    cv2.rectangle(box, (0, 0), (w - 1, h - 1), WHITE, 1)
    cv2.line(box, (w // 2, 0), (w // 2, h), WHITE, 1)
    for p in view["right_team"]:
        cv2.circle(box, px(p), 5, THEIRS, -1)
    for i, p in enumerate(view["left_team"]):
        cv2.circle(box, px(p), 8, OURS, -1)
        _text(box, str(i), (px(p)[0] - 4, px(p)[1] + 4), 0.38, (0, 0, 0), 1)
    cv2.circle(box, px(view["ball"]), 4, BALL, -1)


def annotate(frame, view, intents, sent, seconds, mode) -> np.ndarray:
    """The game's own scoreboard (top left) and radar (bottom centre) stay
    visible: our bar sits top right, our panel bottom left."""
    img = cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA)
    bar_x = 640
    _shade(img, bar_x, 0, SIZE[0], 34, 0.7)
    who = "SCRIPT" if mode == "scripted" else "GAME AI"
    _text(img, f"{seconds:5.1f} s   our side: {who}   ball: {_owner(view)}", (bar_x + 12, 23), 0.6,
          WHITE, 2)
    rows = [i for i in sorted(sent) if i != 0] + [0]
    width, line = PANEL_W, 22
    top = SIZE[1] - 12 - 2 * line * len(rows) - 8
    _shade(img, 0, top, width, SIZE[1], 0.65)
    for n, i in enumerate(rows):
        y = top + 22 + 2 * line * n
        has_ball = int(view["ball_owned_team"]) == 0 and int(view["ball_owned_player"]) == i
        colour = BALL if has_ball else OURS if i else GREY
        _text(img, f"#{i} {ROLES.get(i, '')}", (12, y), 0.55, colour, 2)
        _text(img, intents.get(i, ""), (80, y), 0.52, WHITE if i else GREY)
        _text(img, f"sent: {sent[i]}", (80, y + line - 4), 0.45, GREY)
    _map(img, view)
    return img


class VideoWriter:
    """Annotated frames -> H.264 MP4 via ffmpeg (as the duty lab)."""

    def __init__(self, path: Path):
        self.path = path
        self.proc = subprocess.Popen(
            [shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)],
            stdin=subprocess.PIPE)

    def add(self, frame, view, intents, sent, seconds, mode) -> None:
        img = annotate(frame, view, intents, sent, seconds, mode)
        self.proc.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")
