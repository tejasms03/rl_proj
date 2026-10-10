#!/usr/bin/env bash
# Play against the AI with a PlayStation controller (see grf_tools/play_gamepad.py).
#   ~/projects/rl_project/play.sh            ~/projects/rl_project/play.sh --test
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${GRF_ENV_PREFIX:-$HOME/miniconda3/envs/grf}/bin/python"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" exec "$PY" -m grf_tools.play_gamepad "$@"
