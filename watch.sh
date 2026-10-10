#!/usr/bin/env bash
# Watch a scenario live (see grf_tools/watch.py). Works from any folder:
#   ~/projects/rl_project/watch.sh academy_corner --pitch wet
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${GRF_ENV_PREFIX:-$HOME/miniconda3/envs/grf}/bin/python"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" exec "$PY" -m grf_tools.watch "$@"
