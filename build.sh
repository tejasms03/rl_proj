#!/usr/bin/env bash
# Build the patched GRF engine and install it into the `grf` conda env.
#
#   ./build.sh            incremental rebuild + reinstall
#   ./build.sh --clean    wipe in-tree CMake/object files first
#
# GRF_ENV_PREFIX overrides the env location (default: <conda base>/envs/grf).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENGINE="$ROOT/upstream/third_party/gfootball_engine"

if [[ -z "${GRF_ENV_PREFIX:-}" ]]; then
  CONDA_BIN="${CONDA_EXE:-$(command -v conda || echo "$HOME/miniconda3/bin/conda")}"
  GRF_ENV_PREFIX="$("$CONDA_BIN" info --base)/envs/grf"
fi
PY="$GRF_ENV_PREFIX/bin/python"
[[ -x "$PY" ]] || { echo "No python at $PY; create the env: conda env create -f environment.yml" >&2; exit 1; }

if [[ "${1:-}" == "--clean" ]]; then
  rm -rf "$ENGINE/CMakeCache.txt" "$ENGINE/CMakeFiles" "$ENGINE"/*.dylib "$ENGINE"/*.so "$ROOT/upstream/build"
fi

PY_VERSION="$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"

# build_game_engine.sh shells out to `python3`, so the env must come first on PATH.
export PATH="$GRF_ENV_PREFIX/bin:$PATH"
export CMAKE_PREFIX_PATH="$GRF_ENV_PREFIX"
# Without these, CMake picks up a framework/Homebrew Python instead of the env's.
export GFOOTBALL_CMAKE_ARGS="-DPython_ROOT_DIR=$GRF_ENV_PREFIX -DPython_FIND_FRAMEWORK=NEVER -DPython_FIND_STRATEGY=LOCATION -DBoost_PYTHON_VERSION=$PY_VERSION"

"$PY" -m pip install --no-build-isolation --force-reinstall --no-deps "$ROOT/upstream"
echo "Installed into $GRF_ENV_PREFIX (Python $PY_VERSION)"
