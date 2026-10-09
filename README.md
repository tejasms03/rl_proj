# rl_project

A patched Google Research Football (GRF) engine with more realistic mechanics, for training RL agents. See [PLAN.md](PLAN.md).

## Setup

```bash
conda env create -f environment.yml
./build.sh
```

`build.sh` compiles the engine in `upstream/` and installs it into the `grf` env. Run it again after any engine change; it rebuilds incrementally. Use `--clean` for a full rebuild.

## Tests

```bash
~/miniconda3/envs/grf/bin/python -m pytest
```

Call the env's Python by full path, or use `conda activate --stack grf`. A plain `conda activate grf` can leave another `python3` earlier on PATH.
