# rl_project

A patched Google Research Football (GRF) engine with more realistic mechanics, for training RL agents. See [PLAN.md](PLAN.md).

## Setup

```bash
conda env create -f environment.yml
./build.sh
```

`build.sh` compiles the engine in `upstream/` and installs it into the `grf` env. Run it again after any engine change; it rebuilds incrementally. Use `--clean` for a full rebuild.

## Linux / training machines

```bash
docker build -f docker/Dockerfile -t rl_project .
```
```bash
docker run --rm rl_project
```

The second command runs the test suite inside the container. Without Docker, follow `docker/Dockerfile` by hand: system OpenGL libraries, the conda env, `conda install -n grf -c conda-forge cxx-compiler make`, then `conda run -n grf ./build.sh --clean`. Not yet tested on Linux.

## Tests

```bash
~/miniconda3/envs/grf/bin/python -m pytest
```

Call the env's Python by full path, or use `conda activate --stack grf`. A plain `conda activate grf` can leave another `python3` earlier on PATH.
