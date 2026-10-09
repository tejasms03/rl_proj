# Linux build of the patched engine, for training machines.
#
#   docker build -f docker/Dockerfile -t rl_project .
#   docker run --rm rl_project                      # runs the test suite
#   docker run --rm -it rl_project bash             # shell with env `grf`
#
# Same route as the macOS setup: conda env (environment.yml) for Python,
# Boost and SDL, then ./build.sh. The compiler comes from conda-forge too, so
# it matches the conda libraries.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates wget git ffmpeg \
      libgl1-mesa-dev libegl1-mesa-dev libglu1-mesa xvfb \
    && rm -rf /var/lib/apt/lists/*

ARG MINIFORGE_VERSION=24.11.3-0
RUN wget -q https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda && rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /rl_project
COPY environment.yml .
RUN conda env create -f environment.yml \
    && conda install -y -n grf -c conda-forge cxx-compiler make \
    && conda clean -afy

COPY . .
# `conda run` activates the env, which points CC/CXX at the conda compilers.
RUN conda run -n grf ./build.sh --clean

# Rendering (videos) needs a display; tests and training run headless.
CMD ["conda", "run", "--no-capture-output", "-n", "grf", "python", "-m", "pytest"]
