# Environment

What the reported run was actually executed on, and how to rebuild that environment.

## Machine

| | |
|---|---|
| OS | Windows 10 Enterprise x64 (build 19045) |
| GPU | NVIDIA TITAN RTX, 24 GB, **compute capability 7.5 (Turing, sm_75)** |
| Driver | 581.42 (CUDA 13.0 capable) |
| CPU / RAM | 16 logical cores / 64 GB |
| Python | 3.12.12 |

Turing matters: it has **no TF32 and no bf16 tensor cores**. fp16 tensor cores are available
(measured 73.6 TFLOPS for a 2048³ matmul vs 10.8 TFLOPS in fp32 on this card), which is why the
mixed-precision path uses fp16 rather than the bf16 that is conventional on Blackwell.

## Software versions actually installed

| package | version | source |
|---|---|---|
| torch | **2.14.1+cu130** | `https://download.pytorch.org/whl/cu130` |
| warp-lang | 1.18.0 | `https://pypi.nvidia.com/` |
| mujoco | 3.16.0 | PyPI |
| mujoco-warp | 3.16.0 | PyPI |
| mjlab | 1.6.0 at commit **`8bb2bc9e`** | GitHub archive (pinned by `uv.lock`) |
| triton (triton-windows) | **3.8.0.post29** | PyPI (`triton-windows`) |
| numpy / scipy | 2.5.4 / 1.18.1 | PyPI (pinned by `uv.lock`) |
| imageio-ffmpeg / Pillow | 0.6.0 / 13.0.0 | PyPI (video/GIF output) |

The full frozen list of the 137 installed distributions is reproducible from `uv.lock`; the versions
above are the ones that mattered for behaviour and speed.

## Rebuilding it

The upstream project pins every dependency in `uv.lock` (Python `==3.12.*`). On a machine with normal
network access:

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130
pip install "warp-lang==1.18.0"  --index-url https://pypi.nvidia.com/
pip install -r requirements-locked.txt      # every other package at its uv.lock version
pip install --no-deps mjlab-<commit>.tar.gz # the pinned mjlab archive
pip install --no-deps -e .
```

`requirements-locked.txt` is generated from `uv.lock` (only the plain-PyPI entries; `torch`,
`warp-lang` and `mjlab` are handled by the dedicated lines above). If your network can reach
`github.com`, `uv sync --frozen` with the upstream `pyproject.toml` does all of it in one step.

### Notes specific to this machine

* **`github.com` is not reachable** from it, so `mjlab` was fetched as an archive through a GitHub
  mirror and installed with `--no-deps`. The exact commit is pinned in `uv.lock`, so the source is
  identical either way.
* **`huggingface.co` is not reachable** either; the data script therefore defaults to the public
  mirror `https://hf-mirror.com` (override with `--endpoint` or `HF_ENDPOINT`).
* **`git` is not installed** on this machine. `rsl_rl` (pulled in by `mjlab`) imports `gitpython` at
  import time and raises if no `git` executable is found. A portable MinGit on `PATH` — or
  `GIT_PYTHON_REFRESH=quiet` plus a `git` binary — fixes it. This is an environment quirk of this
  host, not of the code.
* Rendering uses CPU MuJoCo (`mujoco.Renderer`) and works with the **default WGL backend** on this
  host; no `MUJOCO_GL=osmesa` is needed. `scripts/probe_render.py` verifies that in one command.

## Pre-flight check

```bash
python scripts/check_env.py
```

This prints torch/CUDA/GPU, Warp and MuJoCo versions, loads the G1 MJCF, runs the differentiable FK,
builds an 8-environment instance of the real tracking task, does one simulator step, and runs one
forward + **backward** pass of the world model, the policy and the critic. It also measures a 2048³
matmul in fp16 and fp32 so you can see whether your card has the tensor cores this run relies on.

The first invocation also compiles the MuJoCo-Warp kernels into
`%LOCALAPPDATA%\NVIDIA\warp\Cache\1.18.0` (≈ 1–2 minutes); later runs are cached.

## Optional accelerator: `torch.compile` on Windows

Upstream disables `torch.compile` on Windows on the assumption that Inductor needs Triton, which has
no official Windows wheels. Installing the community **`triton-windows`** package makes it work; this
run enabled it and measured a large speed-up (see [TRAINING.md](TRAINING.md)). Verify with:

```bash
python scripts/check_compile.py                 # compiles a pointwise graph, the WM loss step and ImagStep
python scripts/bench_full.py --amp --compile default --num-envs 4096
```

If compilation fails on your setup, drop `--compile default` (and optionally keep `--amp`): the run is
then correct but roughly 3× slower per iteration.