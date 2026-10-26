# Training

## The reported run, verbatim

```bash
python scripts/full_train.py \
    --run-name lafan1_full --log-root logs --npz-dir data/npz/lafan1 \
    --num-envs 4096 --replay-capacity 1000 --episode-length-s 10.0 --anchor-obs \
    --amp --compile default \
    --wm-hidden 2048 2048 2048 --wm-window 8 --wm-batch 2048 --wm-batches 10 \
    --pi-batch 4096 --horizon 32 --critic-epochs 5 --critic-minibatches 4 \
    --term-head analytic --term-temp 0.05 \
    --lr 2e-4 --lr-decay-fraction 0.78125 \
    --total-iterations 320000 --updates-per-env-step 0.4 \
    --warmup 64 --policy-warmup-updates 0 \
    --log-interval 50 --checkpoint-interval 2000 --max-hours 9
```

`--total-iterations 320000` is the paper's budget and fixes the cosine learning-rate schedule; the
run was stopped at iteration 48,900 (the 48,000 checkpoint) because the evaluation had crossed the
target, so the learning rate never left its 2e-4 plateau (`lr_decay_fraction` starts decay only after
78 % of the budget). `--max-hours 9` was the hard safety cap.

Checkpoints (`ckpt_<iteration>.pt`, 206 MB each) are written every 2000 iterations next to
`config.json` and a TensorBoard event file. `ckpt_init.pt` is saved before the first update so that
"before vs after" is measured on one identical protocol.

## What the flags do

| flag | why it is set this way |
|---|---|
| `--anchor-obs` | adds the reference-anchor-relative pose to the goal group; **this is the deviation that makes the task solvable** — see [PAPER_DIFFERENCES.md](PAPER_DIFFERENCES.md) |
| `--amp` | fp16 tensor-core autocast for the MLP matmuls, with a `GradScaler`. Turing has fp16 tensor cores but no TF32/bf16 |
| `--compile default` | `torch.compile` on the world-model loss step and the imagined step, via `triton-windows` |
| `--term-head analytic` | the paper allows an analytic head (§3.2); it keeps the survival weight usable and removes a learned head from Eq. 1 |
| `--warmup 64` | environment steps before the first update: enough for 10-frame observation histories and 8-step windows to become sampleable |
| `--policy-warmup-updates 0` | this run had no world-model-only phase. The flag exists (and the loop honours it) for the alternative schedule used by other reproductions; it is inert at 0 |
| `--max-hours 9` | stop, cleanly, after 9 h of wall clock |

## Cost and throughput on this machine

| | |
|---|---|
| wall clock for 48,900 iterations | **7.93 h** |
| steady-state rate | ≈ 1.8 iterations/s (≈ 0.54 s/iteration) |
| environment steps | 63,400 env-steps/s (4096 envs, 0.064 s per vectorised step) |
| peak GPU memory | ≈ 21.9 GB of 24 GB with the evaluation watcher running concurrently |

Per-iteration breakdown from the run's own logging (per 50 iterations): simulator 3.4 s, world model
6.0 s, policy 17.9 s → **0.068 s / 0.12 s / 0.36 s per iteration**. The policy BPTT dominates.

## Why these speed choices

Measured on this TITAN RTX with `scripts/bench_full.py` (steady-state, world-model update of 10
minibatches × 2048 windows × k = 8, policy BPTT of H = 32 at batch 4096):

| configuration | world-model update | policy update | iteration |
|---|---|---|---|
| fp32, eager | 1.36 s | 6.95 s | 3.39 s |
| **fp16 AMP + `torch.compile`** | **0.30 s** | **1.21 s** | **0.54 s** |

Two things mattered, and both were validated rather than assumed:

* **fp16 instead of bf16.** sm_75 has no bf16 tensor cores; a 2048³ matmul runs at 73.6 TFLOPS in
  fp16 vs 10.8 TFLOPS in fp32 (6.8×). `scripts/validate_amp.py` runs the real trainers with and
  without `--amp` from the same seed and prints the metric trajectories, so the fp16 path was compared
  against fp32 before it was used. The `GradScaler` skips a handful of initial overflow steps and then
  settles; the fp16 and fp32 world-model losses track each other closely.
* **`torch.compile` on Windows.** Upstream disables it because Inductor wants Triton and Triton has no
  official Windows wheel. `triton-windows` provides one; with it, the world-model loss step gets 4.6×
  and the policy BPTT 5.7× faster. Compiling costs ~90 s + ~14 min once at start-up.

If you cannot install `triton-windows`, drop `--compile default`. Everything remains correct, at
roughly 3× the wall clock, which would put this run at ~24 h instead of 8 h.

The other throughput lever is the simulator's constraint headroom: the environment is configured with
`nconmax=128, njmax=500` instead of the upstream defaults (35/250), because a policy that frequently
falls can overflow the MuJoCo-Warp constraint buffer during training.

## Budget estimation before committing

```bash
python scripts/bench_full.py --anchor-obs --amp --compile default \
    --num-envs 4096 --wm-batches 10 --wm-batch 2048 --pi-batch 4096 --horizon 32 \
    --collect-steps 60 --bench-updates 6 --bench-iters 20 \
    --out bench.json
```

It collects a few dozen transitions so the replay buffer is sampleable, times the three costs
separately (it prints every per-call time, so a cold first call is visible), measures peak memory,
probes the fp16-vs-fp32 matmul speed of your card, and projects the wall clock of the requested
schedule. Do this before launching a multi-hour run: on this machine the first, naive estimate was
447 h; the measured, tuned number was 7.93 h for the budget actually spent.

## Monitoring during the run

Training prints one line per `--log-interval` iterations:

```
it 48900 | 7.93 h | 1.8 it/s | updates 19534 | train_ret 0.778 (n=863) | wm nse1 0.0085 nse8 0.0163 | imag_ret 2.806 surv 1.000 | t sim/wm/pi 3.4/6.0/17.9s
```

`train_ret` is the training environment's own episode return (10 s episodes, random start) — it is
*not* comparable to the whole-clip evaluation number. `wm nse1 / nse8` are the world model's online
1-step and 8-step open-loop normalised squared errors. `imag_ret / surv` are the imagined return and
the predicted survival probability at the horizon: watch `imag_ret` rise while the evaluation is flat
— that gap is model exploitation (see [LESSONS.md](LESSONS.md)).

```bash
python scripts/report_progress.py logs/lafan1_full      # compact table + per-family breakdown
```

## Practical notes

* Run the evaluation watcher as a **second process** (`scripts/eval_watch.py`). It costs ~7 minutes per
  checkpoint and briefly slows training by ~15 %, which is a good trade for knowing where the run is.
* Keep `--checkpoint-interval` small enough that a crash costs little. At 2000 iterations and 1.8 it/s,
  a checkpoint is ~18 minutes of work.
* Total disk for 25 checkpoints is ≈ 5 GB. Keep `ckpt_0048000.pt` (the best) and delete the rest if
  space matters — the evaluation JSONs in `results/` describe them all.