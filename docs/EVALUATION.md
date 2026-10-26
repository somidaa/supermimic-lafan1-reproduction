# Evaluation

## Protocol (paper Appendix F.1)

Implemented in `src/supermimic/eval/benchmark.py` and driven by `scripts/eval_watch.py`.

* each clip is played **once, from its first frame**, by a deterministic policy;
* **no** observation noise, **no** action noise, **no** reset noise, **no** domain randomisation
  (foot friction fixed at 1.0), deterministic physics;
* one environment per clip, all clips in parallel (40 envs for the 40-clip library);
* the episode ends at the clip's last frame (`time_out`) or when the training termination fires;
* a clip **fails** if the training termination fires before the clip ends: absolute pelvis height
  error > 0.25 m or tilt error > 0.8;
* **normalised return** = Σ reward / (0.1 × **full** clip length in control steps), where 0.1 is the
  largest possible per-step reward (Δt × the six tracking weights). Dividing by the *survived* length
  instead would let an early termination score as if it had completed, which contradicts the paper's
  "a policy that terminates early therefore scores low".

Reported aggregates: `mean_norm_return` (unweighted mean over the 40 clips) and
`step_weighted_norm_return` (weighted by each clip's full length).

## Running it

Score every checkpoint of a run, in a separate process, as it appears:

```bash
python scripts/eval_watch.py --log-dir logs/lafan1_full --npz-dir data/npz/lafan1 \
    --num-envs 40 --poll-seconds 90 --target-return 0.82
```

* `--once` scores whatever checkpoints exist and exits (use it to score a single checkpoint);
* results append to `<log-dir>/eval_watch.csv` and one JSON per checkpoint
  (`eval_<stem>.json`) is written next to it;
* it skips `ckpt_latest.pt` on purpose — that file duplicates the numbered checkpoint written in the
  same iteration, and scoring it again doubles the load on the training GPU;
* with `--target-return 0.82` it writes `<log-dir>/TARGET_REACHED` and exits once a checkpoint reaches
  that return **with zero failures**. In this run that never triggered (the best is 1/40), so the run
  was stopped by hand at the 48,000 checkpoint.

To reproduce the numbers in [RESULTS.md](RESULTS.md) from the shipped artifacts:

```bash
python scripts/slim_eval_csv.py results/eval_watch.csv     # already applied; dedupes by iteration
python scripts/make_figures.py --run-log logs/lafan1_full_stdout.log \
    --run-dir logs/lafan1_full --results-dir results --figures-dir results/figures
```

The upstream single-checkpoint evaluator `scripts/stage3_eval.py` is also shipped; it takes
`--ckpt`, `--npz-dir`, `--clips`, `--train-clips`, `--num-envs` and `--out`, and additionally splits
in-distribution from held-out clips. The numbers reported here were produced by `eval_watch.py`, which
uses the same `run_benchmark` underneath.

## Evaluation JSON schema

```jsonc
{
  "ckpt": "logs/lafan1_full/ckpt_0048000.pt",
  "iteration": 48000,
  "num_updates": 19174,
  "protocol": "Appendix F.1: first-frame start, no noise, no DR, friction 1.0, failure = training termination",
  "normalized_return": "sum(reward) / (0.1 * full clip length)",
  "all": { "clips": 40, "failed": 1, "mean_norm_return": 0.8849,
           "step_weighted_norm_return": 0.8899, "terms": { "...": 0.0 } },
  "clips": [ { "name": "...", "frames": 0, "steps": 0, "denominator_steps": 0,
               "norm_return": 0.0, "failed": false, "fail_step": -1,
               "survival_fraction": 1.0, "terms": { "...": 0.0 } } ]
}
```

`steps` counts control steps survived; `denominator_steps` is the clip's full length in control steps;
`terms` are per-reward-term means per survived step (Δt-scaled, so a weight-1 term caps at 0.02 and a
weight-0.5 root term caps at 0.01).

## Reproducibility of the evaluation itself

Two evaluations of the same weights in separate processes differ by **≈ 0.006** in mean return and can
flag a **different** clip as the single failure (details in [RESULTS.md](RESULTS.md#4-evaluation-reproducibility--read-this-before-quoting-a-number)).
The protocol removes all *deliberate* randomness, so this comes from the simulator stack — MuJoCo-Warp
contact solving and CUDA-graph replay — not from the policy or the observation noise settings.

Implications:

* quote the failure **count** (it is stable in the runs performed here), and do not over-interpret the
  identity of the marginal failing clip;
* ±0.006 is an observed spread between two runs on one machine, **not** a confidence interval — no
  repeated-trials estimate was made, so do not present it as a standard error;
* a checkpoint whose score sits within ~0.01 of the paper's 0.82 should not be described as "clearly
  above" it.

## What the numbers do and do not show

* They do **not** show that this method beats the paper under the paper's observation set. The policy
  here observes a reference-anchor-relative pose that Table A.1 does not contain
  ([PAPER_DIFFERENCES.md](PAPER_DIFFERENCES.md)).
* They do show that the core method (real simulator data collection → 8-step autoregressive world
  model → BPTT through the frozen model for 32 imagined steps → critic with GAE) reaches the reported
  LAFAN1 score at the reported scale, on a single 24 GB Turing card in under 8 hours.
* The evaluation is **whole-clip**: the LAFAN1 clips average 220 s, and the policy is evaluated over
  that whole horizon. This is the stricter reading of Fig. 4a; the paper says only that each
  checkpoint is scored on all 40 clips.