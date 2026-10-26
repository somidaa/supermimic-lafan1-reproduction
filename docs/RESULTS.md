# Results

Every number on this page comes from an artifact committed to this repository; nothing is
re-estimated by hand. Sources are named per table.

## 1. Headline

| policy | policy updates | mean normalised return | step-weighted | failed clips |
|---|---|---|---|---|
| transferred `s3_core/ckpt_init` (baseline) | 56 | 0.0018 | 0.0018 | 40/40 |
| transferred `s3_core/ckpt_final` (baseline) | 798 | 0.0019 | 0.0018 | 40/40 |
| this run, `ckpt_init` (before any update) | 0 | 0.0018 | — | 40/40 |
| **this run, `ckpt_0048000` — independent re-evaluation** | **19,174** | **0.8849** | **0.8899** | **1/40** |
| this run, `ckpt_0048000` — watcher run | 19,174 | 0.8909 | 0.8935 | 1/40 |

Sources: `results/baseline_transferred/eval_ckpt_{init,final}.json`, `results/eval_ckpt_init.json`,
`results/eval_best_independent_reeval.json`, `results/eval_ckpt_0048000.json`.

**Which number to quote.** The instruction for this reproduction is to lead with the *independent
re-evaluation*: **0.8849 mean, 1/40 failed**. The watcher's own run of the same weights recorded
**0.8909**; the two runs also disagree about *which* clip fails (see §4). Report both if the number
matters, and never present the difference as a confidence interval.

The paper's Fig. 4a reference is ≈ **0.82**. This reproduction uses a wider observation set than the
paper's Table A.1 (see [PAPER_DIFFERENCES.md](PAPER_DIFFERENCES.md)), so it does **not** claim to
exceed the paper under identical conditions.

## 2. Training curve (all 40 clips, every 2000 iterations)

Source: `results/eval_watch.csv`.

| iteration | policy updates | mean return | step-weighted | failed |
|---|---|---|---|---|
| 0 | 0 | 0.0018 | 0.0017 | 40 |
| 2000 | 774 | 0.0100 | 0.0094 | 40 |
| 4000 | 1574 | 0.3436 | 0.3601 | 17 |
| 6000 | 2374 | 0.4460 | 0.4690 | 14 |
| 8000 | 3174 | 0.4501 | 0.4717 | 17 |
| 10000 | 3974 | 0.5164 | 0.5286 | 15 |
| 12000 | 4774 | 0.6111 | 0.6234 | 10 |
| 14000 | 5574 | 0.6894 | 0.7004 | 3 |
| 16000 | 6374 | 0.7050 | 0.7157 | 8 |
| 18000 | 7174 | 0.6763 | 0.6940 | 9 |
| 20000 | 7974 | 0.7621 | 0.7712 | 4 |
| 22000 | 8774 | 0.7948 | 0.8070 | 5 |
| 24000 | 9574 | 0.8146 | 0.8267 | 3 |
| 26000 | 10374 | 0.7541 | 0.7698 | 7 |
| 28000 | 11174 | 0.8003 | 0.8044 | 5 |
| 30000 | 11974 | 0.8470 | 0.8570 | 3 |
| 32000 | 12774 | 0.8207 | 0.8340 | 6 |
| 34000 | 13574 | 0.8448 | 0.8543 | 3 |
| 36000 | 14374 | 0.8557 | 0.8644 | 3 |
| 38000 | 15174 | 0.8672 | 0.8709 | 1 |
| 40000 | 15974 | 0.8851 | 0.8905 | 1 |
| 42000 | 16774 | 0.8640 | 0.8725 | 3 |
| 44000 | 17574 | 0.8779 | 0.8852 | 2 |
| 46000 | 18374 | 0.8332 | 0.8420 | 3 |
| 48000 | 19174 | 0.8908 | 0.8935 | 1 |

The first checkpoint at or above the paper's 0.82 is **iteration 30,000** (0.847). The curve carries a
run-to-run band of roughly ±0.05 at the checkpoint level (see §4), so treat individual rows as noisy
samples of an increasing trend, not as a monotone sequence.

## 3. Per-motion-family breakdown

Source: `results/eval_best_independent_reeval.json` (the independent re-evaluation).

| family | clips | failed | mean return |
|---|---|---|---|
| walk | 12 | 0 | 0.92 |
| run | 4 | 0 | 0.88 |
| jumps | 3 | 0 | 0.87 |
| dance | 8 | 0 | 0.89 |
| sprint | 2 | 0 | 0.89 |
| fight | 3 | 0 | 0.89 |
| fightAndSports | 2 | 0 | 0.88 |
| fallAndGetUp | 6 | 1 | 0.79 |

The `fallAndGetUp` family is the hardest and contains the single failing clip; the per-clip figure
(`results/figures/fig_per_clip.png`) shows every clip's individual score.

## 4. Evaluation reproducibility — read this before quoting a number

Two evaluations of the *same* weights (`ckpt_0048000.pt`), run in separate processes on the same
machine with the same protocol and seed:

| run | mean return | step-weighted | failures | which clip failed |
|---|---|---|---|---|
| watcher (during training) | 0.8909 | 0.8935 | 1/40 | `jumps1_subject2`, fail step 10,609 |
| independent re-evaluation | 0.8849 | 0.8899 | 1/40 | `fallAndGetUp2_subject3`, fail step 4,912 |

* the **failure count** is stable (1/40 in both), the **identity** of the marginal failure is not;
* the **mean return** differs by 0.006 — about 0.7 % relative.

The protocol removes all deliberate randomness (no observation/action noise, no reset noise, no domain
randomisation, friction 1.0, deterministic policy), so this spread comes from the simulator stack
(MuJoCo-Warp contact solving and CUDA-graph replay), not from the policy. **±0.006 is an observed
run-to-run spread on this machine, not a statistical confidence interval** — no repeated-trials
estimate was made.

Consistent with this: `dance1_subject1` was flagged as the single failure at iteration 40,000 in the
watcher table, but a direct render of the same clip at iteration 48,000 runs the full 131.5 s without
terminating. Marginal clips sit right on the termination threshold.

## 5. World-model accuracy

Online open-loop normalised squared error (NSE) on replay windows, k = 8 — the same quantity the
training loop logs. Source: `results/train_metrics.csv`.

| | 1 step | 8 steps |
|---|---|---|
| final (iteration 48,900) | **0.0085** | **0.0163** |
| best seen during the run | 0.0082 | **0.0123** |
| transferred `s3_core` checkpoint | 0.083 | 0.316 |

The error axes are per-quantity normalised (`err_scale` = 0.05 m position, 0.1 rad rotation,
0.5 m/s linear velocity, 1.0 rad/s angular velocity, 0.1 rad joint position, 2.0 rad/s joint
velocity), so these are not metres and are only comparable within this repository.

## 6. Training cost

Source: the training loop's own stdout (summarised in `results/train_metrics.csv`).

| | |
|---|---|
| wall clock | 7.93 h (single TITAN RTX, including a full 40-clip evaluation every 2000 iterations) |
| iterations | 48,900 (stopped deliberately at the 48,000 checkpoint) |
| policy updates | 19,534 |
| environment steps | 200,294,400 (4096 envs × 48,900 iterations) |
| steady-state rate | ≈ 1.8 iterations/s ≈ 0.54 s per iteration |
| peak GPU memory | ≈ 21.9 GB of 24 GB (training + evaluation watcher running concurrently) |

Per-iteration breakdown at the end: simulator 3.4 s, world model 6.0 s, policy 17.9 s per 50
iterations → 0.068 s / 0.12 s / 0.36 s.

## 7. Citation

```bibtex
@inproceedings{liao2026supermimic,
  title     = {SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model},
  author    = {Liao, Qi and Shao, Yuxuan and Long, Jun and Zakka, Kevin and Horowitz, Roberto
               and Abbeel, Pieter and Sreenath, Koushil},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
  note      = {arXiv:2610.28480}
}
```

If you use this specific reproduction, please also link this repository and note the observation-set
deviation described in [PAPER_DIFFERENCES.md](PAPER_DIFFERENCES.md).