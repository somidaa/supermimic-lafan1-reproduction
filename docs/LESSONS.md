# Lessons: why the inherited result looked like a failure

This page separates what is **directly observed in artifacts** from what is **inferred**. The
distinction matters, because the inherited run and the run reported here changed several things at
once.

## A. Observations (log facts)

**A1. The inherited configuration was a heavily downscaled one.** From the transferred run's
`config.json` (not committed here; summarised in `configs/lafan1_full.json`'s sibling artifacts):

| | inherited `s3_core` | this run | ratio |
|---|---|---|---|
| policy updates | 798 | 19,534 | 24× |
| world-model minibatches per update | 2 | 10 | 5× |
| world-model minibatch size | 256 | 2048 | 8× |
| world-model width | 1024 (3 layers) | 2048 (3 layers) | 8.8 M vs 2.3 M params |
| policy BPTT batch | 128 | 4096 | 32× |
| environments | 2048 | 4096 | 2× |
| iterations | 1000 | 48,900 | 49× |
| own time cap | 1 h (`max_hours`) | 9 h cap, stopped at 7.93 h | — |
| `anchor_obs` | **false** | **true** | — |

**A2. Its evaluation was flat at the noise floor.** Measured here on the same 40 clips with the same
protocol:

| inherited checkpoint | policy updates | mean return | failed |
|---|---|---|---|
| `ckpt_init.pt` | 56 | 0.0018 | 40/40 |
| `ckpt_final.pt` | 798 | 0.0019 | 40/40 |

Policy rollouts terminated after ~40–42 control steps (0.8 s) on every clip.

**A3. Its world model was much less accurate than this run's.** From its own training log: final
1-step / 8-step open-loop NSE 0.083 / 0.316. This run: 0.0085 / 0.0163 (best 8-step 0.0123).

**A4. This run, changing the observation set and the budget together, reaches 0.8849 with 1/40
failures** — see [RESULTS.md](RESULTS.md).

**A5. (Second-hand, from the upstream project's own experiment log, not re-run here.)** That log
records the paper-observation configuration **plateauing at ≈ 0.56** on all 40 clips across three
independent runs, and the same code with the anchor-relative observation reaching 0.824 at 24k
iterations and 0.904 at 48k. It also attributes the plateau to the absence of yaw/position feedback,
measuring a systematic ≈ 1.7 °/s drift that exits the 15° band after ~9 s while the clips average
220 s.

## B. Inferences (labelled)

**B1. The observation set is the binding constraint, the budget is the second one.** *Inference.*
This is supported by A5 (which varies the observation at a fixed budget and sees 0.56 → 0.82) far more
than by anything measured here, because this run changed both. The mechanism is understandable: four
of the six reward terms are compared in the world frame, and without any yaw/position feedback the
policy cannot keep those comparisons meaningful over a 220 s clip however long it is trained.

**B2. The inherited policy did not fail because its action scale was too small.** *Inference, with
supporting evidence.* Scaling the trained policy's output by 1×/3×/10×/30× and re-evaluating made
survival worse (39–44 → 13–23 → 14–38 → 9–11 steps), so the failure is not a simple output-magnitude
problem. This is recorded in the inherited evidence set, not re-measured here.

**B3. World-model error alone does not explain the inherited failure.** *Inference.* An 8-step NSE of
0.316 is bad, but the reward difference between a good and a bad action at one step is only ~0.005–0.01
(the six terms are Δt-scaled and capped at 0.1 total), so a model error of that size is enough to let
the policy exploit it over a 32-step horizon. That is consistent with A3+A2 but is not a controlled
measurement; no model-error ablation was run.

**B4. "Small data volume" is not the right diagnosis for LAFAN1.** *Inference.* 2.451 h of motion over
40 clips is exactly the paper's Fig. 4a dataset, and this run reaches the paper's score on it. The
inherited run's problem was the *training* budget and the observation set, not the amount of motion
data. Data volume becomes the binding constraint only at the BONES-SEED scale, which is out of scope
here.

**B5. Imagined return is not a proxy for real tracking.** *Observed trend, interpreted.* In this run
the imagined undiscounted return rose to 2.81 while the real whole-clip return was still climbing
through 0.85 — the two are not on the same scale, and their gap is what model exploitation looks like.
The inherited run showed the extreme version of this: imagined return improved from 0.29 to 0.58 while
the real return stayed at 0.0018. **Never report imagined return as a result.**

## C. Practical rules that follow

1. **Budget first, then blame the method.** Before concluding that a method does not transfer, check
   the ratio of what you ran to what the paper ran. Here it was 0.25 % of the policy updates.
2. **Check the observation set against the task, not just against the paper's table.** If the reward
   is defined in a frame the policy cannot observe, more training will not fix it.
3. **Validate precision changes numerically.** fp16 on Turing was a 6.8× win on matmuls, but it was
   adopted only after `scripts/validate_amp.py` showed the fp32 and fp16 metric trajectories agreeing
   from the same seed. Precision changes silently change what the policy can learn.
4. **Measure per-clip, not just the mean.** The mean hid the fact that `fallAndGetUp*` was carrying
   most of the residual failures; the per-family figure made that obvious.
5. **Treat single-failure identifications as noisy.** Two evaluations of identical weights disagreed on
   which clip failed. Report counts, and re-evaluate before naming a culprit.
6. **Estimate the budget from a benchmark, not from arithmetic on paper.** The first estimate for this
   run was 447 h; the measured, tuned number was 7.93 h.