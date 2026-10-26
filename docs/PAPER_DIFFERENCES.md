# Differences from the paper, and the patches applied to the upstream code

## 1. The one difference that changes the task: `anchor_obs=True`

**What it does.** The policy's goal group gains the reference anchor pose expressed in the robot's
anchor frame — position (3) plus orientation as a 6D rotation (6), i.e. BeyondMimic's
`motion_anchor_pos_b` / `motion_anchor_ori_b`. The goal therefore grows from 680 to **689** dims and
the motion encoder's input from 680 to 689. Nothing else about the algorithm changes.

**Why it is here.** The paper's Table A.1 defines the *observation* set as pure proprioception
(joint position/velocity, base angular velocity, projected gravity, previous action) with **no**
anchor-relative pose, and the paper does not specify what each of the K future reference frames
contains. That unspecified goal content is the highest-risk gap in the specification, and the
consequence is measurable:

* with pure proprioception and no yaw/position feedback, the policy drifts systematically (≈ 1.7 °/s
  in the transferred experiment log), and LAFAN1 clips average 220 s, so after ~9 s the four
  world-frame reward terms are being evaluated far from the reference;
* the transferred experiment log (not committed here) records the paper-observation configuration
  **plateauing at ≈ 0.56** on all 40 clips across three separate runs, versus ≈ 0.82–0.90 for the
  same code with the anchor-relative observation.

**What it costs.** The policy now requires a state estimate (the reference anchor pose relative to the
robot's own anchor). That is IMU-observable in principle, but it is still an input that Table A.1 does
not list.

**What it therefore forbids claiming.** This run must **not** be presented as "beating the paper's
0.82 under the paper's settings". The honest claim is: *the paper's core method, at the paper's
budget, reaches the reported LAFAN1 score on this hardware, using a wider observation set.*

To run the paper-faithful variant, drop `--anchor-obs`. Expect the ~0.56 plateau described above
within ≈ 2 h of training; that is a property of the missing feedback, not of the hardware.

## 2. Mixed precision (fp16 instead of bf16/TF32)

Upstream has no AMP path at all. This repository adds an optional one:

* `src/supermimic/training/amp.py` (new, 26 lines) — an `AmpState` holding the autocast context and a
  per-optimizer `GradScaler`;
* `world_model/wm_trainer.py` — wrap the k-step loss in autocast and route the update through the
  scaler;
* `imagination/policy_trainer.py` — wrap the imagined rollout and the policy/critic losses the same
  way, and cast the critic's bootstrap value back to fp32 before it enters the policy objective.

Only the MLP matmuls are affected; state integration, the analytic heads and the forward kinematics
stay in fp32. The flag is `--amp`, default off, i.e. the code is bit-identical to upstream unless
asked otherwise. Measured on this Turing card: 2048³ matmul 73.6 TFLOPS fp16 vs 10.8 TFLOPS fp32.
`scripts/validate_amp.py` compares the two paths from one seed.

The paper specifies TF32; Turing has no TF32, and bf16 has no tensor-core support before sm_80, so
fp16 is the only accelerated option here. That is a hardware-driven substitution, not a tuning choice.

## 3. `torch.compile` on Windows

Not a code change: `--compile default` simply passes a mode through to the existing
`enable_compile()` hooks. Upstream's harnesses hard-disable it on Windows ("Windows has no Triton");
with the community `triton-windows` wheel it works, and it is worth 4.6× on the world-model loss step
and 5.7× on the policy BPTT. Use `--compile` unset to reproduce upstream behaviour exactly.

## 4. Simulator constraint headroom

`envs/tracking_env_cfg.py` changes `nconmax` 35 → **128** and `njmax` 250 → **500**. A policy that
falls frequently can overflow MuJoCo-Warp's constraint buffers, and the environment already silences
the overflow printf, so an overflow would corrupt replay data silently. This is a robustness fix, not
an algorithmic one; it does not change the dynamics when no overflow would have occurred.

## 5. World-model-only warm-up (implemented, not used)

`ScheduleCfg.policy_warmup_updates` (new field) and three lines in `training/loop.py` let the first N
updates train only the world model, matching the paper's Alg. A.1 ordering and the schedule used by
other reproductions. **The reported run used `--policy-warmup-updates 0`**, so its behaviour is that
of upstream: world-model and policy updates both start once `--warmup` environment steps have been
collected.

## 6. Analytic termination head

`--term-head analytic` softens the simulator's own thresholds (sigmoid of width τ = 0.05) into a
differentiable logit. The paper explicitly permits this ("each head is either learned or computed
analytically", §3.2). A learned BCE-fitted head is implemented and selectable with
`--term-head learned`; a learned head needs the positive rate to be non-negligible to be identifiable,
which is not guaranteed early in training.

## 7. Evaluation denominators

`eval/benchmark.py` divides by the **full** clip length rather than the survived length (see
[EVALUATION.md](EVALUATION.md)). Upstream's own comments identify the survived-length version as a bug
found in an earlier stage of that codebase; this repository keeps the corrected denominator.

## Summary table

| aspect | paper | this run |
|---|---|---|
| observation (Table A.1) | 93-dim proprioception, 10-frame history | same |
| goal content | unspecified | K = 10 future frames (68 dims each) **+ anchor-relative pose (9)** |
| world model | MLP 2048³, k = 8 | same |
| imagination | H = 32, frozen model, BPTT | same |
| critic / GAE | λ = 0.95, γ = 0.99 | same |
| precision | TF32 | **fp16** (Turing has no TF32) |
| envs / UTD / lr | 4096 / 0.4 / 2e-4 cosine | same |
| reward, termination | Table A.2; |e_pz| > 0.25 m, e_tilt > 0.8 | same (analytic heads, §3.2) |