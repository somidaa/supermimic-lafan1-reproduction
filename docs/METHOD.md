# Method

This page describes what the code implements. The mapping from paper sections to files is at the
bottom. Nothing here is specific to the hardware; see [TRAINING.md](TRAINING.md) for the budget.

## The algorithm (paper Algorithm A.1)

One loop, three interleaved activities, all online from the first iteration:

1. **Collect.** 4096 independent G1 environments are simulated with MuJoCo-Warp. Each environment
   tracks one clip at a time, starting from a state sampled out of the 40-clip library. The executed
   action is the policy's output plus Gaussian noise (σ = 0.1); the *clean* action is what the policy
   observes and what the world model is fitted on. Every step of every environment is streamed into a
   GPU replay buffer (state, clean action, executed action, simulator context ξ, reward, done,
   reference frame index).
2. **Fit the world model.** Sample windows of **k = 8** consecutive transitions and train the
   dynamics network on the k-step autoregressive prediction error (paper Eq. 1). 10 minibatches of
   2048 windows per update.
3. **Improve the policy.** Sample start states from the replay buffer, roll the policy forward
   **H = 32** steps *inside the frozen world model*, and back-propagate the discounted, survival-
   weighted return through the whole imagined trajectory into the policy (Eq. 2). A critic is fitted
   to GAE(λ = 0.95) targets computed on the same imagined transitions (Eq. 3) and supplies the
   bootstrap value at the horizon.

The three phases share the same wall-clock loop at an update-to-data ratio of **UTD = 0.4** (one
world-model + policy update per 2.5 environment steps), so early, imperfect data is still used to fit
the dynamics — the dynamics do not depend on the collecting policy.

## State, observation, action

* **Simulator state** `s = [root_pos(3), root_quat(4), joint_pos(29), root_lin_vel(3),
  root_ang_vel_body(3), joint_vel(29)]` (71 dims, MuJoCo layout, expressed relative to the
  environment origin). `src/supermimic/world_model/state_spec.py`.
* **World-model input features** are heading-invariant (height, yaw-relative 6D rotation,
  body-frame linear and angular velocity, joint position, joint velocity), so the model never has to
  represent global yaw. `StateSpec.canonical`.
* **World-model output** is a 70-dim vector of *frame-local increments* — velocity increments in the
  heading/body frame plus small position corrections applied on top of explicit constant-velocity
  integration. The last layer is **zero-initialised**, so at initialisation the model extrapolates at
  constant velocity (a physical prior) and training starts from a sane baseline.
* **Policy observation** (paper Table A.1, 93 dims per frame): `q − q₀`, `q̇`, base angular velocity,
  projected gravity, previous action — stacked over a **10-frame history** in term-major order
  (930 dims).
* **Goal** `g_t`: **K = 10 future reference frames** (680 dims), plus — in this reproduction — the
  reference-anchor pose relative to the robot anchor (9 dims), see
  [PAPER_DIFFERENCES.md](PAPER_DIFFERENCES.md).
* **Action**: residual joint-position targets `q_sp = q₀ + α · a` with mjlab's per-joint G1 action
  scale, 29 dims.

## Analytic heads (paper §3.2)

The paper allows each world-model head to be learned *or* computed analytically. Here the
observation, reward and termination heads are analytic functions of the predicted state, evaluated
through a **differentiable batched forward kinematics** of the G1 (14 tracked bodies):

* **reward** — the BeyondMimic tracking reward as specified in Table A.2 (body position / orientation /
  linear velocity / angular velocity, root position / orientation, action-rate and joint-limit
  regularisers), with the reference bodies re-anchored to the robot's xy position and yaw, all terms
  scaled by Δt = 0.02 s;
* **termination** — the simulator's own rule (|pelvis height error| > 0.25 m or tilt error > 0.8)
  softened by a sigmoid of width τ = 0.05 into a differentiable logit, so the survival weight
  `c_t = Π(1 − d̂)` stays informative. A `learned` head (BCE-fitted, Eq. 1) is also implemented and
  selectable with `--term-head learned`;
* **observation** — the noise-free 93-dim proprioception above, reconstructed from the predicted
  state.

Because the heads are analytic, the *only* learned component of the dynamics is the MLP; that is what
makes BPTT through 32 imagined steps affordable and keeps the imagined reward consistent with the
simulator's reward function.

## Networks

| component | shape | parameters |
|---|---|---|
| dynamics MLP | (71 + 29 + 33) → 2048 → 2048 → 2048 → 70, SiLU, zero-init last layer | 8.81 M |
| motion encoder | 689 → 2048 → 1024 → 512 → 256 → 32 (tanh) | |
| policy decoder | (32 + 930) → 2048 → 1024 → 512 → 256 → 29, last layer gain 0.01 | 8.91 M total |
| critic | (71 + 689 + 33 + 4) → 256 → 128 → 1 | 0.237 M |

## Paper section → file

| paper | code |
|---|---|
| Table A.1 observation, 10-frame history | `envs/tracking_env_cfg.py`, `world_model/heads.py` |
| Table A.2 reward | `envs/mdp/rewards.py` |
| termination thresholds | `envs/mdp/terminations.py`, `world_model/world_model.py` |
| §3.1 collect, domain randomisation | `envs/tracking_env_cfg.py`, `envs/mdp/commands.py`, `data/recorder.py` |
| Eq. 1 (k-step world model) | `world_model/world_model.py`, `world_model/wm_trainer.py` |
| §3.3 imagination, Eq. 2 | `imagination/rollout.py`, `imagination/objective.py`, `imagination/policy_trainer.py` |
| Eq. 3 (critic, GAE) | `imagination/objective.py::gae_targets`, `imagination/policy_trainer.py` |
| Alg. A.1 outer loop | `training/loop.py` |
| Appendix F benchmark | `eval/benchmark.py` |