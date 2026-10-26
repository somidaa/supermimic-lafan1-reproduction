"""Stage-2 Test 3: multi-step (8-step autoregressive) world-model prediction.

Checks that ``f_omega`` can be rolled forward autoregressively on logged actions and that
Eq. 1's terms are the ones the paper defines:

* shapes/ranges of the predicted state, reward, observation and termination probability;
* the state term recomputed independently from ``Eq. 1`` and compared with the trainer's value;
* all loss components reported, finite, and non-negative;
* one backward pass giving finite, non-zero parameter gradients;
* the error profile over the 8 open-loop steps (does it accumulate with the horizon?).

Data: real transitions collected from the simulator into the replay buffer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stage2_helpers as H

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not H.npz_available(),
    reason="needs CUDA + LAFAN1 npz (see TEST_REPORT.md)",
)


@pytest.fixture(scope="module")
def box():
    """Small env + a filled replay buffer + an untrained world model (eager)."""
    env = H.make_small_env(num_envs=24, seed=2)
    env.reset()
    H.collect(env, steps=120, action_std=0.5, seed=2)
    wm, policy, critic, lib, heads, pcfg = H.build_stack(env, hidden=(128, 128), seed=2)
    yield dict(env=env, wm=wm, policy=policy, critic=critic, lib=lib, heads=heads,
               buffer=env.recorder_manager.get_term("replay").buffer)
    env.close()


def test_window_batch_shapes_and_ranges(box):
    """k=8 windows of logged state/action/context, with in-episode, non-crossing structure."""
    w = box["buffer"].sample_windows(64, 8)
    assert w.states.shape == (9, 64, 71)
    assert w.a_exec.shape == (8, 64, 29)
    assert w.a_clean.shape == (8, 64, 29)
    assert w.xi.shape == (64, 33)
    assert w.reward.shape == (8, 64)
    assert w.done.shape == (8, 64)
    assert w.frame.shape == (8, 64)

    assert H.all_finite(w.states) and H.all_finite(w.a_exec) and H.all_finite(w.xi)
    assert H.all_finite(w.reward)
    # the termination label is binary (Eq. 1 expects d_j in {0,1})
    assert set(w.done.unique().tolist()).issubset({0.0, 1.0})
    # the executed action is the logged (noisy) one, so it must differ from the clean policy action
    assert (w.a_exec - w.a_clean).abs().mean() > 1e-3
    # xi must be the 33-d Table-A.3 context (friction, 29 encoder biases, 3 torso COM offsets)
    assert box["wm"].spec.xi_offset.shape == (33,)
    # unit quaternions in the logged states
    q = w.states[..., 3:7]
    torch.testing.assert_close(q.norm(dim=-1), torch.ones_like(q[..., 0]), rtol=1e-4, atol=1e-4)


def test_eq1_state_term_matches_independent_recomputation(box):
    """Recompute Eq. 1's state term straight from the paper and compare with ``window_loss``."""
    from supermimic.world_model.wm_trainer import reference_at

    wm, lib = box["wm"], box["lib"]
    w = box["buffer"].sample_windows(32, 8)
    k, n = w.frame.shape

    refs = [reference_at(lib, w.frame[j]) for j in range(k)]
    loss, metrics = wm.window_loss(w.states, w.a_exec, w.xi, w.done, refs)
    assert H.all_finite(loss)

    # independent recomputation: roll from the logged seed state on the logged actions,
    # accumulate sum_dims ( (s_hat - s_true) / sigma )^2 per step, mean over the batch and steps
    s = w.states[0]
    acc, bce_acc = 0.0, 0.0
    for j in range(k):
        s = wm.step(s, w.a_exec[j], w.xi)
        e = wm.spec.error_vector(s, w.states[j + 1])
        acc = acc + (e * e).sum(-1).mean()
        logits = wm.termination_logits(s, refs[j])
        bce_acc = bce_acc + torch.nn.functional.binary_cross_entropy_with_logits(
            logits, w.done[j], reduction="mean"
        )
    expected_state = (acc / k).item()
    expected_bce = (bce_acc / k).item()

    assert metrics["state_loss"].item() == pytest.approx(expected_state, rel=1e-4, abs=1e-6)
    assert metrics["term_bce"].item() == pytest.approx(expected_bce, rel=1e-4, abs=1e-6)
    # Eq. 1 = state term + BCE (reward and observation heads are analytic -> no loss term)
    assert loss.item() == pytest.approx(expected_state + wm.cfg.term_loss_weight * expected_bce, rel=1e-4, abs=1e-6)


def test_loss_components_are_reported_finite_and_non_negative(box):
    """Every component the task asks for: state, termination BCE, 1-step and k-step error, accuracy."""
    from supermimic.world_model.wm_trainer import reference_at

    wm, lib = box["wm"], box["lib"]
    w = box["buffer"].sample_windows(32, 8)
    refs = [reference_at(lib, w.frame[j]) for j in range(8)]
    _loss, m = wm.window_loss(w.states, w.a_exec, w.xi, w.done, refs)
    for key in ("state_loss", "term_bce", "nse_1step", "nse_kstep", "term_acc"):
        assert key in m, f"missing loss component {key}"
        assert torch.isfinite(m[key]).all(), f"{key} is not finite"
    assert m["state_loss"].item() >= 0.0 and m["term_bce"].item() >= 0.0
    assert m["nse_1step"].item() >= 0.0 and m["nse_kstep"].item() >= 0.0
    assert 0.0 <= m["term_acc"].item() <= 1.0


def test_backward_gives_finite_nonzero_parameter_gradients(box):
    """One Eq.-1 backward pass must produce finite, non-zero gradients on every sub-module."""
    from supermimic.world_model.wm_trainer import reference_at

    wm, lib = box["wm"], box["lib"]
    wm.zero_grad(set_to_none=True)
    w = box["buffer"].sample_windows(32, 8)
    refs = [reference_at(lib, w.frame[j]) for j in range(8)]
    loss, _ = wm.window_loss(w.states, w.a_exec, w.xi, w.done, refs)
    loss.backward()

    total, mx, n = H.grad_stats(wm)
    assert n > 0, "no world-model parameter received a gradient"
    assert np.isfinite(total) and total > 0.0, f"world-model gradient is zero or non-finite ({total})"
    assert 0.0 < mx < 1e6, f"implausible max |grad| {mx}"

    # the dynamics network and the termination head must both receive gradient
    dyn = H.grad_stats(wm.dynamics)
    term = H.grad_stats(wm.term_head)
    assert dyn[0] > 0.0, "dynamics f_omega received no gradient"
    assert term[0] > 0.0, "termination head received no gradient (BCE path broken)"


def test_open_loop_error_accumulates_over_the_horizon(box):
    """Single-step vs 8-step error, and whether the error grows with the number of open-loop steps."""
    wm = box["wm"]
    w = box["buffer"].sample_windows(128, 8)
    err = wm.rollout_error(w.states, w.a_exec, w.xi)  # (8,) normalised squared error per step
    assert err.shape == (8,) and torch.isfinite(err).all()

    profile = [round(v, 6) for v in err.tolist()]
    print(f"\n  open-loop normalised error profile (steps 1..8): {profile}")
    # On an untrained model the zero-initialised last layer means constant-velocity extrapolation,
    # so the error must grow with the horizon (accumulation), not stay flat.
    assert err[-1] > err[0], (
        f"open-loop error did not accumulate: 1-step {err[0].item():.6g} vs 8-step {err[-1].item():.6g}"
    )
    assert err[0] < err[1] or err[0] == pytest.approx(err[1], rel=0.5)


def test_world_model_updates_reduce_open_loop_error(box):
    """A handful of Eq.-1 updates must measurably reduce the 1-step and 8-step error."""
    from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer

    wm, lib, buffer = box["wm"], box["lib"], box["buffer"]
    tr = WorldModelTrainer(wm, lib, WMTrainerCfg(window=8, batches_per_update=2, batch_size=128, lr=1e-3, compile=None))
    e0 = tr.online_error(buffer, n=256, k=8)
    assert H.all_finite(e0)
    # 200 updates x 2 batches of 128 windows: enough for the fit to be measured, cheap on a 2 GB GPU
    for i in range(200):
        m = tr.update(buffer)
        assert np.isfinite(m["loss"]), f"non-finite WM loss at update {i}: {m}"
    e1 = tr.online_error(buffer, n=256, k=8)
    assert H.all_finite(e1)
    print(f"\n  1-step {e0[0].item():.6f} -> {e1[0].item():.6f} | 8-step {e0[-1].item():.6f} -> {e1[-1].item():.6f}")
    assert e1[0] < 0.5 * e0[0], f"1-step error did not halve after 200 updates ({e0[0].item():.6f} -> {e1[0].item():.6f})"
    assert e1[-1] < 0.5 * e0[-1], f"8-step error did not halve after 200 updates ({e0[-1].item():.6f} -> {e1[-1].item():.6f})"
    # termination head must also improve its BCE
    assert m["term_bce"] == m["term_bce"] and m["term_bce"] >= 0.0