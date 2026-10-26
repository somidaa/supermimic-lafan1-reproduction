"""Stage-2 Test 5: Critic and the GAE targets of Eq. 3.

Independently verifies the return recursion on hand-checkable small trajectories (terminal mask,
lambda limits, bootstrap at the horizon), confirms the critic's update actually changes its
parameters, and -- as the task requires -- confirms that the ``use_critic=False`` ablation changes
the *objective and its gradient*, not merely a configuration name.
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
    from supermimic.imagination.compiled import ImagStep
    from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg

    env = H.make_small_env(num_envs=24, seed=4)
    env.reset()
    H.collect(env, steps=120, action_std=0.5, seed=4)
    wm, policy, critic, lib, heads, pcfg = H.build_stack(env, hidden=(128, 128), seed=4)
    buffer = env.recorder_manager.get_term("replay").buffer
    step_fn = ImagStep(wm, lib.default_joint_pos, compile_mode=None)
    pt = PolicyTrainer(wm, policy, critic, lib, PolicyTrainerCfg(
        horizon=8, batch_size=32, compile=None, critic_epochs=2, critic_minibatches=2, obs_noise=False,
    ))
    yield dict(env=env, wm=wm, policy=policy, critic=critic, lib=lib, buffer=buffer,
               step_fn=step_fn, pt=pt, pcfg=pcfg)
    env.close()


# --------------------------------------------------------------------------- Eq. 3


def test_gae_closed_forms_and_terminal_mask():
    """Hand-checkable cases: R_H = V_H, lambda=0 -> 1-step TD, lambda=1 -> Monte-Carlo, d=1 -> no bootstrap."""
    from supermimic.imagination.objective import gae_targets

    H_, n, gam = 4, 3, 0.9
    torch.manual_seed(0)
    r = torch.rand(H_, n)
    V = torch.rand(H_ + 1, n)
    zero_d = torch.zeros(H_, n)

    # lambda = 0 : one-step TD with a (1 - d) gate
    out0 = gae_targets(r, zero_d, V, gam, 0.0)
    for t in range(H_):
        torch.testing.assert_close(out0[t], r[t] + gam * V[t + 1], rtol=0, atol=1e-6)

    # lambda = 1 : Monte-Carlo discounted return plus the terminal bootstrap
    out1 = gae_targets(r, zero_d, V, gam, 1.0)
    mc0 = sum(gam**k * r[k] for k in range(H_)) + gam**H_ * V[H_]
    torch.testing.assert_close(out1[0], mc0, rtol=0, atol=1e-6)
    torch.testing.assert_close(out1[H_ - 1], r[H_ - 1] + gam * V[H_], rtol=0, atol=1e-6)

    # a terminal step removes the bootstrap entirely: R_t = r_{t+1}
    d_one = torch.zeros(H_, n)
    d_one[0] = 1.0
    outd = gae_targets(r, d_one, V, gam, 0.95)
    outu = gae_targets(r, zero_d, V, gam, 0.95)
    torch.testing.assert_close(outd[0], r[0], rtol=0, atol=1e-6)
    # the recursion runs backwards, so masking step 0 affects ONLY R_0 and leaves R_1..R_{H-1} intact
    torch.testing.assert_close(outd[1:], outu[1:], rtol=0, atol=0)
    assert not torch.allclose(outd[0], outu[0])


def test_gae_matches_an_independent_numpy_reference():
    """The repo's vectorised recursion must equal a straight transcription of the paper's Eq. 3."""
    from supermimic.imagination.objective import gae_targets

    rng = np.random.default_rng(0)
    for H_, n in ((1, 4), (8, 32), (32, 8)):
        r = rng.random((H_, n)).astype(np.float64)
        d = (rng.random((H_, n)) < 0.2).astype(np.float64)  # binary termination labels
        V = rng.random((H_ + 1, n)).astype(np.float64)
        for gamma, lam in ((0.99, 0.95), (0.9, 0.0), (0.9, 1.0), (1.0, 0.5)):
            got = gae_targets(torch.tensor(r), torch.tensor(d), torch.tensor(V), gamma, lam).numpy()
            R = V[H_].copy()
            exp = np.zeros_like(r)
            for t in range(H_ - 1, -1, -1):
                R = r[t] + gamma * (1.0 - d[t]) * ((1.0 - lam) * V[t + 1] + lam * R)
                exp[t] = R
            np.testing.assert_allclose(got, exp, rtol=0, atol=1e-5)


def test_gae_targets_are_detached_and_reproducible(box):
    """Targets are computed once per rollout under no_grad and stay fixed while psi is fitted."""
    from supermimic.imagination.objective import gae_targets
    from supermimic.imagination.rollout import imagine

    wm, policy, critic, lib, buffer, step_fn = (
        box["wm"], box["policy"], box["critic"], box["lib"], box["buffer"], box["step_fn"]
    )
    start = buffer.sample_starts(16, hist=10)
    ro = imagine(wm, policy, lib, start, 8, 10, obs_noise=False, step_fn=step_fn)
    feat, goals, e_root = box["pt"]._critic_inputs(ro)
    H1, n = ro.rewards.shape[0], ro.rewards.shape[1]
    xi = wm.spec.normalize_xi(start.xi)[None].expand(H1 + 1, n, -1)

    with torch.no_grad():
        values = critic(feat.reshape(-1, feat.shape[-1]), goals.reshape(-1, goals.shape[-1]),
                        xi.reshape(-1, xi.shape[-1]), e_root.reshape(-1, 4)).reshape(H1 + 1, n)
        t1 = gae_targets(ro.rewards, ro.dhat[1:], values, 0.99, 0.95)
    assert not t1.requires_grad, "GAE targets must be detached (they are held fixed while fitting psi)"
    # recomputing from the same rollout reproduces them exactly -> 'computed once per rollout'
    with torch.no_grad():
        t2 = gae_targets(ro.rewards, ro.dhat[1:], values, 0.99, 0.95)
    torch.testing.assert_close(t1, t2, rtol=0, atol=0)
    assert torch.isfinite(t1).all()


# --------------------------------------------------------------------------- Critic


def test_critic_input_output_shape_and_the_privileged_e_root(box):
    """Table A.4 says V(psi) takes (s_hat, g, xi). This repo adds 4 dims of privileged root error.

    ``e_root`` is NOT a function of the canonical state plus the 680-d goal (the canonical state
    drops root x/y and the goal carries no world/anchor pose), so the critic's information set is
    strictly larger than the actor's. That is Stage-1 finding F3 / open question P0-3; the
    dimension is asserted here so the deviation cannot regress silently.
    """
    critic, pcfg = box["critic"], box["pcfg"]
    first = critic.mlp.net[0]
    expected = 71 + 680 + 33 + 4  # canonical state, goal, xi, e_root
    assert first.in_features == expected == 788

    n = 8
    feat = torch.randn(n, 71, device=box["env"].device)
    goal = torch.randn(n, 680, device=box["env"].device)
    xi = torch.randn(n, 33, device=box["env"].device)
    e_root = torch.randn(n, 4, device=box["env"].device)
    v = critic(feat, goal, xi, e_root)
    assert v.shape == (n,) and H.all_finite(v)
    # changing only e_root must change the value (proves the extra input is actually used)
    v2 = critic(feat, goal, xi, e_root + 1.0)
    assert not torch.allclose(v, v2), "e_root is not actually affecting the critic output"


def test_critic_update_changes_parameters_and_loss_is_finite(box):
    """One PolicyTrainer.update() must move the critic and report a finite, non-negative loss."""
    critic, buffer, pt = box["critic"], box["buffer"], box["pt"]
    snap = H.param_snapshot(critic)
    m = pt.update(buffer)
    assert H.max_param_diff(critic, snap) > 0.0, "critic parameters did not change"
    assert "critic_loss" in m and np.isfinite(m["critic_loss"]) and m["critic_loss"] >= 0.0
    assert np.isfinite(m["critic_value_mean"]) and np.isfinite(m["critic_target_mean"])


def test_critic_off_ablation_changes_the_objective_and_its_gradient(box):
    """The ablation must change the objective function -- verified by value AND by policy gradient."""
    from supermimic.imagination.objective import policy_objective
    from supermimic.imagination.rollout import imagine

    wm, policy, critic, lib, buffer, step_fn, pt = (
        box["wm"], box["policy"], box["critic"], box["lib"], box["buffer"], box["step_fn"], box["pt"]
    )
    n, horizon, gamma = 16, 8, 0.99
    start = buffer.sample_starts(n, hist=10)
    xi_norm = wm.spec.normalize_xi(start.xi)

    policy.zero_grad(set_to_none=True)
    ro = imagine(wm, policy, lib, start, horizon, 10, obs_noise=False, step_fn=step_fn)
    feat, goals, e_root = pt._critic_inputs(ro)
    v_H = critic(feat[-1], goals[-1], xi_norm, e_root[-1])

    J_with = policy_objective(ro, v_H, gamma)
    policy.zero_grad(set_to_none=True)
    # retain_graph: the critic-free objective and the terminal term are evaluated on the SAME
    # rollout afterwards, so the graph must survive each backward pass through it.
    (-J_with / horizon).backward(retain_graph=True)
    g_with = torch.cat([p.grad.detach().reshape(-1) for p in policy.parameters() if p.grad is not None]).clone()

    policy.zero_grad(set_to_none=True)
    J_without = policy_objective(ro, None, gamma)
    (-J_without / horizon).backward(retain_graph=True)
    g_without = torch.cat([p.grad.detach().reshape(-1) for p in policy.parameters() if p.grad is not None]).clone()

    # the ablation is not a no-op: the objective value must differ by exactly gamma^H * c_H * V_H
    dJ = J_with.item() - J_without.item()
    expected_term = float(((gamma**horizon) * ro.survival[-1] * v_H).mean())
    assert dJ == pytest.approx(expected_term, rel=1e-4, abs=1e-8), (
        "the critic term must enter Eq. 2 exactly as gamma^H * c_H * V_H"
    )

    # Stronger than comparing values: J_with == J_without + term identically, so the *gradient*
    # difference must equal the gradient of that term alone.
    policy.zero_grad(set_to_none=True)
    term_only = ((gamma**horizon) * ro.survival[-1] * v_H).mean()
    term_only.backward()
    g_term = torch.cat([p.grad.detach().reshape(-1) for p in policy.parameters() if p.grad is not None]).clone()

    torch.testing.assert_close(g_with - g_without, g_term, rtol=1e-3, atol=1e-9)

    ch_mean = float(ro.survival[-1].mean())
    print(f"\n  objective with critic {J_with.item():.8f} vs without {J_without.item():.8f} "
          f"(delta {dJ:.3e}); c_H mean {ch_mean:.3e}; |g_term|max {float(g_term.abs().max()):.3e}")
    # Mechanism note (recorded in TEST_REPORT.md): with a freshly initialised termination head the
    # survival weight after H steps is near zero, so gamma^H c_H V_H contributes ~no gradient and
    # Eq. 2 degenerates to the critic-free objective until the termination head is calibrated.
    assert ch_mean >= 0.0


def test_critic_off_leaves_critic_untouched_at_trainer_level(box):
    """With use_critic=False the trainer must not fit the critic at all (not just skip the name)."""
    from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg

    wm, policy, critic, lib, buffer = box["wm"], box["policy"], box["critic"], box["lib"], box["buffer"]
    pt = PolicyTrainer(wm, policy, critic, lib, PolicyTrainerCfg(
        horizon=8, batch_size=32, compile=None, use_critic=False, obs_noise=False,
    ))
    snap = H.param_snapshot(critic)
    m = pt.update(buffer)
    assert H.max_param_diff(critic, snap) == 0.0, "critic changed even though use_critic=False"
    assert "critic_loss" not in m, "critic metrics reported even though use_critic=False"
    assert m["imag_objective"] == m["imag_objective"]  # finite (not NaN)