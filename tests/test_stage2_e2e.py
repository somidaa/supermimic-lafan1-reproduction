"""Stage-2 Test 6: minimal end-to-end closed loop.

Runs the whole Algorithm-A.1 cycle at a deliberately small scale -- collect, replay-buffer write,
world-model fit, imagined rollout, policy update, critic update, checkpoint save/restore -- and
checks that it is reproducible under a fixed seed and free of numerical anomalies.

Scale is small on purpose (16 envs, H=8, 40 iterations, reduced widths) to fit a 2 GB GPU. This is a
*correctness* closure, not a performance run: nothing here is evidence about the paper's results.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

import stage2_helpers as H

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not H.npz_available(),
    reason="needs CUDA + LAFAN1 npz (see TEST_REPORT.md)",
)


def make_train_cfg(log_root, seed: int, run_name: str):
    """A small but complete TrainConfig. compile=None because Triton is unavailable on Windows."""
    from supermimic.config import TrainConfig

    cfg = TrainConfig(run_name=run_name, log_root=log_root, seed=seed)
    cfg.data.npz_dir = H.NPZ_DIR
    cfg.data.clips = ["walk1_subject1"]
    cfg.env.num_envs = 16
    cfg.env.episode_length_s = 5.0
    cfg.env.replay_capacity = 200
    cfg.world_model.hidden = (128, 128)
    cfg.world_model.term_hidden = (32, 32)
    cfg.wm_trainer.batch_size = 64
    cfg.wm_trainer.batches_per_update = 1
    cfg.wm_trainer.compile = None
    cfg.policy.encoder_hidden = (64, 32)
    cfg.policy.decoder_hidden = (64, 32)
    cfg.critic.hidden = (32, 16)
    cfg.policy_trainer.batch_size = 32
    cfg.policy_trainer.horizon = 8
    cfg.policy_trainer.compile = None
    cfg.policy_trainer.critic_epochs = 1
    cfg.policy_trainer.critic_minibatches = 2
    cfg.schedule.total_iterations = 40
    cfg.schedule.warmup_iterations = 20
    cfg.schedule.log_interval = 20
    cfg.schedule.checkpoint_interval = 40
    return cfg


def _run_once(log_root, seed, run_name):
    """Run one minimal closure and return (trainer, captured state)."""
    from supermimic.training.loop import Trainer

    cfg = make_train_cfg(log_root, seed, run_name)
    tr = Trainer(cfg)
    captured = {
        "init": {m: H.param_snapshot(getattr(tr, m)) for m in ("wm", "policy", "critic")},
    }
    obs, _ = tr.env.reset()
    captured["reset_obs_policy"] = obs["policy"].detach().clone()
    captured["reset_obs_goal"] = obs["goal"].detach().clone()
    tr.run()
    captured["iteration"] = tr.iteration
    captured["num_updates"] = tr.num_updates
    captured["buffer_size"] = tr.buffer.size
    captured["bin_return"] = tr.command.bin_return.detach().clone()
    captured["bin_memory"] = tr.command.bin_memory.detach().clone()
    captured["final"] = {m: H.param_snapshot(getattr(tr, m)) for m in ("wm", "policy", "critic")}
    captured["max_state_abs"] = float(tr.buffer.s[tr.buffer.valid].abs().max())
    captured["ckpt"] = tr.log_dir / "ckpt_latest.pt"
    tr.env.close()
    return tr, captured


def test_minimal_closure_runs_and_updates_every_component(tmp_path):
    """All six stages of the loop must actually execute and leave a trace."""
    tr, cap = _run_once(tmp_path, seed=11, run_name="s2_min")

    assert cap["num_updates"] > 0, "no gradient update happened"
    assert cap["iteration"] == 40
    assert cap["buffer_size"] > 40 * 16 * 0.5, f"replay buffer barely filled ({cap['buffer_size']})"
    assert np.isfinite(cap["max_state_abs"]) and cap["max_state_abs"] > 0

    # world model, policy and critic must all have moved
    for name in ("wm", "policy", "critic"):
        d = H.max_param_diff(getattr(tr, name), cap["init"][name])
        assert d > 0.0, f"{name} did not change during training"
        assert np.isfinite(d), f"{name} diverged to non-finite weights"

    # checkpoints and logs
    assert cap["ckpt"].exists(), "checkpoint was not written"
    assert (tr.log_dir / "config.json").exists(), "config.json was not written"
    cfg_on_disk = json.loads((tr.log_dir / "config.json").read_text())
    assert cfg_on_disk["env"]["num_envs"] == 16
    assert cfg_on_disk["policy_trainer"]["horizon"] == 8
    tb_files = list((tr.log_dir / "tb").glob("events.out.tfevents.*"))
    assert tb_files, "no TensorBoard event file was produced (logs incomplete)"


def test_no_numerical_anomalies_anywhere_in_the_run(tmp_path):
    """Every parameter, buffer entry and sampled window must stay finite."""
    tr, cap = _run_once(tmp_path, seed=12, run_name="s2_finite")
    for name in ("wm", "policy", "critic"):
        for k, v in getattr(tr, name).state_dict().items():
            assert torch.isfinite(v).all(), f"non-finite weight {name}.{k}"
    buf = tr.buffer
    for field in ("s", "a", "a_exec", "xi", "r", "done"):
        t = getattr(buf, field)[buf.valid]
        assert torch.isfinite(t).all(), f"non-finite replay field {field}"
    w = buf.sample_windows(64, 8)
    assert H.all_finite(w.states) and H.all_finite(w.a_exec) and H.all_finite(w.xi)
    assert torch.isfinite(tr.command.bin_return).all()
    # (_run_once already closed the env handle)


def test_two_runs_with_the_same_seed_are_reproducible(tmp_path):
    """Same seed: identical initialisation and identical reset data; the full trajectory is NOT
    bit-reproducible and diverges by a measured, bounded amount.

    FINDING (see TEST_REPORT.md Test 6): two runs with the same seed produce bit-identical model
    initialisation and bit-identical reset observations, but after 40 iterations the trained weights
    differ by ~1e-4 and the sampler state by ~2e-4. Root cause is a sub-ulp (~1e-7) difference in the
    state written at reset across two env instances, amplified by the chaotic contact dynamics of a
    humanoid (qpos divergence reaches 8.9e-2 within 20 steps). ``torch.use_deterministic_algorithms``
    does not remove it. The assertion below therefore bounds the divergence instead of demanding
    bit-equality; the bound is 10x the measured value, so a real reproducibility regression still fails.
    """
    tr1, c1 = _run_once(tmp_path, seed=13, run_name="s2_rep_a")
    tr2, c2 = _run_once(tmp_path, seed=13, run_name="s2_rep_b")

    # (a) initialisation is fully determined by the seed
    for name in ("wm", "policy", "critic"):
        for k in c1["init"][name]:
            assert torch.equal(c1["init"][name][k], c2["init"][name][k]), f"seeding is not deterministic for {name}.{k}"

    # (b) the collected data starts identical (same seed -> same simulator RNG path)
    obs_delta = (c1["reset_obs_policy"] - c2["reset_obs_policy"]).abs().max().item()
    goal_delta = (c1["reset_obs_goal"] - c2["reset_obs_goal"]).abs().max().item()
    assert obs_delta == 0.0, f"reset observations differ between identical seeds (delta {obs_delta})"
    assert goal_delta == 0.0, f"reset goals differ between identical seeds (delta {goal_delta})"
    assert c1["iteration"] == c2["iteration"] and c1["num_updates"] == c2["num_updates"]
    assert c1["buffer_size"] == c2["buffer_size"]

    # (c) the whole trajectory: bounded divergence, not bit-equality
    finals = {}
    for name in ("wm", "policy", "critic"):
        finals[name] = max((c1["final"][name][k] - c2["final"][name][k]).abs().max().item() for k in c1["final"][name])
    bins = float((c1["bin_return"] - c2["bin_return"]).abs().max())
    print(f"\n  same-seed run-to-run delta: params {finals}, bin_return {bins:.3e}")
    assert all(np.isfinite(v) for v in finals.values())
    assert max(finals.values()) < 1e-3, f"same-seed divergence exceeded the documented bound: {finals}"
    assert bins < 1e-2, f"sampler state diverged beyond the documented bound: {bins}"


def test_same_seed_environment_inputs_are_identical_across_instances():
    """Isolate the divergence: all *inputs* to the environment are bit-identical under a fixed seed.

    Reference motion data, env origins, clip/frame selection and the domain-randomisation context xi
    all match exactly. The only mismatch is the state written at reset, at the ~1e-7 level (one fp32
    ulp in a root-quaternion component), which the dynamics then amplify.
    """
    def build():
        torch.manual_seed(99)
        return H.make_small_env(num_envs=8, seed=99)

    e1, e2 = build(), build()
    try:
        torch.manual_seed(7)
        e1.reset()
        torch.manual_seed(7)
        e2.reset()
        c1 = e1.command_manager.get_term("motion")
        c2 = e2.command_manager.get_term("motion")

        assert torch.equal(e1.scene.env_origins, e2.scene.env_origins)
        assert torch.equal(c1.frame, c2.frame), "clip/frame selection differs under the same seed"
        assert torch.equal(c1.library.ref(c1.frame).root_pos_w, c2.library.ref(c2.frame).root_pos_w)
        assert torch.equal(c1.library.ref(c1.frame).joint_pos, c2.library.ref(c2.frame).joint_pos)
        xi1 = e1.recorder_manager.get_term("replay").ctx.read()
        xi2 = e2.recorder_manager.get_term("replay").ctx.read()
        assert torch.equal(xi1, xi2), "domain-randomisation context differs under the same seed"
        assert torch.equal(e1.scene["robot"].data.root_link_pos_w, e2.scene["robot"].data.root_link_pos_w)

        state_delta = float((e1.sim.data.qpos[:] - e2.sim.data.qpos[:]).abs().max())
        torch.testing.assert_close(
            e1.sim.data.qpos[:], e2.sim.data.qpos[:], rtol=0, atol=1e-6,
            msg="reset state must agree to well within fp32 tolerance",
        )
        torch.manual_seed(5)
        acts = [torch.randn(8, 29, device=e1.device) * 0.4 for _ in range(20)]
        worst = 0.0
        for a in acts:
            e1.step(a)
            e2.step(a)
            worst = max(worst, float((e1.sim.data.qpos[:] - e2.sim.data.qpos[:]).abs().max()))
        print(f"\n  reset |dqpos| = {state_delta:.3e} (inputs identical) -> after 20 identical steps {worst:.3e}")
    finally:
        e1.close()
        e2.close()


def test_checkpoint_save_restore_roundtrip(tmp_path):
    """Save -> load into a fresh Trainer restores weights, optimizers and sampler state.

    FINDING (see TEST_REPORT.md Test 6): the checkpoint does **not** contain the replay buffer
    (``loop.py:109-124`` saves wm/policy/critic, their optimizers, ``bin_return``/``bin_memory`` and
    the counters, but no replay rows). So a resumed run can only update once the buffer has refilled;
    resuming with ``warmup_iterations`` below the restored iteration raises
    ``RuntimeError: replay buffer has no sampleable rows yet``. Because the warmup guard compares
    against the *absolute* iteration counter (``loop.py:180``), continuing after a resume requires
    setting ``warmup_iterations`` above the restored iteration. The repo's own smoke test does not
    catch this: it resumes at ``total_iterations`` and therefore never runs an update.
    """
    from supermimic.training.loop import Trainer

    tr1, cap = _run_once(tmp_path, seed=14, run_name="s2_ckpt")

    # document the gap structurally, so a future fix makes this assertion fail loudly
    ck = torch.load(cap["ckpt"], map_location="cpu", weights_only=False)
    assert not any("buffer" in k for k in ck), "checkpoint now contains a replay buffer: update this test and the docs"
    for required in ("wm", "policy", "critic", "wm_opt", "policy_opt", "critic_opt", "bin_return", "bin_memory"):
        assert required in ck, f"checkpoint is missing {required}"

    cfg2 = copy.deepcopy(make_train_cfg(tmp_path, seed=14, run_name="s2_ckpt_resume"))
    cfg2.resume = cap["ckpt"]
    # the buffer must refill before updates can resume; the warmup guard is absolute, so place it
    # above the restored iteration (40) with enough headroom for k-step windows.
    cfg2.schedule.total_iterations = 90
    cfg2.schedule.warmup_iterations = 60
    tr2 = Trainer(cfg2)
    try:
        assert tr2.iteration == cap["iteration"]
        assert tr2.num_updates == cap["num_updates"]
        for name in ("wm", "policy", "critic"):
            a, b = getattr(tr1, name).state_dict(), getattr(tr2, name).state_dict()
            for k in a:
                assert torch.equal(a[k], b[k]), f"{name}.{k} not restored"
        assert torch.equal(tr2.command.bin_return, cap["bin_return"]), "sampler bin_return not restored"
        assert torch.equal(tr2.command.bin_memory, cap["bin_memory"]), "sampler bin_memory not restored"

        before_pi = H.param_snapshot(tr2.policy)
        before_wm = H.param_snapshot(tr2.wm)
        tr2.run()
        assert tr2.iteration == 90
        assert tr2.num_updates > cap["num_updates"], "no further updates after resuming"
        assert H.max_param_diff(tr2.policy, before_pi) > 0.0, "policy did not continue training after resume"
        assert H.max_param_diff(tr2.wm, before_wm) > 0.0, "world model did not continue training after resume"
    finally:
        tr2.env.close()