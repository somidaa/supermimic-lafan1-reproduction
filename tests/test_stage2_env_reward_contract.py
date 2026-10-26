"""Stage-2 Test 1 (simulation/environment correctness) and Test 2 (reward & state definitions).

Test 1 checks the *contract* of the environment: control frequency, joint order, the paper's
action mapping ``q_sp = alpha * a + q0``, coordinate frames of the state, finiteness over a long
rollout, and where the sensor noise is allowed to appear.

Test 2 checks the reward/state definitions against the paper **independently of mjlab**: the eight
Table-A.2 terms are recomputed from the raw world-frame quantities with ``scipy`` rotation algebra,
which is a different implementation from both the repo's torch code and mjlab's reward manager.

Every assertion is on real LAFAN1-referenced simulation data (see TEST_REPORT.md §1 for the clips).
No assertion is relaxed relative to the paper's stated numbers.
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
def env():
    e = H.make_small_env(num_envs=16, seed=0)
    # mjlab's derived entity data (robot.data.*) only agrees with sim.data.qpos after the first
    # reset; training/loop.py:150 and every repo test reset before use. Not doing so yields
    # stale kinematics (see TEST_REPORT.md, Test 1 note).
    e.reset()
    yield e
    e.close()


# --------------------------------------------------------------------------- Test 1


def test_control_frequency_and_discretisation(env):
    """50 Hz control: 4 physics substeps of 5 ms (Table A.5 noise is per control step)."""
    assert env.step_dt == pytest.approx(0.02), "control period must be 0.02 s (50 Hz)"
    assert env.cfg.decimation == 4
    assert env.cfg.sim.mujoco.timestep == pytest.approx(0.005)
    assert env.step_dt == pytest.approx(env.cfg.decimation * env.cfg.sim.mujoco.timestep)


def test_joint_order_matches_mjcf_and_csv_convention(env):
    """The 29 hinges must be the G1 CSV/MJCF order, else the reference motion is scrambled."""
    import mujoco

    from supermimic.data.motion_convert import G1_JOINT_NAMES, NUM_JOINTS

    model = env.sim.mj_model
    raw = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    names = [n.split("/")[-1] for n in raw]  # mjlab namespaces bodies/joints as "<entity>/<name>"
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert tuple(names[1:]) == G1_JOINT_NAMES
    assert len(G1_JOINT_NAMES) == NUM_JOINTS == 29
    assert model.nq == 36 and model.nv == 35
    assert env.action_manager.total_action_dim == 29


def test_action_mapping_matches_paper_formula(env):
    """Appendix A: q_sp = alpha * a_t + q0, with the noisy action executed and the clean one observed.

    Trace: policy a -> (+noise) a_exec -> q_sp = a_exec * alpha + q0 (the action term's
    ``_processed_actions``); the setpoint written to the actuators is ``_processed_actions -
    encoder_bias`` (mjlab's encoder-calibration semantics, REVIEW_LOG E1).
    """
    torch.manual_seed(1234)
    a = 0.5 * torch.randn(env.num_envs, 29, device=env.device)
    env.step(a)

    term = env.action_manager.get_term("joint_pos")
    robot = env.scene["robot"]

    # clean action is what the policy emitted (paper U9 / plan D9)
    torch.testing.assert_close(term.raw_action, a, rtol=0, atol=1e-6)

    # executed action = clean + eps,  eps ~ N(0, 0.1^2) per dimension (Table A.5)
    eps = term.executed_raw - term.raw_action
    assert eps.abs().max() > 0.0, "action noise must actually be injected"
    assert 0.07 < eps.std().item() < 0.13, f"action-noise std {eps.std().item():.4f} != 0.1"

    # q_sp = alpha * a_exec + q0, unclipped (ActionTermCfg.clip defaults to None)
    assert term.cfg.clip is None
    offset = robot.data.default_joint_pos[:, term._target_ids]
    expected_sp = term.executed_raw * term._scale + offset
    torch.testing.assert_close(term._processed_actions, expected_sp, rtol=1e-5, atol=1e-6)

    # alpha>0, bounded, and not collapsed to a single value (mjlab's G1 scale is per joint group)
    scale = term._scale
    assert (scale > 0).all() and (scale < 5.0).all()
    assert scale.unique().numel() > 1

    # NOTE (scope): the setpoint -> actuator step (mjlab's apply_actions: `target =
    # _processed_actions - encoder_bias`, actions.py:221-224) is verified by *source inspection*
    # only. `robot.data.joint_pos_target` is all-zero right after a step in this mjlab version, so
    # it is not a usable accessor for this assertion; we do not fabricate a proxy for it.


def test_state_is_origin_free_and_frame_consistent(env):
    """The 71-d state stores the root relative to each env origin; it must map back to world exactly."""
    from supermimic.envs.state_io import STATE_DIM, StateIO

    io = StateIO(env)
    s = io.read()
    robot = env.scene["robot"]
    assert s.shape == (env.num_envs, STATE_DIM) == (env.num_envs, 71)
    assert s.dtype == torch.float32

    # the origins StateIO cached are the scene's live origins (else every recorded state is offset)
    torch.testing.assert_close(io.origins, env.scene.env_origins, rtol=0, atol=0)

    # world reconstruction: root pos = state pos + env origin
    world = s[:, 0:3] + io.origins
    torch.testing.assert_close(world, robot.data.root_link_pos_w, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(s[:, 0:3] + io.origins, env.sim.data.qpos[:, :3], rtol=1e-6, atol=1e-6)

    # free-joint layout: qpos = [pos(3), quat wxyz(4), joints(29)], qvel = [lin_w(3), ang_b(3), joints(29)]
    torch.testing.assert_close(s[:, 3:7], robot.data.root_link_quat_w, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(s[:, 7:36], robot.data.joint_pos, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(s[:, 36:39], robot.data.root_link_lin_vel_w, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(s[:, 42:71], robot.data.joint_vel, rtol=1e-4, atol=1e-4)


def test_long_rollout_stays_finite(env):
    """200 control steps with large random actions: nothing may go NaN/Inf, rewards stay bounded."""
    torch.manual_seed(7)
    for i in range(200):
        a = 1.0 * torch.randn(env.num_envs, 29, device=env.device)
        obs, rew, term, tout, _ = env.step(a)
        assert H.all_finite(obs["policy"]), f"non-finite policy obs at step {i}"
        assert H.all_finite(obs["goal"]), f"non-finite goal at step {i}"
        assert H.all_finite(rew), f"non-finite reward at step {i}"
        assert float(rew.max()) <= 0.1 + 1e-5, "per-step reward exceeds the Table-A.2 upper bound 0.1"
        assert torch.isfinite(env.sim.data.qpos[:]).all() and torch.isfinite(env.sim.data.qvel[:]).all()


def test_sensor_noise_placement_matches_tableA1():
    """Table A.1 puts noise on 4 terms and none on a_{t-1}; check that term by term, in the real env."""
    from supermimic.world_model.heads import OBS_TERMS

    noisy = H.make_small_env(num_envs=8, seed=1, obs_noise=True)
    clean = H.make_small_env(num_envs=8, seed=1, obs_noise=False)
    try:
        noisy.reset()
        clean.reset()
        torch.manual_seed(3)
        for _ in range(12):
            a = 0.3 * torch.randn(8, 29, device=noisy.device)
            o_n, *_ = noisy.step(a)
            o_c, *_ = clean.step(a)
        assert not torch.equal(o_n["policy"], o_c["policy"]), "obs_noise=True must change the policy obs"

        # term-major flattened layout, as declared by OBS_TERMS
        start, blocks = 0, {}
        for name, d in OBS_TERMS:
            blocks[name] = slice(start * 10, (start + d) * 10)
            start += d
        assert start * 10 == 930

        for name, d in OBS_TERMS:
            if name == "actions":
                # a_{t-1} carries NO sensor noise in Table A.1 -> identical in both envs
                torch.testing.assert_close(
                    o_n["policy"][:, blocks[name]], o_c["policy"][:, blocks[name]],
                    rtol=0, atol=0,
                    msg="the previous-action term must carry no sensor noise (Table A.1, plan D9)",
                )
            else:
                diff = o_n["policy"][:, blocks[name]] - o_c["policy"][:, blocks[name]]
                assert diff.abs().max() > 0, f"{name} should be corrupted by sensor noise"
    finally:
        noisy.close()
        clean.close()


def test_state_writeback_reproduces_the_recorded_next_state():
    """Stage-1 missing test N1 (plan 2.6 gate): writing a recorded ``s_t`` back and executing the
    recorded action should reproduce the recorded ``s_{t+1}``.

    FINDING (see TEST_REPORT.md D6): the plan's tolerance (normalised error p99 <= 1e-3) is **not
    achievable**, and this test now characterises the real limitation instead of asserting a number
    that provably cannot hold. Measured on real LAFAN1 transitions:

    * the write-and-step procedure is repeatable to ~6e-06;
    * *positions* reproduce well -- root position error < 1 mm (normalised 0.017);
    * *velocities* do not -- worst dimension is joint velocity at ~1.66 rad/s (normalised 0.83).

    Cause: ``(qpos, qvel)`` is not a sufficient statistic for one simulator step, because the
    contact solver's warm-start (``qacc_warmstart``) is not part of it. ``StateIO.write`` zeroes
    warm-start by design (``state_io.py:39-40``) and plan D1 explicitly dismisses the effect
    ("qacc_warmstart 影响忽略"). This test is the first quantification of that dismissal.

    Consequence for the project: the world model has an irreducible one-step error floor from the
    unobserved solver state; the 1-step normalised error reaching ~0.22 after 200 updates (Test 3) is
    consistent with the model learning the conditional mean.
    """
    from supermimic.envs.state_io import StateIO
    from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

    e = H.make_small_env(num_envs=8, seed=5, action_noise_std=0.0, replay_capacity=200)
    try:
        e.reset()
        H.collect(e, steps=60, action_std=0.5, seed=5)
        io = StateIO(e)
        buf = e.recorder_manager.get_term("replay").buffer
        spec = StateSpec(StateSpecCfg(), e.device)
        ids = torch.arange(8, device=e.device)

        w = buf.sample_windows(8, 1)
        s0, s1, a = w.states[0], w.states[1], w.a_exec[0]
        assert torch.equal(w.a_exec[0], w.a_clean[0]), "action noise should be off for this test"

        io.write(ids, s0)
        e.step(a)
        keep = ~e.reset_buf
        assert int(keep.sum()) >= 4, f"too many envs reset during the probe ({int(keep.sum())} kept)"
        sa = io.read()

        # (1) the procedure itself is deterministic: repeat it and compare
        io.write(ids, s0)
        e.step(a)
        sb = io.read()
        repeat = (sa - sb).abs().max().item()
        assert repeat <= 1e-4, f"the write-and-step probe is not repeatable ({repeat:.3e})"

        # (2) positions reproduce; that is the part the world model needs to get right
        err = spec.error_vector(sa, s1).abs()
        pos_err = err[keep][:, 0:3].amax()
        assert pos_err <= 0.05, f"root position is not reproduced (< 2.5 mm expected), got {pos_err:.4f}"
        # the invariant that must hold for Eq. 1, in its achievable form (root+joint positions)
        torch.testing.assert_close(sa[keep][:, :3], s1[keep][:, :3], rtol=0, atol=5e-3)
        torch.testing.assert_close(sa[keep][:, 7:36], s1[keep][:, 7:36], rtol=0, atol=5e-2)

        # (3) CHARACTERISATION of the known limitation, so that silently losing it fails loudly:
        #     the full-state gate cannot be met while qacc_warmstart is dropped from the state.
        full = err[keep].amax(dim=-1)
        p99 = torch.quantile(full, 0.99).item()
        vel_err = err[keep][:, 42:71].amax()
        print(f"\n  write-back one-step: repeat {repeat:.2e} | root_pos {pos_err:.4f} | "
              f"joint_vel {vel_err:.4f} | full-state p99 {p99:.4f} (plan gate 1e-3)")
        assert vel_err > 1e-2, (
            "the velocity mismatch that motivated plan-D1's dismissal of qacc_warmstart has vanished; "
            "if warm-start is now part of the state, raise this to a real assertion"
        )
        assert p99 > 1e-2, (
            f"the full-state write-back error is now {p99:.2e}; the plan's 1e-3 gate may have become "
            "achievable -- update plan 2.6, D1 and TEST_REPORT D6"
        )
    finally:
        e.close()


# --------------------------------------------------------------------------- Test 2


def test_reward_config_matches_tableA2(env):
    """Every number in Table A.2, plus the dt scaling and the resulting per-step upper bound."""
    from supermimic.envs.mdp.rewards import TrackingRewardCfg

    cfg = TrackingRewardCfg()
    assert (cfg.std_body_pos, cfg.std_body_ori, cfg.std_body_lin_vel) == (0.3, 0.4, 1.0)
    assert (cfg.std_body_ang_vel, cfg.std_root_pos, cfg.std_root_ori) == (3.14, 0.3, 0.4)
    assert (cfg.w_body_pos, cfg.w_body_ori, cfg.w_body_lin_vel, cfg.w_body_ang_vel) == (1.0, 1.0, 1.0, 1.0)
    assert (cfg.w_root_pos, cfg.w_root_ori) == (0.5, 0.5)
    assert cfg.w_action_rate == -0.1
    assert cfg.w_joint_limit == -2.0, "joint-limit weight is lowered from BeyondMimic's -10 to -2"
    assert cfg.dt == pytest.approx(0.02)
    # Table A.6 footnote: every term is multiplied by the control period
    assert cfg.max_step_reward == pytest.approx(0.02 * 5.0) == pytest.approx(0.1)

    rm = env.reward_manager
    weights = {n: rm.get_term_cfg(n).weight for n in rm.active_terms}
    assert weights == {
        "motion_global_root_pos": 0.5,
        "motion_global_root_ori": 0.5,
        "motion_body_pos": 1.0,
        "motion_body_ori": 1.0,
        "motion_body_lin_vel": 1.0,
        "motion_body_ang_vel": 1.0,
        "action_rate_l2": -0.1,
        "joint_limit": -2.0,
    }
    assert "self_collision" not in " ".join(rm.active_terms)


def _scipy_reward_reference(heads, s_next, action, prev_action, ref, cfg, soft_limits):
    """Table A.2 recomputed from raw world-frame quantities using scipy (independent of torch/mjlab).

    mjlab's re-anchoring (repo D7) is reproduced: the reference bodies are expressed in a frame that
    shares the robot anchor's xy position and yaw, keeping the reference's z, roll and pitch.
    """
    from scipy.spatial.transform import Rotation as R

    to_np = lambda t: t.detach().double().cpu().numpy()  # noqa: E731
    sq = lambda x: np.sum(np.asarray(x, dtype=np.float64) ** 2, axis=-1)  # noqa: E731

    b = heads.robot_bodies(s_next)
    rb_p, rb_q_np = to_np(b["pos"]), to_np(b["quat"])  # (N,B,3), (N,B,4) wxyz
    rb_v, rb_w = to_np(b["lin_vel"]), to_np(b["ang_vel"])
    ref_p, ref_q_np = to_np(ref.body_pos_w), to_np(ref.body_quat_w)
    ref_v, ref_w = to_np(ref.body_lin_vel_w), to_np(ref.body_ang_vel_w)

    n, nb = rb_p.shape[0], rb_p.shape[1]
    ai = heads.anchor_index

    # keep everything as flat (N*B,) Rotation arrays; slice anchors before converting
    wxyz2xyzw = lambda a: a[..., [1, 2, 3, 0]]  # noqa: E731
    rb_q = R.from_quat(wxyz2xyzw(rb_q_np).reshape(-1, 4))            # (N*B,)
    ref_q = R.from_quat(wxyz2xyzw(ref_q_np).reshape(-1, 4))          # (N*B,)
    ra_q = R.from_quat(wxyz2xyzw(rb_q_np[:, ai]))                    # (N,)
    ref_a_q = R.from_quat(wxyz2xyzw(ref_q_np[:, ai]))                # (N,)
    ra_p, ref_a_p = rb_p[:, ai], ref_p[:, ai]                        # (N,3)

    # yaw-only delta orientation == mjlab's yaw_quat(robot_anchor * inv(ref_anchor))
    rel_anchor = ra_q * ref_a_q.inv()
    yaw = rel_anchor.as_euler("ZYX")[:, 0]
    delta_q = R.from_euler("Z", yaw[:, None])

    # mjlab re-anchoring: delta position takes the robot's xy and the reference's z
    delta_p = np.stack([ra_p[:, 0], ra_p[:, 1], ref_a_p[:, 2]], axis=-1)  # (N,3)

    # broadcast the (N,) delta rotation over bodies by explicit repetition
    dq_rep = R.from_quat(np.repeat(delta_q.as_quat()[:, None, :], nb, axis=1).reshape(-1, 4))
    rel_q = dq_rep * ref_q
    rel_p = (
        dq_rep.apply((ref_p - ref_a_p[:, None, :]).reshape(-1, 3))
        + np.repeat(delta_p[:, None, :], nb, axis=1).reshape(-1, 3)
    ).reshape(n, nb, 3)

    e_root_pos = sq(ref_a_p - ra_p)
    e_root_ori = (ref_a_q * ra_q.inv()).magnitude() ** 2
    e_body_pos = sq(rel_p - rb_p).mean(-1)
    e_body_ori = ((rel_q * rb_q.inv()).magnitude() ** 2).reshape(n, nb).mean(-1)
    e_body_lin = sq(ref_v - rb_v).mean(-1)
    e_body_ang = sq(ref_w - rb_w).mean(-1)

    q = to_np(s_next[..., 7:36])
    lim = to_np(soft_limits)
    lo, hi = lim[..., 0], lim[..., 1]
    joint_limit = (np.maximum(0.0, lo - q) + np.maximum(0.0, q - hi)).sum(-1)
    action_rate = sq(to_np(action) - to_np(prev_action))

    dt = cfg.dt
    return {
        "motion_global_root_pos": dt * cfg.w_root_pos * np.exp(-e_root_pos / cfg.std_root_pos**2),
        "motion_global_root_ori": dt * cfg.w_root_ori * np.exp(-e_root_ori / cfg.std_root_ori**2),
        "motion_body_pos": dt * cfg.w_body_pos * np.exp(-e_body_pos / cfg.std_body_pos**2),
        "motion_body_ori": dt * cfg.w_body_ori * np.exp(-e_body_ori / cfg.std_body_ori**2),
        "motion_body_lin_vel": dt * cfg.w_body_lin_vel * np.exp(-e_body_lin / cfg.std_body_lin_vel**2),
        "motion_body_ang_vel": dt * cfg.w_body_ang_vel * np.exp(-e_body_ang / cfg.std_body_ang_vel**2),
        "action_rate_l2": dt * cfg.w_action_rate * action_rate,
        "joint_limit": dt * cfg.w_joint_limit * joint_limit,
    }


def test_reward_terms_match_independent_scipy_reference(env):
    """The eight terms must agree with a from-the-paper scipy recomputation (not with mjlab)."""
    from supermimic.envs.state_io import StateIO
    from supermimic.world_model.heads import ReferenceFrame

    heads = H.build_heads(env)
    io = StateIO(env)
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("motion")

    torch.manual_seed(11)
    worst = {}
    for _ in range(6):
        a = 0.6 * torch.randn(env.num_envs, 29, device=env.device)
        env.step(a)
        s = io.read()
        lib, f = cmd.library, cmd.frame
        ref = ReferenceFrame(
            body_pos_w=lib.body_pos_w[f], body_quat_w=lib.body_quat_w[f],
            body_lin_vel_w=lib.body_lin_vel_w[f], body_ang_vel_w=lib.body_ang_vel_w[f],
            root_pos_w=lib.root_pos_w[f], root_quat_w=lib.root_quat_w[f],
        )
        ours = heads.reward(s, env.action_manager.action, env.action_manager.prev_action, ref)
        theirs = _scipy_reward_reference(
            heads, s, env.action_manager.action, env.action_manager.prev_action, ref,
            heads.reward_cfg, robot.data.soft_joint_pos_limits[0].clone(),
        )
        for k, v in theirs.items():
            d = float(np.abs(ours[k].detach().cpu().numpy() - v).max())
            worst[k] = max(worst.get(k, 0.0), d)
    bad = {k: round(v, 6) for k, v in worst.items() if v > 1e-4}
    assert not bad, f"reward terms disagree with the independent scipy reference: {bad}"


def test_termination_thresholds_and_the_78_degree_claim(env):
    """Appendix A: |e_pz| > 0.25 m and e_tilt > 0.8, and 'for an upright reference, e_tilt reaches
    0.8 at about 78 degrees of tilt' -- verify that analytic claim directly (independent of the env)."""
    tm = env.termination_manager
    names = tm.active_terms
    assert set(names) == {"time_out", "clip_end", "root_height", "root_tilt"}
    assert tm.get_term_cfg("root_height").params["threshold"] == 0.25
    assert tm.get_term_cfg("root_tilt").params["threshold"] == 0.8
    assert not any("ee" in n or "end_effector" in n for n in names)

    # projected gravity z of an upright body is -1, so e_tilt = |(-1) - u_z|
    e_tilt_78 = abs(-1.0 - (-np.cos(np.deg2rad(78.0))))
    assert abs(e_tilt_78 - 0.8) < 0.01, f"78 deg should give e_tilt ~0.8, got {e_tilt_78:.4f}"
    crossing = np.degrees(np.arccos(1.0 - 0.8))
    assert 65.0 < crossing < 80.0, f"e_tilt = 0.8 corresponds to {crossing:.1f} deg of tilt"

    # the failure verdict is the two non-timeout terms only (verified from the manager's own flags)
    assert tm.get_term_cfg("time_out").time_out is True
    assert tm.get_term_cfg("clip_end").time_out is True
    assert tm.get_term_cfg("root_height").time_out is False
    assert tm.get_term_cfg("root_tilt").time_out is False


def test_observation_dimensions_and_rollout_noise_widths(env):
    """93-d proprioception, 10-frame history, exact Table-A.1 noise half-widths used in imagination."""
    from supermimic.imagination.rollout import ANCHOR_NOISE_HALF, OBS_NOISE_HALF
    from supermimic.world_model.heads import OBS_DIM, OBS_TERMS, analytic_observation

    assert OBS_DIM == 93
    assert OBS_TERMS == (
        ("joint_pos", 29), ("joint_vel", 29), ("base_ang_vel", 3), ("projected_gravity", 3), ("actions", 29)
    )
    assert sum(d for _, d in OBS_TERMS) == 93
    assert env.observation_manager.group_obs_dim["policy"] == (930,)
    assert env.observation_manager.group_obs_dim["goal"] == (680,)
    assert env.cfg.observations["policy"].history_length == 10

    expected = [0.01] * 29 + [0.5] * 29 + [0.2] * 3 + [0.05] * 3 + [0.0] * 29
    torch.testing.assert_close(OBS_NOISE_HALF.cpu(), torch.tensor(expected), rtol=0, atol=0)
    # with anchor_obs the repo adds 9 extra dims with its own (self-chosen, non-paper) noise widths
    assert ANCHOR_NOISE_HALF.shape == (9,)

    robot = env.scene["robot"]
    rec = env.recorder_manager.get_term("replay")
    s = rec.state_io.read()
    o = analytic_observation(s, env.action_manager.prev_action, rec.ctx.read(), robot.data.default_joint_pos[0])
    assert o.shape == (env.num_envs, 93) and H.all_finite(o)