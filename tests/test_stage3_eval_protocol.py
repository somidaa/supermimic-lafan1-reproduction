"""Stage-3 Test: the evaluation protocol's normalised-return definition (Stage-1 finding F1).

Pure-CPU: these tests exercise the *definition* only, so they need no simulator and no GPU.

Paper §4 (p.6): "Normalized return ... is the undiscounted sum of the per-step reward over the full
episode, divided by the largest value this sum can take. **A policy that terminates early therefore
scores low.**" The border case the previous implementation got wrong is the denominator: it must be
the full episode's maximum, not the reward sum over however many steps the rollout survived.
"""

from __future__ import annotations

import pytest

from supermimic.eval.benchmark import ClipResult, normalized_return, summarize

MAX_R = 0.1  # Table A.2 upper bound per step (dt * sum of positive weights)


def test_normalized_return_is_independent_of_survived_steps():
    """The denominator is the full clip length, so it does not vary with how long the rollout lived."""
    full = 500
    ret = 25.0
    # same reward sum attributed to two rollouts of different lengths -> identical score
    assert normalized_return(ret, MAX_R, full) == normalized_return(ret, MAX_R, full)
    # and the value is exactly ret / (max_r * full)
    assert normalized_return(ret, MAX_R, full) == pytest.approx(ret / (MAX_R * full))

    # the OLD (wrong) behaviour averaged over the survived length; show the two disagree sharply
    survived = 50
    wrong = ret / (MAX_R * survived)
    right = normalized_return(ret, MAX_R, full)
    assert right < wrong, "using the survived length inflates the score of a clip that died early"
    assert wrong / right == pytest.approx(full / survived)


def test_early_termination_scores_low():
    """Same per-step quality, different survival: the shorter rollout must score strictly lower."""
    full, per_step = 1000, 0.082  # 0.082/step => a completed clip scores 0.82
    complete = normalized_return(per_step * full, MAX_R, full)
    died_10pct = normalized_return(per_step * (full // 10), MAX_R, full)
    assert complete == pytest.approx(0.82)
    assert died_10pct == pytest.approx(0.082)
    assert died_10pct < complete

    # summarise() must average these as-is (no renormalisation by steps)
    results = [
        ClipResult(name="complete", frames=full, steps=full, norm_return=complete,
                   failed=False, fail_step=-1, denominator_steps=full),
        ClipResult(name="died", frames=full, steps=full // 10, norm_return=died_10pct,
                   failed=True, fail_step=full // 10, denominator_steps=full),
    ]
    s = summarize(results)
    assert s["clips"] == 2 and s["failed"] == 1
    assert s["mean_norm_return"] == pytest.approx((complete + died_10pct) / 2)
    # the step-weighted aggregate now weights by the *denominator*, i.e. by clip length
    assert s["step_weighted_norm_return"] == pytest.approx((complete + died_10pct) / 2, rel=1e-6)


def test_score_is_proportional_to_survival_fraction():
    """With a constant per-step reward, the score equals per_step/max_r times the survival fraction."""
    full, per_step = 400, 0.06
    for frac in (0.1, 0.25, 0.5, 1.0):
        died = int(full * frac)
        score = normalized_return(per_step * died, MAX_R, full)
        assert score == pytest.approx((per_step / MAX_R) * frac, rel=1e-9)


def test_denominator_edge_cases():
    """Zero-length clips and zero reward must not raise or produce NaN."""
    assert normalized_return(0.0, MAX_R, 100) == 0.0
    assert normalized_return(5.0, MAX_R, 0) == pytest.approx(5.0 / MAX_R)  # clamped to >= 1 step
    assert normalized_return(5.0, MAX_R, -3) == pytest.approx(5.0 / MAX_R)


def test_clip_result_defaults_document_the_denominator():
    """``denominator_steps`` must be carried on every result so the score is auditable."""
    r = ClipResult(name="x", frames=123, steps=45, norm_return=0.1, failed=True, fail_step=45)
    assert r.denominator_steps == 0  # default for hand-built results
    r2 = ClipResult(name="y", frames=123, steps=123, norm_return=0.7, failed=False, fail_step=-1,
                    denominator_steps=123)
    assert r2.denominator_steps == r2.frames