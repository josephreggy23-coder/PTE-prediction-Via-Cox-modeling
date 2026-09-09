"""Tests for the Brier score calibration module.

Covers Kaplan-Meier estimation, IPCW Brier score, reference-model
comparison, and the Brier skill score.
"""

import numpy as np
import pytest

from pte.calibration import (
    BrierResult,
    brier_score_at_time,
    brier_score_curve,
    kaplan_meier,
)


# ------------------------------------------------------------------
# Kaplan-Meier
# ------------------------------------------------------------------
class TestKaplanMeier:
    def test_no_censoring(self) -> None:
        """With all events observed, KM is the empirical survival function."""
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event = np.ones(5)
        km_t, km_s = kaplan_meier(time, event)

        # At each event, survival drops by 1/n_at_risk.
        expected_surv = np.array([4 / 5, 3 / 5, 2 / 5, 1 / 5, 0.0])
        np.testing.assert_allclose(km_t, [1, 2, 3, 4, 5])
        np.testing.assert_allclose(km_s, expected_surv, atol=1e-10)

    def test_with_censoring(self) -> None:
        """Censored observations reduce the risk set without producing a drop."""
        time = np.array([1.0, 2.0, 3.0, 4.0])
        event = np.array([1, 0, 1, 1])
        km_t, km_s = kaplan_meier(time, event)

        # t=1: 4 at risk, 1 event -> S = 3/4
        # t=2: censored, no KM event
        # t=3: 2 at risk, 1 event -> S = 3/4 * 1/2 = 3/8
        # t=4: 1 at risk, 1 event -> S = 0
        np.testing.assert_allclose(km_t, [1, 3, 4])
        np.testing.assert_allclose(km_s, [3 / 4, 3 / 8, 0.0], atol=1e-10)

    def test_all_censored_returns_empty(self) -> None:
        """If no events occur, KM has no event times."""
        time = np.array([1.0, 2.0, 3.0])
        event = np.zeros(3)
        km_t, km_s = kaplan_meier(time, event)
        assert len(km_t) == 0


# ------------------------------------------------------------------
# Brier score at a single time
# ------------------------------------------------------------------
class TestBrierScoreAtTime:
    def test_perfect_predictions_give_zero(self) -> None:
        """If S_hat matches reality exactly, Brier score is zero."""
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event = np.ones(5)
        eval_t = 3.5
        # Subjects 1,2,3 had events before 3.5 -> y(t)=0 -> S_hat should be 0
        # Subjects 4,5 still at risk -> y(t)=1 -> S_hat should be 1
        s_pred = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        bs = brier_score_at_time(time, event, s_pred, eval_t)
        np.testing.assert_allclose(bs, 0.0, atol=1e-6)

    def test_worst_predictions_give_high_score(self) -> None:
        """Inverting predictions should give a high Brier score."""
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event = np.ones(5)
        eval_t = 3.5
        # Predictions are exactly wrong.
        s_pred = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        bs = brier_score_at_time(time, event, s_pred, eval_t)
        assert bs > 0.5

    def test_score_is_between_zero_and_one(self) -> None:
        """Brier score should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        time = rng.exponential(2.0, size=50)
        event = rng.binomial(1, 0.7, size=50).astype(float)
        s_pred = rng.uniform(0, 1, size=50)
        eval_t = float(np.median(time))
        bs = brier_score_at_time(time, event, s_pred, eval_t)
        if np.isfinite(bs):
            assert 0.0 <= bs <= 1.0


# ------------------------------------------------------------------
# Brier score curve and skill score
# ------------------------------------------------------------------
class TestBrierScoreCurve:
    def test_skill_score_positive_for_good_model(self) -> None:
        """A model that separates well should beat the marginal KM."""
        rng = np.random.default_rng(123)
        n = 100
        # High-risk and low-risk groups.
        risk = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        time = np.where(
            risk == 1,
            rng.exponential(1.0, size=n),
            rng.exponential(5.0, size=n),
        )
        event = np.ones(n)  # no censoring for a clean test

        # Predict S(t) = exp(-hazard * t) where hazard depends on risk.
        def survival_fn(t: float) -> np.ndarray:
            hazard = np.where(risk == 1, 1.0, 0.2)
            return np.exp(-hazard * t)

        result = brier_score_curve(time, event, survival_fn, n_times=20)
        assert isinstance(result, BrierResult)
        assert result.brier_skill_score > 0.0

    def test_marginal_model_has_zero_skill(self) -> None:
        """Using KM marginal as the prediction should give BSS near 0."""
        rng = np.random.default_rng(77)
        n = 80
        time = rng.exponential(2.0, size=n)
        event = np.ones(n)

        km_t, km_s = kaplan_meier(time, event)

        def marginal_fn(t: float) -> np.ndarray:
            idx = np.searchsorted(km_t, t, side="right") - 1
            s = km_s[idx] if idx >= 0 else 1.0
            return np.full(n, s)

        result = brier_score_curve(time, event, marginal_fn, n_times=20)
        # BSS should be very close to 0 (model = reference).
        np.testing.assert_allclose(result.brier_skill_score, 0.0, atol=0.05)

    def test_too_few_events_returns_empty(self) -> None:
        """With fewer than 2 events, the curve should be empty."""
        time = np.array([1.0, 2.0, 3.0])
        event = np.array([1, 0, 0])
        result = brier_score_curve(
            time, event, lambda t: np.full(3, 0.5), n_times=10
        )
        assert len(result.eval_times) == 0
