"""Tests for proportional hazards diagnostics.

Covers Schoenfeld residuals, scaled Schoenfeld residuals, and the
correlation test for the PH assumption using synthetic survival data
with known properties.
"""

import numpy as np
import pytest
from scipy import stats as sp_stats

from pte.ph_diagnostics import (
    CovariateTestResult,
    PHTestResult,
    ph_test,
    scaled_schoenfeld_residuals,
    schoenfeld_residuals,
)


# ------------------------------------------------------------------
# Helpers for generating synthetic survival data
# ------------------------------------------------------------------
def _generate_ph_data(
    n: int = 1000,
    beta: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate data satisfying PH (exponential baseline).

    T_i = -log(U_i) / exp(x_i * beta), which gives an exponential
    distribution with rate exp(x_i * beta), satisfying PH exactly.

    Returns time, event, covariates (n, 1), coef (1,).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    u = rng.uniform(0, 1, n)
    time = -np.log(u) / np.exp(x * beta)
    event = np.ones(n)
    return time, event, x.reshape(-1, 1), np.array([beta])


def _generate_ph_violation_data(
    n: int = 500,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate data that violates PH.

    Group 0 (x=0): T ~ Exp(rate=1)       -- constant hazard h(t)=1
    Group 1 (x=1): T ~ Weibull(shape=3)  -- increasing hazard h(t)=3t^2

    The hazard ratio h_1(t)/h_0(t) = 3t^2 changes with time, so PH
    is violated.  The average log-hazard ratio is used as beta.

    Returns time, event, covariates (n, 1), coef (1,).
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    x = np.concatenate([np.zeros(half), np.ones(n - half)])

    time_0 = rng.exponential(1.0, size=half)
    time_1 = rng.weibull(3.0, size=n - half)
    time = np.concatenate([time_0, time_1])
    event = np.ones(n)

    # Use a rough average log-HR as the coefficient.
    mean_rate_0 = 1.0 / time_0.mean()
    mean_rate_1 = 1.0 / time_1.mean()
    beta_approx = np.log(mean_rate_1 / mean_rate_0)
    return time, event, x.reshape(-1, 1), np.array([beta_approx])


# ------------------------------------------------------------------
# Schoenfeld residuals: basic properties
# ------------------------------------------------------------------
class TestSchoenfeldResiduals:
    def test_zero_mean_under_ph(self) -> None:
        """Under PH at the true beta, residuals have mean near zero."""
        time, event, X, coef = _generate_ph_data(n=1000, seed=42)
        event_times, residuals = schoenfeld_residuals(time, event, X, coef)

        # With n=1000 the sample mean should be close to the
        # theoretical value of zero.
        assert abs(residuals.mean()) < 0.1

    def test_residual_count_equals_events(self) -> None:
        """One residual per event per covariate."""
        n = 200
        rng = np.random.default_rng(7)
        time = rng.exponential(1.0, n)
        event = rng.binomial(1, 0.7, n)
        x = rng.standard_normal(n)
        coef = np.array([0.3])

        event_times, residuals = schoenfeld_residuals(
            time, event, x.reshape(-1, 1), coef
        )
        n_events = int(event.sum())
        assert len(event_times) == n_events
        assert residuals.shape == (n_events, 1)

    def test_event_times_sorted(self) -> None:
        """Returned event times should be in ascending order."""
        time, event, X, coef = _generate_ph_data(n=100, seed=0)
        event_times, _ = schoenfeld_residuals(time, event, X, coef)
        assert np.all(np.diff(event_times) >= 0)

    def test_two_covariates(self) -> None:
        """Residuals have shape (d, p) for p > 1."""
        rng = np.random.default_rng(11)
        n = 200
        X = rng.standard_normal((n, 2))
        coef = np.array([0.5, -0.3])
        u = rng.uniform(0, 1, n)
        time = -np.log(u) / np.exp(X @ coef)
        event = np.ones(n)

        event_times, residuals = schoenfeld_residuals(time, event, X, coef)
        assert residuals.shape == (n, 2)


# ------------------------------------------------------------------
# Scaled Schoenfeld residuals
# ------------------------------------------------------------------
class TestScaledSchoenfeldResiduals:
    def test_mean_near_beta_under_ph(self) -> None:
        """Scaled residuals should average approximately beta_hat."""
        time, event, X, coef = _generate_ph_data(n=1000, seed=99)
        _, _, scaled = scaled_schoenfeld_residuals(time, event, X, coef)

        # The mean of scaled residuals estimates the constant beta(t).
        assert abs(scaled.mean() - coef[0]) < 0.3

    def test_returns_three_arrays(self) -> None:
        """Function returns (event_times, residuals, scaled)."""
        time, event, X, coef = _generate_ph_data(n=50, seed=1)
        result = scaled_schoenfeld_residuals(time, event, X, coef)
        assert len(result) == 3
        et, raw, sc = result
        assert et.shape == raw.shape[:1]
        assert raw.shape == sc.shape


# ------------------------------------------------------------------
# PH test: proportional hazards satisfied
# ------------------------------------------------------------------
class TestPHTestUnderPH:
    def test_non_significant_under_ph(self) -> None:
        """Under PH the test should usually not reject (p > 0.05).

        With the true beta and a large sample the p-value should be
        well above 0.05 more often than not.  We run with a fixed seed
        that produces a clearly non-significant result.
        """
        time, event, X, coef = _generate_ph_data(n=500, seed=77)
        result = ph_test(time, event, X, coef, time_transform="rank")

        assert result.covariate_results[0].p_value > 0.05
        assert result.global_p_value > 0.05

    def test_result_structure(self) -> None:
        """PHTestResult has the expected fields and shapes."""
        time, event, X, coef = _generate_ph_data(n=100, seed=3)
        result = ph_test(time, event, X, coef)

        assert isinstance(result, PHTestResult)
        assert len(result.covariate_results) == 1
        assert isinstance(result.covariate_results[0], CovariateTestResult)
        assert result.schoenfeld_residuals.shape == (100, 1)
        assert result.scaled_schoenfeld_residuals.shape == (100, 1)
        assert result.time_transform == "rank"

    def test_time_transforms(self) -> None:
        """All three time transforms should work."""
        time, event, X, coef = _generate_ph_data(n=100, seed=5)
        for tf in ("identity", "log", "rank"):
            result = ph_test(time, event, X, coef, time_transform=tf)
            assert result.time_transform == tf
            assert np.isfinite(result.covariate_results[0].p_value)


# ------------------------------------------------------------------
# PH test: known violation
# ------------------------------------------------------------------
class TestPHTestViolation:
    def test_violation_detected(self) -> None:
        """A time-varying hazard ratio should produce a significant test."""
        time, event, X, coef = _generate_ph_violation_data(n=500, seed=123)
        result = ph_test(time, event, X, coef, time_transform="rank")

        assert result.covariate_results[0].p_value < 0.05
        assert result.global_p_value < 0.05

    def test_violation_detected_log_transform(self) -> None:
        """Log transform should also detect the violation."""
        time, event, X, coef = _generate_ph_violation_data(n=500, seed=456)
        result = ph_test(time, event, X, coef, time_transform="log")

        assert result.covariate_results[0].p_value < 0.05

    def test_violation_chi_sq_positive(self) -> None:
        """The chi-squared statistic should be positive."""
        time, event, X, coef = _generate_ph_violation_data(n=300, seed=789)
        result = ph_test(time, event, X, coef)

        assert result.covariate_results[0].chi_sq > 0
        assert result.global_chi_sq > 0


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------
class TestPHTestEdgeCases:
    def test_single_covariate(self) -> None:
        """Test works correctly with a single covariate."""
        rng = np.random.default_rng(7)
        n = 100
        x = rng.standard_normal(n)
        coef = np.array([0.3])
        u = rng.uniform(0, 1, n)
        time = -np.log(u) / np.exp(x * coef[0])
        event = np.ones(n)

        result = ph_test(time, event, x.reshape(-1, 1), coef)
        assert len(result.covariate_results) == 1
        assert result.global_df == 1
        assert result.covariate_results[0].df == 1
        assert result.covariate_results[0].name == "x0"

    def test_all_events_at_same_time(self) -> None:
        """When all events are simultaneous, correlation is undefined."""
        rng = np.random.default_rng(0)
        n = 50
        time = np.full(n, 5.0)
        event = np.ones(n)
        x = rng.standard_normal(n)
        coef = np.array([0.0])

        result = ph_test(time, event, x.reshape(-1, 1), coef)

        # Time has zero variance, so correlation and p-value are NaN.
        assert np.isnan(result.covariate_results[0].correlation)
        assert np.isnan(result.covariate_results[0].p_value)
        assert np.isnan(result.global_p_value)

    def test_custom_covariate_names(self) -> None:
        """User-provided covariate names are used in the result."""
        time, event, X, coef = _generate_ph_data(n=50, seed=2)
        result = ph_test(
            time, event, X, coef, covariate_names=["age"]
        )
        assert result.covariate_results[0].name == "age"

    def test_covariate_names_length_mismatch_raises(self) -> None:
        """Wrong number of covariate names should raise."""
        time, event, X, coef = _generate_ph_data(n=50, seed=2)
        with pytest.raises(ValueError, match="covariate_names"):
            ph_test(time, event, X, coef, covariate_names=["a", "b"])

    def test_invalid_time_transform_raises(self) -> None:
        """Unknown transform name should raise."""
        time, event, X, coef = _generate_ph_data(n=50, seed=2)
        with pytest.raises(ValueError, match="Unknown time_transform"):
            ph_test(time, event, X, coef, time_transform="sqrt")

    def test_dimension_mismatch_raises(self) -> None:
        """Mismatched covariate and coefficient dimensions raise."""
        time = np.array([1.0, 2.0, 3.0])
        event = np.ones(3)
        X = np.ones((3, 2))
        coef = np.array([0.5])  # 1 coef for 2 covariates
        with pytest.raises(ValueError, match="columns"):
            schoenfeld_residuals(time, event, X, coef)

    def test_with_censoring(self) -> None:
        """Residuals computed only at uncensored event times."""
        rng = np.random.default_rng(33)
        n = 200
        x = rng.standard_normal(n)
        coef = np.array([0.5])
        u = rng.uniform(0, 1, n)
        time = -np.log(u) / np.exp(x * coef[0])
        # Censor 30% of observations.
        event = (rng.uniform(size=n) > 0.3).astype(float)
        n_events = int(event.sum())

        event_times, residuals = schoenfeld_residuals(
            time, event, x.reshape(-1, 1), coef
        )
        assert len(event_times) == n_events
        assert residuals.shape[0] == n_events

    def test_multiple_covariates_global_test(self) -> None:
        """Global test aggregates across covariates."""
        rng = np.random.default_rng(55)
        n = 300
        p = 3
        X = rng.standard_normal((n, p))
        coef = np.array([0.3, -0.2, 0.1])
        u = rng.uniform(0, 1, n)
        time = -np.log(u) / np.exp(X @ coef)
        event = np.ones(n)

        result = ph_test(time, event, X, coef)
        assert len(result.covariate_results) == 3
        assert result.global_df == 3

        # Global chi-sq should be sum of per-covariate chi-sq.
        expected = sum(r.chi_sq for r in result.covariate_results)
        assert result.global_chi_sq == pytest.approx(expected)
