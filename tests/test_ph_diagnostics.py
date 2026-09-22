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
    _apply_time_transform,
    _validate_inputs,
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
    censor_rate: float = 0.0,
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
    if censor_rate > 0:
        event = (rng.uniform(size=n) > censor_rate).astype(float)
    else:
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
# 1. Input validation (_validate_inputs)
# ------------------------------------------------------------------
class TestValidateInputs:
    """Dimension mismatches and shape coercion."""

    def test_time_covariates_mismatch_raises(self) -> None:
        """Covariates rows != time length raises ValueError."""
        time = np.array([1.0, 2.0, 3.0])
        event = np.ones(3)
        X = np.ones((4, 1))  # 4 rows vs 3 times
        coef = np.array([0.5])
        with pytest.raises(ValueError, match="covariates has 4 rows but time has 3"):
            _validate_inputs(time, event, X, coef)

    def test_event_covariates_mismatch_raises(self) -> None:
        """Covariates rows != event length raises ValueError."""
        time = np.array([1.0, 2.0, 3.0])
        event = np.ones(4)  # 4 events vs 3 rows
        X = np.ones((3, 1))
        coef = np.array([0.5])
        with pytest.raises(ValueError, match="covariates has 3 rows but event has 4"):
            _validate_inputs(time, event, X, coef)

    def test_coef_covariates_mismatch_raises(self) -> None:
        """Covariates columns != coef length raises ValueError."""
        time = np.array([1.0, 2.0, 3.0])
        event = np.ones(3)
        X = np.ones((3, 2))
        coef = np.array([0.5])  # 1 coef for 2 columns
        with pytest.raises(ValueError, match="columns"):
            _validate_inputs(time, event, X, coef)

    def test_1d_covariates_reshaped(self) -> None:
        """A 1-D covariate vector is reshaped to (n, 1)."""
        time = np.array([1.0, 2.0, 3.0])
        event = np.ones(3)
        x = np.array([0.1, 0.2, 0.3])  # 1-D
        coef = np.array([0.5])
        t_out, e_out, X_out, c_out = _validate_inputs(time, event, x, coef)
        assert X_out.ndim == 2
        assert X_out.shape == (3, 1)

    def test_coerces_to_float(self) -> None:
        """Integer inputs are coerced to float arrays."""
        time = np.array([1, 2, 3])
        event = np.array([1, 1, 0])
        X = np.array([[1], [2], [3]])
        coef = np.array([1])
        t_out, e_out, X_out, c_out = _validate_inputs(time, event, X, coef)
        assert t_out.dtype == float
        assert e_out.dtype == float
        assert X_out.dtype == float
        assert c_out.dtype == float


# ------------------------------------------------------------------
# 2. Time transforms (_apply_time_transform)
# ------------------------------------------------------------------
class TestTimeTransforms:
    """Direct verification that transforms produce correct output."""

    def test_identity(self) -> None:
        t = np.array([3.0, 1.0, 2.0])
        result = _apply_time_transform(t, "identity")
        np.testing.assert_array_equal(result, t)
        # Should be a copy, not the same object.
        assert result is not t

    def test_log(self) -> None:
        t = np.array([1.0, np.e, np.e**2])
        result = _apply_time_transform(t, "log")
        np.testing.assert_allclose(result, [0.0, 1.0, 2.0])

    def test_rank(self) -> None:
        t = np.array([10.0, 30.0, 20.0])
        result = _apply_time_transform(t, "rank")
        np.testing.assert_array_equal(result, [1.0, 3.0, 2.0])

    def test_rank_with_ties(self) -> None:
        t = np.array([5.0, 5.0, 10.0])
        result = _apply_time_transform(t, "rank")
        # scipy.stats.rankdata uses average rank for ties by default.
        np.testing.assert_array_equal(result, [1.5, 1.5, 3.0])

    def test_unknown_transform_raises(self) -> None:
        t = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Unknown time_transform"):
            _apply_time_transform(t, "sqrt")


# ------------------------------------------------------------------
# 3. Schoenfeld residuals on known examples
# ------------------------------------------------------------------
class TestSchoenfeldResiduals:
    def test_shape_equals_events_by_covariates(self) -> None:
        """Residual array shape is (d, p) where d = number of events."""
        rng = np.random.default_rng(7)
        n = 200
        time = rng.exponential(1.0, n)
        event = rng.binomial(1, 0.7, n).astype(float)
        x = rng.standard_normal(n)
        coef = np.array([0.3])

        event_times, residuals = schoenfeld_residuals(
            time, event, x.reshape(-1, 1), coef
        )
        n_events = int(event.sum())
        assert len(event_times) == n_events
        assert residuals.shape == (n_events, 1)

    def test_residuals_only_at_event_times(self) -> None:
        """Residuals exist only at uncensored event times, not censored."""
        rng = np.random.default_rng(33)
        n = 200
        x = rng.standard_normal(n)
        coef = np.array([0.5])
        u = rng.uniform(0, 1, n)
        time = -np.log(u) / np.exp(x * coef[0])
        # Censor 30% of observations.
        event = (rng.uniform(size=n) > 0.3).astype(float)

        event_times, residuals = schoenfeld_residuals(
            time, event, x.reshape(-1, 1), coef
        )
        n_events = int(event.sum())
        assert len(event_times) == n_events
        assert residuals.shape[0] == n_events

        # Every returned event time must correspond to an uncensored subject.
        uncensored_times = set(time[event == 1])
        for t in event_times:
            assert t in uncensored_times

    def test_event_times_sorted(self) -> None:
        """Returned event times are in ascending order."""
        time, event, X, coef = _generate_ph_data(n=100, seed=0)
        event_times, _ = schoenfeld_residuals(time, event, X, coef)
        assert np.all(np.diff(event_times) >= 0)

    def test_zero_mean_under_ph(self) -> None:
        """Under PH at the true beta, mean residual is near zero."""
        time, event, X, coef = _generate_ph_data(n=1000, seed=42)
        _, residuals = schoenfeld_residuals(time, event, X, coef)

        # With 1000 events the sample mean should be close to zero.
        assert abs(residuals.mean()) < 0.1

    def test_two_covariates_shape(self) -> None:
        """Residuals have shape (d, p) for p = 2."""
        rng = np.random.default_rng(11)
        n = 200
        X = rng.standard_normal((n, 2))
        coef = np.array([0.5, -0.3])
        u = rng.uniform(0, 1, n)
        time = -np.log(u) / np.exp(X @ coef)
        event = np.ones(n)

        event_times, residuals = schoenfeld_residuals(time, event, X, coef)
        assert residuals.shape == (n, 2)

    def test_all_events_no_censoring(self) -> None:
        """When all subjects have events, d == n."""
        time, event, X, coef = _generate_ph_data(n=50, seed=10)
        assert np.all(event == 1)
        event_times, residuals = schoenfeld_residuals(time, event, X, coef)
        assert len(event_times) == 50
        assert residuals.shape == (50, 1)

    def test_small_hand_computed_example(self) -> None:
        """Verify residuals on a tiny dataset with known structure.

        Three subjects, one binary covariate, beta=0 (all weights equal).
        Event at t=1 for subject 0 (x=1), risk set = {0,1,2}.
        Weighted mean of x = (1+0+1)/3 = 2/3.
        Residual = x_0 - x_bar = 1 - 2/3 = 1/3.

        Event at t=2 for subject 1 (x=0), risk set = {1,2}.
        Weighted mean of x = (0+1)/2 = 1/2.
        Residual = 0 - 1/2 = -1/2.
        """
        time = np.array([1.0, 2.0, 3.0])
        event = np.array([1.0, 1.0, 0.0])  # subject 2 censored at t=3
        X = np.array([[1.0], [0.0], [1.0]])
        coef = np.array([0.0])  # all weights equal to 1

        event_times, residuals = schoenfeld_residuals(time, event, X, coef)

        assert len(event_times) == 2
        np.testing.assert_allclose(event_times, [1.0, 2.0])
        np.testing.assert_allclose(residuals[0, 0], 1.0 / 3.0, atol=1e-12)
        np.testing.assert_allclose(residuals[1, 0], -0.5, atol=1e-12)


# ------------------------------------------------------------------
# 4. PH satisfied scenario
# ------------------------------------------------------------------
class TestPHTestUnderPH:
    def test_non_significant_with_200_events(self) -> None:
        """Under PH with 200+ events, test should not reject (p > 0.05)."""
        time, event, X, coef = _generate_ph_data(n=500, seed=77)
        result = ph_test(time, event, X, coef, time_transform="rank")

        assert result.covariate_results[0].p_value > 0.05
        assert result.global_p_value > 0.05

    def test_non_significant_identity_transform(self) -> None:
        """Under PH, identity transform should also be non-significant."""
        time, event, X, coef = _generate_ph_data(n=500, seed=77)
        result = ph_test(time, event, X, coef, time_transform="identity")
        assert result.covariate_results[0].p_value > 0.05

    def test_non_significant_log_transform(self) -> None:
        """Under PH, log transform should also be non-significant."""
        time, event, X, coef = _generate_ph_data(n=500, seed=77)
        result = ph_test(time, event, X, coef, time_transform="log")
        assert result.covariate_results[0].p_value > 0.05

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

    def test_correlation_near_zero_under_ph(self) -> None:
        """Under PH, correlation between residuals and time is near zero."""
        time, event, X, coef = _generate_ph_data(n=500, seed=88)
        result = ph_test(time, event, X, coef, time_transform="rank")
        # Absolute correlation should be small under PH.
        assert abs(result.covariate_results[0].correlation) < 0.15


# ------------------------------------------------------------------
# 5. PH violation scenario
# ------------------------------------------------------------------
class TestPHTestViolation:
    def test_violation_detected_rank(self) -> None:
        """A time-varying hazard ratio produces a significant test (rank)."""
        time, event, X, coef = _generate_ph_violation_data(n=500, seed=123)
        result = ph_test(time, event, X, coef, time_transform="rank")

        assert result.covariate_results[0].p_value < 0.05
        assert result.global_p_value < 0.05

    def test_violation_detected_log(self) -> None:
        """Log transform should also detect the violation."""
        time, event, X, coef = _generate_ph_violation_data(n=500, seed=456)
        result = ph_test(time, event, X, coef, time_transform="log")

        assert result.covariate_results[0].p_value < 0.05

    def test_violation_chi_sq_positive(self) -> None:
        """The chi-squared statistic should be positive under violation."""
        time, event, X, coef = _generate_ph_violation_data(n=300, seed=789)
        result = ph_test(time, event, X, coef)

        assert result.covariate_results[0].chi_sq > 0
        assert result.global_chi_sq > 0

    def test_violation_pvalue_smaller_than_ph(self) -> None:
        """Violation scenario should yield a smaller p-value than PH scenario."""
        _, _, X_ph, coef_ph = _generate_ph_data(n=500, seed=77)
        time_ph, event_ph = _generate_ph_data(n=500, seed=77)[:2]
        result_ph = ph_test(time_ph, event_ph, X_ph, coef_ph)

        time_v, event_v, X_v, coef_v = _generate_ph_violation_data(n=500, seed=123)
        result_v = ph_test(time_v, event_v, X_v, coef_v)

        assert result_v.global_p_value < result_ph.global_p_value

    def test_chi_sq_equals_rho_squared_times_d(self) -> None:
        """Verify chi2 = rho^2 * d for each covariate."""
        time, event, X, coef = _generate_ph_violation_data(n=400, seed=321)
        result = ph_test(time, event, X, coef)

        d = len(result.event_times)
        for cr in result.covariate_results:
            if not np.isnan(cr.correlation):
                expected_chi_sq = cr.correlation ** 2 * d
                assert cr.chi_sq == pytest.approx(expected_chi_sq, rel=1e-10)


# ------------------------------------------------------------------
# 6. Scaled residuals: verify the Grambsch-Therneau formula
# ------------------------------------------------------------------
class TestScaledSchoenfeldResiduals:
    def test_scaling_formula(self) -> None:
        """Verify r*_j = beta_j + d * r_j / I_jj by recomputing from raw.

        We call both schoenfeld_residuals and scaled_schoenfeld_residuals
        with the same inputs, then use _compute_residuals_and_info to get
        info_diag and manually apply the formula.
        """
        from pte.ph_diagnostics import _compute_residuals_and_info

        time, event, X, coef = _generate_ph_data(n=200, seed=50)
        event_times, residuals, info_diag = _compute_residuals_and_info(
            time, event, X, coef
        )
        _, _, scaled = scaled_schoenfeld_residuals(time, event, X, coef)

        d = len(event_times)
        p = X.shape[1]
        for j in range(p):
            if info_diag[j] > 0:
                expected = coef[j] + d * residuals[:, j] / info_diag[j]
            else:
                expected = np.full(d, coef[j])
            np.testing.assert_allclose(scaled[:, j], expected, rtol=1e-12)

    def test_scaling_formula_multi_covariate(self) -> None:
        """Same formula verification with p=3 covariates."""
        from pte.ph_diagnostics import _compute_residuals_and_info

        rng = np.random.default_rng(60)
        n = 300
        p = 3
        X = rng.standard_normal((n, p))
        coef = np.array([0.3, -0.2, 0.1])
        u = rng.uniform(0, 1, n)
        time = -np.log(u) / np.exp(X @ coef)
        event = np.ones(n)

        event_times, residuals, info_diag = _compute_residuals_and_info(
            time, event, X, coef
        )
        _, _, scaled = scaled_schoenfeld_residuals(time, event, X, coef)

        d = len(event_times)
        for j in range(p):
            expected = coef[j] + d * residuals[:, j] / info_diag[j]
            np.testing.assert_allclose(scaled[:, j], expected, rtol=1e-12)

    def test_mean_near_beta_under_ph(self) -> None:
        """Under PH, scaled residuals average approximately beta_hat."""
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
# 7. Edge cases
# ------------------------------------------------------------------
class TestEdgeCases:
    def test_single_covariate_1d_input(self) -> None:
        """A 1-D covariate array (not reshaped) still works."""
        rng = np.random.default_rng(7)
        n = 100
        x = rng.standard_normal(n)  # 1-D, not (n, 1)
        coef = np.array([0.3])
        u = rng.uniform(0, 1, n)
        time = -np.log(u) / np.exp(x * coef[0])
        event = np.ones(n)

        result = ph_test(time, event, x, coef)
        assert len(result.covariate_results) == 1
        assert result.global_df == 1
        assert result.covariate_results[0].df == 1
        assert result.covariate_results[0].name == "x0"

    def test_all_events_no_censoring(self) -> None:
        """When event indicator is all 1, d == n."""
        time, event, X, coef = _generate_ph_data(n=100, seed=20)
        assert np.all(event == 1)
        result = ph_test(time, event, X, coef)
        assert len(result.event_times) == 100

    def test_custom_covariate_names(self) -> None:
        """User-provided covariate names appear in the result."""
        time, event, X, coef = _generate_ph_data(n=50, seed=2)
        result = ph_test(time, event, X, coef, covariate_names=["age"])
        assert result.covariate_results[0].name == "age"

    def test_covariate_names_length_mismatch_raises(self) -> None:
        """Wrong number of covariate names should raise."""
        time, event, X, coef = _generate_ph_data(n=50, seed=2)
        with pytest.raises(ValueError, match="covariate_names"):
            ph_test(time, event, X, coef, covariate_names=["a", "b"])

    def test_invalid_time_transform_raises(self) -> None:
        """Unknown transform name should raise ValueError."""
        time, event, X, coef = _generate_ph_data(n=50, seed=2)
        with pytest.raises(ValueError, match="Unknown time_transform"):
            ph_test(time, event, X, coef, time_transform="sqrt")

    def test_all_events_at_same_time(self) -> None:
        """When all events are simultaneous, correlation is undefined."""
        rng = np.random.default_rng(0)
        n = 50
        time = np.full(n, 5.0)
        event = np.ones(n)
        x = rng.standard_normal(n)
        coef = np.array([0.0])

        result = ph_test(time, event, x.reshape(-1, 1), coef)

        # Time has zero variance after rank, so correlation is NaN.
        assert np.isnan(result.covariate_results[0].correlation)
        assert np.isnan(result.covariate_results[0].p_value)
        assert np.isnan(result.global_p_value)

    def test_multiple_covariates_global_chi_sq_is_sum(self) -> None:
        """Global chi-sq equals sum of per-covariate chi-sq values."""
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

    def test_p_value_from_chi2_distribution(self) -> None:
        """Verify p-value = 1 - chi2.cdf(rho^2 * d, df=1)."""
        time, event, X, coef = _generate_ph_data(n=200, seed=44)
        result = ph_test(time, event, X, coef, time_transform="rank")

        cr = result.covariate_results[0]
        if not np.isnan(cr.chi_sq):
            expected_p = float(1.0 - sp_stats.chi2.cdf(cr.chi_sq, df=1))
            assert cr.p_value == pytest.approx(expected_p, rel=1e-10)

    def test_with_censoring_fewer_residuals(self) -> None:
        """With censoring, fewer residuals are returned than n."""
        time, event, X, coef = _generate_ph_data(
            n=200, seed=33, censor_rate=0.3
        )
        n_events = int(event.sum())
        assert n_events < 200  # sanity: some censoring occurred

        event_times, residuals = schoenfeld_residuals(time, event, X, coef)
        assert len(event_times) == n_events
        assert residuals.shape[0] == n_events

    def test_default_covariate_names(self) -> None:
        """When no names given, defaults to x0, x1, etc."""
        rng = np.random.default_rng(70)
        n = 100
        X = rng.standard_normal((n, 3))
        coef = np.array([0.1, 0.2, 0.3])
        u = rng.uniform(0, 1, n)
        time = -np.log(u) / np.exp(X @ coef)
        event = np.ones(n)

        result = ph_test(time, event, X, coef)
        names = [cr.name for cr in result.covariate_results]
        assert names == ["x0", "x1", "x2"]

    def test_dimension_mismatch_via_public_api(self) -> None:
        """Mismatched dimensions raise ValueError through public functions."""
        time = np.array([1.0, 2.0, 3.0])
        event = np.ones(3)
        X = np.ones((3, 2))
        coef = np.array([0.5])  # 1 coef for 2 covariates

        with pytest.raises(ValueError, match="columns"):
            schoenfeld_residuals(time, event, X, coef)

        with pytest.raises(ValueError, match="columns"):
            scaled_schoenfeld_residuals(time, event, X, coef)

        with pytest.raises(ValueError, match="columns"):
            ph_test(time, event, X, coef)
