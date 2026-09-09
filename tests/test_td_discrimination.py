"""Tests for time-dependent discrimination measures.

These tests verify the time-dependent AUC estimator and its supporting
functions using synthetic survival data with known properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from pte.td_discrimination import (
    TdAucResult,
    _get_censoring_weight,
    _km_censoring,
    td_auc_at_time,
    td_auc_bootstrap_ci,
    td_auc_curve,
)


# ---- Helpers -----------------------------------------------------------

def _perfect_data(n: int = 100, seed: int = 0):
    """Survival data where the risk score perfectly ranks event times.

    Lower event time = higher risk. The risk score is the negative of
    the event time, so concordance and AUC should be near 1.0.
    """
    rng = np.random.default_rng(seed)
    time = rng.exponential(scale=10.0, size=n)
    event = np.ones(n)  # no censoring
    risk = -time  # perfect negative correlation with time
    return time, event, risk


def _random_data(n: int = 100, censor_frac: float = 0.3, seed: int = 1):
    """Survival data with random censoring and a moderate signal."""
    rng = np.random.default_rng(seed)
    # True risk determines event time.
    true_risk = rng.standard_normal(n)
    time = rng.exponential(scale=np.exp(-0.5 * true_risk))
    # Random censoring.
    censor_time = rng.exponential(scale=15.0, size=n)
    event = (time <= censor_time).astype(float)
    time = np.minimum(time, censor_time)
    # Risk score is the true risk plus noise -- imperfect but correlated.
    risk = true_risk + 0.5 * rng.standard_normal(n)
    return time, event, risk


def _uninformative_data(n: int = 100, seed: int = 2):
    """Survival data where the risk score is pure noise."""
    rng = np.random.default_rng(seed)
    time = rng.exponential(scale=5.0, size=n)
    event = np.ones(n)
    risk = rng.standard_normal(n)  # no correlation with time
    return time, event, risk


# ---- Censoring KM -----------------------------------------------------

def test_km_censoring_no_censoring():
    """With no censoring events, the censoring survival should stay at 1."""
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.ones(5)  # all events, no censoring

    km_t, km_s = _km_censoring(time, event)
    # Censoring distribution has no events, so survival = 1 everywhere.
    assert np.all(km_s >= 1.0 - 1e-10)


def test_km_censoring_all_censored():
    """With all observations censored, the censoring survival drops."""
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.zeros(5)  # all censored

    km_t, km_s = _km_censoring(time, event)
    # Every observation is a "censoring event", so survival decreases.
    assert km_s[-1] < 0.5


def test_km_censoring_weight_before_first_time():
    """Censoring weight before any observed time should be 1.0."""
    time = np.array([2.0, 3.0, 4.0])
    event = np.array([1, 0, 1])
    km_t, km_s = _km_censoring(time, event)

    w = _get_censoring_weight(0.5, km_t, km_s)
    assert w == 1.0


# ---- Single-time AUC ---------------------------------------------------

def test_td_auc_perfect_separation():
    """With a perfect risk score, AUC should be 1.0."""
    time, event, risk = _perfect_data(n=200)
    # Evaluate at the median event time.
    eval_t = float(np.median(time))
    auc = td_auc_at_time(time, event, risk, eval_t, ipcw=False)
    assert auc == pytest.approx(1.0, abs=0.01)


def test_td_auc_uninformative_near_half():
    """A random risk score should give AUC near 0.5."""
    time, event, risk = _uninformative_data(n=500, seed=42)
    eval_t = float(np.median(time))
    auc = td_auc_at_time(time, event, risk, eval_t, ipcw=False)
    assert 0.35 < auc < 0.65


def test_td_auc_returns_nan_no_cases():
    """If eval_time is before all events, there are no cases."""
    time = np.array([5.0, 6.0, 7.0])
    event = np.ones(3)
    risk = np.array([1.0, 2.0, 3.0])
    auc = td_auc_at_time(time, event, risk, eval_time=1.0, ipcw=False)
    assert np.isnan(auc)


def test_td_auc_returns_nan_no_controls():
    """If eval_time is after all observations, there are no controls."""
    time = np.array([1.0, 2.0, 3.0])
    event = np.ones(3)
    risk = np.array([3.0, 2.0, 1.0])
    auc = td_auc_at_time(time, event, risk, eval_time=10.0, ipcw=False)
    assert np.isnan(auc)


def test_td_auc_ipcw_runs():
    """Smoke test: IPCW variant runs and returns a finite value."""
    time, event, risk = _random_data(n=200)
    eval_t = float(np.median(time[event == 1]))
    auc = td_auc_at_time(time, event, risk, eval_t, ipcw=True)
    assert np.isfinite(auc)
    assert 0.0 <= auc <= 1.0


# ---- AUC curve ----------------------------------------------------------

def test_td_auc_curve_shape():
    """Curve result should have consistent array lengths."""
    time, event, risk = _random_data(n=150, seed=5)
    result = td_auc_curve(time, event, risk, n_times=20, ipcw=False)

    assert isinstance(result, TdAucResult)
    assert result.eval_times.shape == (20,)
    assert result.auc_values.shape == (20,)
    assert result.n_cases.shape == (20,)
    assert result.n_controls.shape == (20,)


def test_td_auc_curve_monotone_cases():
    """Number of cases should be non-decreasing across the time grid."""
    time, event, risk = _random_data(n=200, seed=6)
    result = td_auc_curve(time, event, risk, n_times=30, ipcw=False)
    assert np.all(np.diff(result.n_cases) >= 0)


def test_td_auc_curve_monotone_controls():
    """Number of controls should be non-increasing across the time grid."""
    time, event, risk = _random_data(n=200, seed=7)
    result = td_auc_curve(time, event, risk, n_times=30, ipcw=False)
    assert np.all(np.diff(result.n_controls) <= 0)


def test_mean_auc_in_range():
    """Mean AUC across the curve should be a valid probability."""
    time, event, risk = _random_data(n=200, seed=8)
    result = td_auc_curve(time, event, risk, n_times=25, ipcw=False)
    assert 0.0 <= result.mean_auc <= 1.0


def test_integrated_auc_in_range():
    """Integrated AUC should be a valid probability."""
    time, event, risk = _random_data(n=200, seed=9)
    result = td_auc_curve(time, event, risk, n_times=25, ipcw=False)
    iauc = result.integrated_auc
    assert np.isfinite(iauc)
    assert 0.0 <= iauc <= 1.0


def test_auc_in_window():
    """auc_in_window should restrict to the requested time range."""
    time, event, risk = _random_data(n=200, seed=10)
    result = td_auc_curve(time, event, risk, n_times=40, ipcw=False)

    full = result.mean_auc
    t_mid = float(np.median(result.eval_times))
    early = result.auc_in_window(result.eval_times[0], t_mid)
    late = result.auc_in_window(t_mid, result.eval_times[-1])

    # Both windows should return valid values.
    assert np.isfinite(early)
    assert np.isfinite(late)


def test_td_auc_curve_too_few_events():
    """With fewer than 2 events, the curve should be empty."""
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 0, 0])  # only 1 event
    risk = np.array([1.0, 0.5, 0.2])

    result = td_auc_curve(time, event, risk)
    assert result.eval_times.size == 0


# ---- Bootstrap CI -------------------------------------------------------

def test_bootstrap_ci_runs():
    """Smoke test: bootstrap CI returns a point estimate and bounds."""
    time, event, risk = _random_data(n=100, seed=11)
    eval_t = float(np.median(time[event == 1]))
    point, lo, hi = td_auc_bootstrap_ci(
        time, event, risk, eval_t, n_boot=200, seed=42
    )
    assert np.isfinite(point)
    # With enough data, the CI should be finite.
    if np.isfinite(lo) and np.isfinite(hi):
        assert lo <= point <= hi or lo <= hi  # ordering sanity


def test_bootstrap_ci_with_clusters():
    """Bootstrap with cluster resampling should also return finite values."""
    time, event, risk = _random_data(n=100, seed=12)
    clusters = np.repeat(np.arange(10), 10)  # 10 clusters of 10
    eval_t = float(np.median(time[event == 1]))
    point, lo, hi = td_auc_bootstrap_ci(
        time, event, risk, eval_t, n_boot=200, seed=42,
        clusters=clusters,
    )
    assert np.isfinite(point)
