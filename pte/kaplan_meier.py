"""Kaplan-Meier survival curve estimation and log-rank testing.

This module implements the product-limit estimator of the survival
function, its variance via Greenwood's formula, confidence intervals
on the complementary log-log scale, the two-sample log-rank test,
and median survival extraction.  These are implemented from first
principles using only numpy and scipy, serving as both an educational
reference and a cross-check against the lifelines-based fits in
:mod:`pte.inference`.

Kaplan-Meier product-limit estimator
-------------------------------------
Given *n* subjects with observed times t_1 <= t_2 <= ... <= t_n and
event indicators delta_i in {0, 1}, the Kaplan-Meier estimator of
the survival function is:

    S_hat(t) = prod_{t_i <= t, delta_i = 1} (1 - d_i / n_i)

where the product runs over distinct event times, d_i is the number
of events at time t_i, and n_i is the number of subjects still at
risk (alive and uncensored) just before t_i.  Subjects censored at
time t_i are treated as still at risk at t_i but leave the risk set
immediately afterwards.

S_hat(t) is a right-continuous step function: it remains constant
between event times and drops at each observed event.  This is the
nonparametric maximum likelihood estimator of the survival function
(Kaplan & Meier, 1958).

Greenwood's formula
-------------------
The variance of S_hat(t) is estimated by Greenwood's formula:

    Var(S_hat(t)) = S_hat(t)^2 * sum_{t_i <= t} d_i / (n_i * (n_i - d_i))

The sum accumulates only over event times up to t.  At a time where
n_i = d_i (everyone at risk experiences the event), the term is
undefined; in practice S_hat drops to zero and the variance is set
to zero as well.

Log-log confidence intervals
-----------------------------
The standard "plain" confidence interval S_hat(t) +/- z * se(t) can
produce limits outside [0, 1].  The complementary log-log
transformation theta(t) = log(-log(S_hat(t))) maps (0, 1) to the
real line.  By the delta method:

    Var(theta(t)) ≈ Var(S_hat(t)) / (S_hat(t) * log(S_hat(t)))^2

A symmetric interval on the theta scale is then back-transformed:

    CI(S(t)) = exp(-exp(theta_hat(t) +/- z_{alpha/2} * se_theta(t)))

This guarantees the interval stays in (0, 1) and has better coverage
for small samples (Borgan & Liestol, 1990).

Log-rank test
-------------
The log-rank test compares the survival distributions of two groups
without assuming a parametric form.  At each distinct event time t_j
across both groups:

    O_{1j} = observed events in group 1
    E_{1j} = n_{1j} * d_j / n_j       (expected under H0)
    V_j   = n_{1j} * n_{2j} * d_j * (n_j - d_j) / (n_j^2 * (n_j - 1))

The test statistic is:

    chi^2 = (sum_j (O_{1j} - E_{1j}))^2 / sum_j V_j

Under H0 (identical survival curves), this follows a chi-squared
distribution with 1 degree of freedom.  This is equivalent to the
Mantel (1966) form of the Mantel-Haenszel test applied to survival
data and places equal weight on all time points.

Median survival time
--------------------
The median survival time is the smallest time t at which S_hat(t)
<= 0.5.  Its confidence interval is found by inverting the CI for
S(t): the lower and upper bounds of the median are the times where
the upper and lower CI bounds (respectively) first cross 0.5.

References
----------
Kaplan EL, Meier P. Nonparametric estimation from incomplete
observations. Journal of the American Statistical Association,
1958; 53(282):457-481.

Greenwood M. The natural duration of cancer. Reports on Public
Health and Medical Subjects, 1926; 33:1-26.

Borgan O, Liestol K. A note on confidence intervals and bands for
the survival function based on transformations. Scandinavian Journal
of Statistics, 1990; 17(1):35-41.

Mantel N. Evaluation of survival data and two new rank order
statistics arising in its consideration. Cancer Chemotherapy
Reports, 1966; 50(3):163-170.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------
@dataclass(frozen=True)
class KaplanMeierResult:
    """Result of a Kaplan-Meier survival curve estimation.

    Attributes
    ----------
    time
        Sorted distinct event times at which S(t) changes.
    survival
        S_hat(t) at each event time.
    n_at_risk
        Number at risk just before each event time.
    n_events
        Number of events at each event time.
    n_censored
        Number of censorings at each event time (before leaving
        the risk set).
    se
        Standard error of S_hat(t) via Greenwood's formula.
    """

    time: np.ndarray
    survival: np.ndarray
    n_at_risk: np.ndarray
    n_events: np.ndarray
    n_censored: np.ndarray
    se: np.ndarray


@dataclass(frozen=True)
class ConfidenceInterval:
    """Pointwise confidence band for S(t).

    Attributes
    ----------
    lower
        Lower bound of the confidence interval at each event time.
    upper
        Upper bound of the confidence interval at each event time.
    alpha
        Significance level (two-sided).
    method
        Name of the CI method (e.g. ``"log-log"``).
    """

    lower: np.ndarray
    upper: np.ndarray
    alpha: float
    method: str


@dataclass(frozen=True)
class LogRankResult:
    """Result of a two-sample log-rank test.

    Attributes
    ----------
    statistic
        Chi-squared test statistic.
    df
        Degrees of freedom (always 1 for a two-sample test).
    p_value
        P-value from the chi-squared distribution.
    observed_1
        Total observed events in group 1.
    expected_1
        Total expected events in group 1 under H0.
    """

    statistic: float
    df: int
    p_value: float
    observed_1: float
    expected_1: float


@dataclass(frozen=True)
class MedianSurvival:
    """Median survival time with confidence interval.

    Attributes
    ----------
    median
        Median survival time, or ``np.inf`` if S(t) never reaches 0.5.
    ci_lower
        Lower bound of the CI for the median.
    ci_upper
        Upper bound of the CI for the median.
    alpha
        Significance level used for the CI.
    """

    median: float
    ci_lower: float
    ci_upper: float
    alpha: float


# ------------------------------------------------------------------
# Input validation
# ------------------------------------------------------------------
def _validate_inputs(
    times: np.ndarray,
    events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce survival data arrays.

    Parameters
    ----------
    times
        Observed durations; must be non-negative.
    events
        Event indicators (1 = event, 0 = censored).

    Returns
    -------
    times, events
        1-D float64 arrays.
    """
    times = np.asarray(times, dtype=np.float64).ravel()
    events = np.asarray(events, dtype=np.float64).ravel()
    if times.shape[0] != events.shape[0]:
        raise ValueError(
            f"times and events must have the same length, "
            f"got {times.shape[0]} and {events.shape[0]}."
        )
    if times.shape[0] == 0:
        raise ValueError("times and events must not be empty.")
    if np.any(times < 0):
        raise ValueError("Observation times must be non-negative.")
    if not np.all(np.isin(events, [0.0, 1.0])):
        raise ValueError("Event indicators must be 0 or 1.")
    return times, events


# ------------------------------------------------------------------
# Kaplan-Meier estimator
# ------------------------------------------------------------------
def kaplan_meier(
    times: np.ndarray,
    events: np.ndarray,
) -> KaplanMeierResult:
    """Compute the Kaplan-Meier product-limit survival estimator.

    The KM estimator is a step function that decreases at each distinct
    event time.  At time t_i with d_i events out of n_i subjects at
    risk, the survival probability drops by the factor (1 - d_i / n_i).

    Parameters
    ----------
    times
        Observed durations (time to event or censoring).
    events
        Event indicator: 1 = event occurred, 0 = right-censored.

    Returns
    -------
    KaplanMeierResult
        Contains distinct event times, survival probabilities,
        number at risk, number of events, number censored, and
        standard errors from Greenwood's formula.
    """
    times, events = _validate_inputs(times, events)
    n_total = len(times)

    # Sort by time; within ties, censored observations come after
    # events so that they are still at risk at their censoring time.
    order = np.lexsort((-events, times))
    t_sorted = times[order]
    e_sorted = events[order]

    # Find distinct time points and aggregate events and censorings.
    unique_times = np.unique(t_sorted)
    n_times = len(unique_times)

    d = np.zeros(n_times, dtype=np.float64)  # events at each time
    c = np.zeros(n_times, dtype=np.float64)  # censorings at each time

    for i, t in enumerate(unique_times):
        mask = t_sorted == t
        d[i] = e_sorted[mask].sum()
        c[i] = mask.sum() - d[i]

    # Number at risk just before each time point.
    n_risk = np.zeros(n_times, dtype=np.float64)
    n_risk[0] = n_total
    for i in range(1, n_times):
        n_risk[i] = n_risk[i - 1] - d[i - 1] - c[i - 1]

    # Restrict to times where at least one event occurred.
    event_mask = d > 0
    event_times = unique_times[event_mask]
    d_events = d[event_mask]
    n_events_risk = n_risk[event_mask]
    c_events = c[event_mask]

    if len(event_times) == 0:
        # All observations are censored: survival stays at 1.
        return KaplanMeierResult(
            time=unique_times,
            survival=np.ones(n_times),
            n_at_risk=n_risk,
            n_events=d,
            n_censored=c,
            se=np.zeros(n_times),
        )

    # Product-limit estimate.
    survival = np.cumprod(1.0 - d_events / n_events_risk)

    # Greenwood variance.
    se = greenwood_variance(survival, n_events_risk, d_events)

    return KaplanMeierResult(
        time=event_times,
        survival=survival,
        n_at_risk=n_events_risk,
        n_events=d_events,
        n_censored=c_events,
        se=se,
    )


# ------------------------------------------------------------------
# Greenwood's variance
# ------------------------------------------------------------------
def greenwood_variance(
    survival: np.ndarray,
    n_at_risk: np.ndarray,
    n_events: np.ndarray,
) -> np.ndarray:
    """Greenwood's formula for the standard error of S_hat(t).

    The variance of the Kaplan-Meier estimator at time t is:

        Var(S_hat(t)) = S_hat(t)^2 * sum_{t_i <= t} d_i / (n_i * (n_i - d_i))

    where d_i is events at t_i and n_i is the number at risk.  The
    standard error is the square root.

    When n_i == d_i (all at risk experience the event), the term
    d_i / (n_i * (n_i - d_i)) is undefined; S_hat drops to zero and
    we set the SE to zero.

    Parameters
    ----------
    survival
        Cumulative survival probabilities at event times.
    n_at_risk
        Number at risk at each event time.
    n_events
        Number of events at each event time.

    Returns
    -------
    np.ndarray
        Standard error of S_hat at each event time.
    """
    survival = np.asarray(survival, dtype=np.float64)
    n_at_risk = np.asarray(n_at_risk, dtype=np.float64)
    n_events = np.asarray(n_events, dtype=np.float64)

    denom = n_at_risk * (n_at_risk - n_events)
    # Guard against division by zero when all at risk had the event.
    safe = denom > 0
    safe_denom = np.where(safe, denom, 1.0)  # avoid runtime warning
    terms = np.where(safe, n_events / safe_denom, 0.0)
    cumulative_sum = np.cumsum(terms)
    variance = survival**2 * cumulative_sum
    return np.sqrt(np.maximum(variance, 0.0))


# ------------------------------------------------------------------
# Log-log confidence interval
# ------------------------------------------------------------------
def log_log_ci(
    survival: np.ndarray,
    se: np.ndarray,
    alpha: float = 0.05,
) -> ConfidenceInterval:
    """Confidence interval on the complementary log-log scale.

    The transformation theta(t) = log(-log(S_hat(t))) maps (0, 1) to
    the real line.  By the delta method, the variance of theta is:

        Var(theta) = se(S)^2 / (S * log(S))^2

    A symmetric interval on the theta scale is back-transformed:

        S_lower, S_upper = exp(-exp(theta +/- z * se_theta))

    This guarantees the CI stays in (0, 1) and has better small-sample
    coverage than the plain interval.

    Parameters
    ----------
    survival
        KM survival estimates at event times (from :func:`kaplan_meier`).
    se
        Standard errors of S_hat (from :func:`greenwood_variance`).
    alpha
        Significance level (default 0.05 for 95 % CI).

    Returns
    -------
    ConfidenceInterval
        Lower and upper bounds at each event time.
    """
    survival = np.asarray(survival, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    lower = np.full_like(survival, np.nan)
    upper = np.full_like(survival, np.nan)

    # The transformation is only valid for 0 < S < 1.
    valid = (survival > 0) & (survival < 1) & (se > 0)

    log_S = np.log(survival[valid])
    theta = np.log(-log_S)
    # Delta-method SE on the theta scale:
    #   se_theta = se(S) / |S * log(S)|
    se_theta = se[valid] / np.abs(survival[valid] * log_S)

    theta_lower = theta - z * se_theta
    theta_upper = theta + z * se_theta

    lower[valid] = np.exp(-np.exp(theta_upper))  # note inversion
    upper[valid] = np.exp(-np.exp(theta_lower))

    # Where S == 1, the CI is [1, 1]; where S == 0, CI is [0, 0].
    lower[survival == 1.0] = 1.0
    upper[survival == 1.0] = 1.0
    lower[survival == 0.0] = 0.0
    upper[survival == 0.0] = 0.0

    # Clip to [0, 1] for safety.
    lower = np.clip(lower, 0.0, 1.0)
    upper = np.clip(upper, 0.0, 1.0)

    return ConfidenceInterval(
        lower=lower,
        upper=upper,
        alpha=alpha,
        method="log-log",
    )


# ------------------------------------------------------------------
# Log-rank test
# ------------------------------------------------------------------
def log_rank_test(
    times1: np.ndarray,
    events1: np.ndarray,
    times2: np.ndarray,
    events2: np.ndarray,
) -> LogRankResult:
    """Two-sample log-rank test for equality of survival functions.

    At each distinct event time t_j (pooled across groups), the test
    computes:

        O_{1j} = observed events in group 1 at t_j
        E_{1j} = n_{1j} * d_j / n_j

    where n_{1j} is the number at risk in group 1, d_j is total events,
    and n_j is total at risk.  The hypergeometric variance at each time
    is:

        V_j = n_{1j} * n_{2j} * d_j * (n_j - d_j) / (n_j^2 * (n_j - 1))

    The test statistic is:

        chi^2 = (sum_j (O_{1j} - E_{1j}))^2 / sum_j V_j

    Under H0 it follows chi^2 with 1 degree of freedom.

    Parameters
    ----------
    times1, events1
        Observed times and event indicators for group 1.
    times2, events2
        Observed times and event indicators for group 2.

    Returns
    -------
    LogRankResult
        Test statistic, degrees of freedom, p-value, and observed
        vs expected event counts for group 1.
    """
    times1, events1 = _validate_inputs(times1, events1)
    times2, events2 = _validate_inputs(times2, events2)

    # Pool all distinct event times (where at least one event occurs).
    all_times = np.concatenate([times1, times2])
    all_events = np.concatenate([events1, events2])
    event_times = np.unique(all_times[all_events == 1])

    if len(event_times) == 0:
        # No events in either group: test is undefined.
        return LogRankResult(
            statistic=0.0,
            df=1,
            p_value=1.0,
            observed_1=0.0,
            expected_1=0.0,
        )

    # At each event time, compute observed events and number at risk
    # in each group.
    O_minus_E = 0.0
    V_total = 0.0
    total_observed_1 = 0.0
    total_expected_1 = 0.0

    for t in event_times:
        # Number at risk: subjects with observed time >= t.
        n1 = np.sum(times1 >= t)
        n2 = np.sum(times2 >= t)
        n = n1 + n2

        # Observed events at this time.
        d1 = np.sum((times1 == t) & (events1 == 1))
        d2 = np.sum((times2 == t) & (events2 == 1))
        d = d1 + d2

        # Expected events in group 1 under H0.
        e1 = n1 * d / n

        total_observed_1 += d1
        total_expected_1 += e1
        O_minus_E += d1 - e1

        # Hypergeometric variance.
        if n > 1:
            V_total += n1 * n2 * d * (n - d) / (n**2 * (n - 1))

    if V_total == 0:
        return LogRankResult(
            statistic=0.0,
            df=1,
            p_value=1.0,
            observed_1=total_observed_1,
            expected_1=total_expected_1,
        )

    chi2 = O_minus_E**2 / V_total
    p_value = 1.0 - stats.chi2.cdf(chi2, df=1)

    return LogRankResult(
        statistic=float(chi2),
        df=1,
        p_value=float(p_value),
        observed_1=float(total_observed_1),
        expected_1=float(total_expected_1),
    )


# ------------------------------------------------------------------
# Median survival time
# ------------------------------------------------------------------
def median_survival_time(
    km_result: KaplanMeierResult,
    alpha: float = 0.05,
) -> MedianSurvival:
    """Extract median survival time from a Kaplan-Meier curve.

    The median is the smallest time t at which S_hat(t) <= 0.5.  If
    the survival curve never reaches 0.5 (e.g. heavy censoring), the
    median is reported as ``np.inf``.

    The confidence interval for the median is obtained by inverting the
    log-log confidence band: the CI lower bound is the time where the
    upper CI of S(t) first crosses 0.5, and the CI upper bound is the
    time where the lower CI of S(t) first crosses 0.5.

    Parameters
    ----------
    km_result
        Output of :func:`kaplan_meier`.
    alpha
        Significance level for the CI (default 0.05 for 95 % CI).

    Returns
    -------
    MedianSurvival
        Median and its confidence interval.
    """
    survival = km_result.survival
    time = km_result.time

    # Find the median: first time S(t) <= 0.5.
    median = _find_crossing(time, survival, 0.5)

    # Confidence interval by inverting the CI band.
    # The lower CI of S(t) crosses 0.5 earliest => lower bound of median.
    # The upper CI of S(t) crosses 0.5 latest  => upper bound of median.
    ci = log_log_ci(survival, km_result.se, alpha=alpha)
    ci_lower = _find_crossing(time, ci.lower, 0.5)
    ci_upper = _find_crossing(time, ci.upper, 0.5)

    return MedianSurvival(
        median=float(median),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        alpha=alpha,
    )


def _find_crossing(
    time: np.ndarray,
    curve: np.ndarray,
    level: float,
) -> float:
    """Find the first time a step function crosses below a level.

    Returns ``np.inf`` if the curve never reaches the level.
    """
    indices = np.where(curve <= level)[0]
    if len(indices) == 0:
        return np.inf
    return float(time[indices[0]])
