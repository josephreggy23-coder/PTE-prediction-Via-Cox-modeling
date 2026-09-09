"""Time-dependent discrimination measures for Cox survival models.

The concordance index in ``cv.py`` gives a single summary of
discrimination across the entire follow-up window.  That is fine for an
overall report card, but post-traumatic epileptogenesis has a specific
clinical window of interest: early versus late seizure onset.  A model
might discriminate well among larvae that seize in the first hour but
poorly among those with longer latencies (or vice versa).

Time-dependent AUC answers "at time t, how well does the risk score
separate larvae that have already seized from those still seizure-free?"
Plotting AUC(t) across the follow-up reveals where the model's
discrimination is strongest and where it degrades.

Methodology
-----------
We implement a variant of Uno's estimator (Uno et al., Statistics in
Medicine 2007, 26:2389-2430).  At each evaluation time t:

1.  Define cases as subjects with event time <= t and event == 1.
2.  Define controls as subjects with observed time > t (still at risk).
3.  Among (case, control) pairs, the AUC is the fraction for which
    the case has a higher risk score.
4.  Censored subjects with observed time <= t are excluded -- they are
    neither clear cases nor clear controls.

Inverse-probability-of-censoring weights (IPCW) correct for the bias
introduced when censoring is informative.  The Kaplan-Meier estimator
of the censoring distribution provides the weights.

This module exposes both the full AUC(t) curve and summary statistics
that collapse it into clinically relevant numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C


# ------------------------------------------------------------------
# Kaplan-Meier for the censoring distribution
# ------------------------------------------------------------------
def _km_censoring(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier estimate of the CENSORING survival function.

    The censoring distribution swaps the role of events and censoring:
    an observed event is "censored" and a censoring event is the "event".
    This is used for inverse-probability-of-censoring weights.

    Returns
    -------
    km_times : ndarray
        Sorted unique times at which the censoring survival changes.
    km_surv : ndarray
        Censoring survival probability just after each time.
    """
    # Swap: event -> censored, censored -> event.
    cens_event = 1.0 - event

    order = np.argsort(time, kind="stable")
    t_sorted = time[order]
    e_sorted = cens_event[order]

    unique_times = np.unique(t_sorted)
    surv = 1.0
    km_times = []
    km_surv = []

    for t in unique_times:
        at_risk = np.sum(t_sorted >= t)
        n_cens_events = np.sum((t_sorted == t) & (e_sorted == 1))
        if at_risk > 0 and n_cens_events > 0:
            surv *= 1.0 - n_cens_events / at_risk
        km_times.append(t)
        km_surv.append(surv)

    return np.array(km_times), np.array(km_surv)


def _get_censoring_weight(t: float, km_times: np.ndarray, km_surv: np.ndarray) -> float:
    """Look up G(t) = P(C > t) from the censoring KM curve.

    Returns the censoring survival probability at the largest KM time <= t.
    If t is before all KM times, returns 1.0 (no censoring yet).
    """
    idx = np.searchsorted(km_times, t, side="right") - 1
    if idx < 0:
        return 1.0
    return max(km_surv[idx], 1e-8)  # floor to avoid division by zero


# ------------------------------------------------------------------
# Time-dependent AUC at a single evaluation time
# ------------------------------------------------------------------
def td_auc_at_time(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    eval_time: float,
    ipcw: bool = True,
) -> float:
    """Time-dependent AUC at a single evaluation time.

    Parameters
    ----------
    time
        Observed durations (time to event or censoring).
    event
        Event indicator (1 = seizure, 0 = censored).
    risk
        Risk score from the model (higher = higher hazard).
    eval_time
        The time point t at which to evaluate discrimination.
    ipcw
        Whether to apply inverse-probability-of-censoring weights.

    Returns
    -------
    AUC at ``eval_time``, or ``nan`` if there are fewer than one case
    and one control.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk = np.asarray(risk, dtype=float)

    # Cases: event occurred at or before eval_time.
    cases = (time <= eval_time) & (event == 1)
    # Controls: still at risk (observed time > eval_time).
    controls = time > eval_time

    n_cases = int(cases.sum())
    n_controls = int(controls.sum())
    if n_cases < 1 or n_controls < 1:
        return np.nan

    # Compute IPCW weights if requested.
    if ipcw:
        km_t, km_s = _km_censoring(time, event)
    else:
        km_t, km_s = None, None

    # Count concordant, discordant, and tied pairs.
    case_risks = risk[cases]
    control_risks = risk[controls]

    if ipcw:
        case_times = time[cases]
        case_weights = np.array([
            1.0 / _get_censoring_weight(t_i, km_t, km_s) for t_i in case_times
        ])
        control_weight = 1.0 / _get_censoring_weight(eval_time, km_t, km_s)
    else:
        case_weights = np.ones(n_cases)
        control_weight = 1.0

    numerator = 0.0
    denominator = 0.0
    for i in range(n_cases):
        w_i = case_weights[i] * control_weight
        concordant = np.sum((case_risks[i] > control_risks) * w_i)
        tied = 0.5 * np.sum((case_risks[i] == control_risks) * w_i)
        numerator += concordant + tied
        denominator += n_controls * w_i

    if denominator == 0:
        return np.nan
    return numerator / denominator


# ------------------------------------------------------------------
# Full AUC(t) curve
# ------------------------------------------------------------------
@dataclass
class TdAucResult:
    """Time-dependent AUC curve and summary statistics."""

    eval_times: np.ndarray
    auc_values: np.ndarray
    n_cases: np.ndarray
    n_controls: np.ndarray

    @property
    def mean_auc(self) -> float:
        """Mean AUC across all evaluated time points (excluding NaN)."""
        valid = np.isfinite(self.auc_values)
        if not valid.any():
            return np.nan
        return float(np.mean(self.auc_values[valid]))

    @property
    def integrated_auc(self) -> float:
        """Time-weighted (integrated) AUC, area under the AUC(t) curve.

        Uses the trapezoidal rule over the evaluated time points,
        normalized by the time span.  This gives more weight to
        regions of the follow-up where observations are dense.
        """
        valid = np.isfinite(self.auc_values)
        if valid.sum() < 2:
            return np.nan
        t = self.eval_times[valid]
        a = self.auc_values[valid]
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        return float(_trapz(a, t) / (t[-1] - t[0]))

    def auc_in_window(self, t_lo: float, t_hi: float) -> float:
        """Mean AUC restricted to a time window [t_lo, t_hi]."""
        mask = (
            (self.eval_times >= t_lo)
            & (self.eval_times <= t_hi)
            & np.isfinite(self.auc_values)
        )
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(self.auc_values[mask]))


def td_auc_curve(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    n_times: int = 50,
    quantile_range: tuple[float, float] = (0.05, 0.95),
    ipcw: bool = True,
) -> TdAucResult:
    """Compute the time-dependent AUC across the follow-up.

    Parameters
    ----------
    time
        Observed durations.
    event
        Event indicator (1 = event, 0 = censored).
    risk
        Risk score (higher = higher hazard).
    n_times
        Number of evaluation time points, evenly spaced across the
        quantile range of observed event times.
    quantile_range
        Quantiles of event times that define the evaluation grid.
        Restricting to (0.05, 0.95) avoids edge effects where there
        are very few cases or controls.
    ipcw
        Whether to apply inverse-probability-of-censoring weights.

    Returns
    -------
    TdAucResult
        Contains the AUC(t) curve, per-time-point case/control counts,
        and summary statistics.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk = np.asarray(risk, dtype=float)

    # Build evaluation grid from event-time quantiles.
    event_times = time[event == 1]
    if event_times.size < 2:
        return TdAucResult(
            eval_times=np.array([]),
            auc_values=np.array([]),
            n_cases=np.array([], dtype=int),
            n_controls=np.array([], dtype=int),
        )
    t_lo = float(np.quantile(event_times, quantile_range[0]))
    t_hi = float(np.quantile(event_times, quantile_range[1]))
    eval_times = np.linspace(t_lo, t_hi, n_times)

    auc_vals = np.full(n_times, np.nan)
    n_cases_arr = np.zeros(n_times, dtype=int)
    n_controls_arr = np.zeros(n_times, dtype=int)

    for i, t in enumerate(eval_times):
        cases = (time <= t) & (event == 1)
        controls = time > t
        n_cases_arr[i] = int(cases.sum())
        n_controls_arr[i] = int(controls.sum())
        auc_vals[i] = td_auc_at_time(time, event, risk, t, ipcw=ipcw)

    return TdAucResult(
        eval_times=eval_times,
        auc_values=auc_vals,
        n_cases=n_cases_arr,
        n_controls=n_controls_arr,
    )


# ------------------------------------------------------------------
# Bootstrap CI for time-dependent AUC
# ------------------------------------------------------------------
def td_auc_bootstrap_ci(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    eval_time: float,
    n_boot: int = C.N_BOOTSTRAP,
    seed: int = C.BOOTSTRAP_SEED,
    alpha: float = C.ALPHA,
    clusters: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for time-dependent AUC at one time point.

    Parameters
    ----------
    clusters
        If given (clutch ids), the bootstrap resamples whole clusters
        rather than individual subjects.

    Returns
    -------
    (point, lo, hi) : tuple of floats
        Point estimate and (1-alpha) percentile CI bounds.
    """
    rng = np.random.default_rng(seed)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk = np.asarray(risk, dtype=float)
    n = len(time)

    point = td_auc_at_time(time, event, risk, eval_time, ipcw=True)

    if clusters is not None:
        clusters = np.asarray(clusters)
        groups = [np.flatnonzero(clusters == g) for g in np.unique(clusters)]

    boots = []
    for _ in range(n_boot):
        if clusters is None:
            s = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(groups), len(groups))
            s = np.concatenate([groups[k] for k in pick])

        if event[s].sum() < 2:
            continue
        try:
            b = td_auc_at_time(time[s], event[s], risk[s], eval_time, ipcw=True)
            if np.isfinite(b):
                boots.append(b)
        except (ZeroDivisionError, ValueError):
            continue

    if len(boots) < 50:
        return point, np.nan, np.nan
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
