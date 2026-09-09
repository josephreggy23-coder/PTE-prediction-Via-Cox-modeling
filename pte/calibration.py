"""Brier score and calibration diagnostics for survival predictions.

Discrimination (concordance, AUC) measures whether a model ranks
subjects correctly; calibration measures whether the predicted
probabilities match the observed event rates.  A well-discriminating
model with poor calibration can assign a 60 % seizure probability to
a group whose true rate is 20 %.  For clinical translation of PTE
risk estimates, calibration is as important as discrimination.

Brier score for survival models
-------------------------------
The standard Brier score for binary outcomes is:

    BS = (1/N) * sum_i (p_i - y_i)^2

For survival data, Graf et al. (1999) extended this to a time-dependent
Brier score that accounts for censoring.  At evaluation time t:

    BS(t) = (1/N) * sum_i  w_i * (y_i(t) - S_hat(t | x_i))^2

where y_i(t) = I(T_i > t) is the observed survival status at t, and
w_i is an inverse-probability-of-censoring weight (IPCW) that
corrects for the fact that censored subjects have unknown status:

    - If T_i <= t and delta_i = 1 (observed event before t):
      y_i(t) = 0 (did not survive to t), w_i = 1 / G(T_i)

    - If T_i > t (still at risk):
      y_i(t) = 1 (survived to t), w_i = 1 / G(t)

    - If T_i <= t and delta_i = 0 (censored before t):
      excluded (unknown status).

G(.) is the Kaplan-Meier estimate of the censoring survival function.

The integrated Brier score (IBS) averages BS(t) over a range of
evaluation times:

    IBS = (1 / (t_max - t_min)) * integral_{t_min}^{t_max} BS(t) dt

A reference-model Brier score (using the Kaplan-Meier marginal
survival as the prediction for every subject) provides a benchmark.
The scaled Brier skill score:

    BSS(t) = 1 - BS_model(t) / BS_reference(t)

is analogous to R^2: positive values mean the model outperforms the
marginal, 0 means no improvement, and negative values mean the model
is worse than predicting the overall KM curve for everyone.

References
----------
Graf E, Schmoor C, Sauerbrei W, Schumacher M. Assessment and
comparison of prognostic classification schemes for survival data.
Statistics in Medicine, 1999; 18:2529-2545.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# NumPy >= 2.0 renamed trapz to trapezoid; support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ------------------------------------------------------------------
# Kaplan-Meier estimator
# ------------------------------------------------------------------
def kaplan_meier(
    time: np.ndarray,
    event: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier survival curve.

    Parameters
    ----------
    time
        Observed durations.
    event
        Event indicator (1 = event, 0 = censored).

    Returns
    -------
    km_times : ndarray
        Sorted unique event times.
    km_surv : ndarray
        Survival probability S(t) just after each event time.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    order = np.argsort(time, kind="stable")
    t_sorted = time[order]
    e_sorted = event[order]

    unique_times = np.unique(t_sorted[e_sorted == 1])
    surv = 1.0
    km_times, km_surv = [], []

    for t in unique_times:
        at_risk = int(np.sum(t_sorted >= t))
        n_events = int(np.sum((t_sorted == t) & (e_sorted == 1)))
        if at_risk > 0:
            surv *= 1.0 - n_events / at_risk
        km_times.append(t)
        km_surv.append(surv)

    return np.array(km_times), np.array(km_surv)


def _km_at_time(t: float, km_times: np.ndarray, km_surv: np.ndarray) -> float:
    """Look up S(t) from a KM curve at time t."""
    idx = np.searchsorted(km_times, t, side="right") - 1
    if idx < 0:
        return 1.0
    return float(km_surv[idx])


def _censoring_km(
    time: np.ndarray, event: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """KM estimate of the censoring distribution (events and censoring swapped)."""
    return kaplan_meier(time, 1.0 - event)


# ------------------------------------------------------------------
# Brier score at a single time point
# ------------------------------------------------------------------
def brier_score_at_time(
    time: np.ndarray,
    event: np.ndarray,
    survival_pred: np.ndarray,
    eval_time: float,
) -> float:
    """IPCW Brier score at a single evaluation time.

    Parameters
    ----------
    time
        Observed durations (time to event or censoring).
    event
        Event indicator (1 = event, 0 = censored).
    survival_pred
        Predicted survival probability S(eval_time | x_i) for each subject.
    eval_time
        The time point at which to evaluate calibration.

    Returns
    -------
    Brier score at ``eval_time``, or NaN if the censoring KM drops
    to zero before the evaluation time.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    survival_pred = np.asarray(survival_pred, dtype=float)
    n = len(time)

    km_t, km_s = _censoring_km(time, event)

    score = 0.0
    weight_sum = 0.0

    for i in range(n):
        # Case 1: observed event before eval_time
        if time[i] <= eval_time and event[i] == 1:
            g = max(_km_at_time(time[i], km_t, km_s), 1e-8)
            w = 1.0 / g
            # y_i(t) = 0 (did not survive), so squared error is S_hat^2
            score += w * survival_pred[i] ** 2
            weight_sum += w

        # Case 2: still at risk at eval_time
        elif time[i] > eval_time:
            g = max(_km_at_time(eval_time, km_t, km_s), 1e-8)
            w = 1.0 / g
            # y_i(t) = 1 (survived), so squared error is (1 - S_hat)^2
            score += w * (1.0 - survival_pred[i]) ** 2
            weight_sum += w

        # Case 3: censored before eval_time -- excluded

    if weight_sum == 0:
        return np.nan
    return float(score / weight_sum)


# ------------------------------------------------------------------
# Full Brier score curve
# ------------------------------------------------------------------
@dataclass
class BrierResult:
    """Brier score curve and integrated summary."""

    eval_times: np.ndarray
    brier_scores: np.ndarray
    reference_scores: np.ndarray

    @property
    def integrated_brier_score(self) -> float:
        """Integrated Brier score across the evaluation window."""
        valid = np.isfinite(self.brier_scores)
        if valid.sum() < 2:
            return np.nan
        t = self.eval_times[valid]
        bs = self.brier_scores[valid]
        return float(_trapezoid(bs, t) / (t[-1] - t[0]))

    @property
    def integrated_reference(self) -> float:
        """Integrated reference (KM marginal) Brier score."""
        valid = np.isfinite(self.reference_scores)
        if valid.sum() < 2:
            return np.nan
        t = self.eval_times[valid]
        rs = self.reference_scores[valid]
        return float(_trapezoid(rs, t) / (t[-1] - t[0]))

    @property
    def brier_skill_score(self) -> float:
        """Scaled Brier skill score: 1 - IBS_model / IBS_reference."""
        ibs = self.integrated_brier_score
        ref = self.integrated_reference
        if not np.isfinite(ref) or ref == 0:
            return np.nan
        return float(1.0 - ibs / ref)


def brier_score_curve(
    time: np.ndarray,
    event: np.ndarray,
    survival_pred_fn,
    n_times: int = 50,
    quantile_range: tuple[float, float] = (0.05, 0.95),
) -> BrierResult:
    """Compute the IPCW Brier score across the follow-up.

    Parameters
    ----------
    time
        Observed durations.
    event
        Event indicator.
    survival_pred_fn
        Callable that takes an evaluation time t and returns an array of
        predicted survival probabilities S(t | x_i) for each subject.
    n_times
        Number of evaluation time points.
    quantile_range
        Quantiles of event times defining the evaluation grid.

    Returns
    -------
    BrierResult
        Contains the BS(t) curve, reference (marginal KM) curve, and
        integrated summaries.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)

    event_times = time[event == 1]
    if event_times.size < 2:
        return BrierResult(
            eval_times=np.array([]),
            brier_scores=np.array([]),
            reference_scores=np.array([]),
        )

    t_lo = float(np.quantile(event_times, quantile_range[0]))
    t_hi = float(np.quantile(event_times, quantile_range[1]))
    eval_times = np.linspace(t_lo, t_hi, n_times)

    # Reference model: Kaplan-Meier marginal survival.
    km_t, km_s = kaplan_meier(time, event)

    bs_model = np.full(n_times, np.nan)
    bs_ref = np.full(n_times, np.nan)

    for i, t in enumerate(eval_times):
        s_pred = survival_pred_fn(t)
        bs_model[i] = brier_score_at_time(time, event, s_pred, t)

        # Reference: predict KM marginal for everyone.
        s_km = _km_at_time(t, km_t, km_s)
        bs_ref[i] = brier_score_at_time(
            time, event, np.full(len(time), s_km), t
        )

    return BrierResult(
        eval_times=eval_times,
        brier_scores=bs_model,
        reference_scores=bs_ref,
    )
