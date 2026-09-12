"""Proportional hazards diagnostics via Schoenfeld residuals.

The proportional hazards (PH) assumption
-----------------------------------------
The Cox model specifies the hazard for subject *i* as:

    h(t | x_i) = h_0(t) * exp(x_i . beta)

where h_0(t) is an unspecified baseline hazard and beta is a vector
of regression coefficients.  The key structural assumption is that the
hazard ratio between any two subjects is constant over time:

    HR(t) = h(t | x_1) / h(t | x_2) = exp((x_1 - x_2) . beta)

When PH holds, each coefficient summarises the covariate's effect
across the entire follow-up.  When PH is violated -- for example, a
treatment helps early but its benefit wanes -- the single coefficient
is a misleading time-averaged quantity, and the predicted hazard
ratios may be systematically wrong at the time horizons that matter
most.  Checking PH is therefore arguably the most important diagnostic
for any Cox model.

Schoenfeld residuals
--------------------
Schoenfeld (1982) defined partial residuals that exist only at
uncensored event times, one per covariate per event.  At event time
t_k, where subject (k) experiences the event, the Schoenfeld residual
for covariate j is:

    r_j(t_k) = x_j(k) - E[x_j | R(t_k)]

where x_j(k) is covariate j for the subject who had the event, and
E[x_j | R(t_k)] is the risk-set-weighted expectation:

    E[x_j | R(t_k)] = sum_{i in R(t_k)} x_{ij} w_i
                       / sum_{i in R(t_k)} w_i

with weights w_i = exp(x_i . beta_hat).  The risk set R(t_k) contains
all subjects whose observed time T_i >= t_k.

Under the null hypothesis that PH holds and beta is the true parameter
vector, the Schoenfeld residuals have expected value zero and are
uncorrelated with time.

Scaled Schoenfeld residuals and time-varying coefficients
---------------------------------------------------------
Grambsch & Therneau (1994) showed that *scaled* Schoenfeld residuals
estimate the time-varying coefficient beta(t).  Define:

    r*_j(t_k) = beta_hat_j + d * r_j(t_k) / I_jj

where d is the total number of events and I_jj is the (j, j) element
of the observed information matrix of the Cox partial likelihood:

    I_jj = sum_{k=1}^{d} Var_w(x_j | R(t_k))

The weighted variance at each event time is:

    Var_w(x_j | R(t_k)) = sum_{i in R(t_k)} w_i (x_{ij} - x_bar_j)^2
                           / sum_{i in R(t_k)} w_i

If the true model has a time-varying coefficient beta_j(t) = beta_j +
f(t), then E[r*_j(t_k)] = beta_j(t).  Under PH (f(t) = 0), the
scaled residuals scatter about the constant beta_hat_j.  A systematic
trend in r*_j against time signals a PH violation: the covariate's
effect is changing.

Correlation test
----------------
The formal test correlates the scaled Schoenfeld residuals with a
transformation of event time g(t).  Common choices are:

    - Identity: g(t) = t         (linear drift)
    - Log:      g(t) = log(t)    (early departures)
    - Rank:     g(t) = rank(t)   (robust, default)

The per-covariate test statistic is:

    chi^2_j = rho_j^2 * d

where rho_j is the Pearson correlation between r*_j(t_k) and g(t_k),
and d is the number of events.  Under H0 this is approximately
chi-squared with 1 degree of freedom.

The global test sums the per-covariate statistics:

    chi^2_global = sum_j chi^2_j     (df = p)

A small p-value (conventionally < 0.05) indicates evidence against the
PH assumption for that covariate (or globally).

References
----------
Schoenfeld D. Partial residuals for the proportional hazards
regression model. Biometrika, 1982; 69(1):239-241.

Grambsch PM, Therneau TM. Proportional hazards tests and diagnostics
based on weighted residuals. Biometrika, 1994; 81(3):515-526.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------
@dataclass(frozen=True)
class CovariateTestResult:
    """PH test result for one covariate.

    Attributes
    ----------
    name
        Covariate label.
    correlation
        Pearson correlation between scaled Schoenfeld residuals and
        the time transformation.
    chi_sq
        Test statistic (rho^2 * d).
    df
        Degrees of freedom (always 1 for a single covariate).
    p_value
        P-value from the chi-squared distribution.
    """

    name: str
    correlation: float
    chi_sq: float
    df: int
    p_value: float


@dataclass
class PHTestResult:
    """Proportional hazards assumption test results.

    Attributes
    ----------
    covariate_results
        Per-covariate test statistics.
    global_chi_sq
        Sum of per-covariate chi-squared statistics.
    global_df
        Total degrees of freedom (number of testable covariates).
    global_p_value
        P-value for the global test.
    event_times
        Sorted uncensored event times, shape (d,).
    schoenfeld_residuals
        Raw Schoenfeld residuals, shape (d, p).
    scaled_schoenfeld_residuals
        Scaled Schoenfeld residuals, shape (d, p).
    time_transform
        Name of the time transformation used.
    """

    covariate_results: list[CovariateTestResult]
    global_chi_sq: float
    global_df: int
    global_p_value: float
    event_times: np.ndarray
    schoenfeld_residuals: np.ndarray
    scaled_schoenfeld_residuals: np.ndarray
    time_transform: str


# ------------------------------------------------------------------
# Input helpers
# ------------------------------------------------------------------
def _validate_inputs(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    coef: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coerce and check dimensions."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    coef = np.asarray(coef, dtype=float).ravel()

    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n, p = covariates.shape
    if n != len(time):
        raise ValueError(
            f"covariates has {n} rows but time has {len(time)} entries"
        )
    if n != len(event):
        raise ValueError(
            f"covariates has {n} rows but event has {len(event)} entries"
        )
    if p != len(coef):
        raise ValueError(
            f"covariates has {p} columns but coef has {len(coef)} entries"
        )

    return time, event, covariates, coef


def _apply_time_transform(t: np.ndarray, transform: str) -> np.ndarray:
    """Apply a time transformation for the correlation test.

    Parameters
    ----------
    t
        Event times.
    transform
        One of ``'identity'``, ``'log'``, or ``'rank'``.

    Returns
    -------
    Transformed time values.
    """
    if transform == "identity":
        return t.copy()
    elif transform == "log":
        return np.log(t)
    elif transform == "rank":
        return stats.rankdata(t).astype(float)
    else:
        raise ValueError(
            f"Unknown time_transform {transform!r}; "
            "use 'identity', 'log', or 'rank'"
        )


# ------------------------------------------------------------------
# Core computation
# ------------------------------------------------------------------
def _compute_residuals_and_info(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    coef: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Schoenfeld residuals and the information diagonal.

    This is the workhorse shared by the public functions.

    Returns
    -------
    event_times : ndarray, shape (d,)
        Sorted uncensored event times.
    residuals : ndarray, shape (d, p)
        Schoenfeld residuals.
    info_diag : ndarray, shape (p,)
        Diagonal of the observed information matrix,
        I_jj = sum_k Var_w(x_j | R(t_k)).
    """
    time, event, covariates, coef = _validate_inputs(
        time, event, covariates, coef
    )
    n, p = covariates.shape

    # Linear predictor.
    eta = covariates @ coef

    # Indices of subjects with observed events, sorted by time.
    event_mask = event == 1
    event_idx = np.where(event_mask)[0]
    order = np.argsort(time[event_idx], kind="stable")
    event_idx_sorted = event_idx[order]

    d = len(event_idx_sorted)
    event_times_out = time[event_idx_sorted]
    residuals = np.zeros((d, p))
    info_diag = np.zeros(p)

    for k, idx in enumerate(event_idx_sorted):
        t_k = time[idx]

        # Risk set: all subjects with observed time >= t_k.
        risk_mask = time >= t_k
        eta_risk = eta[risk_mask]

        # Numerically stable weights: subtract max before exp.
        eta_shift = eta_risk - np.max(eta_risk)
        w = np.exp(eta_shift)
        w_sum = w.sum()
        if w_sum == 0:
            continue

        x_risk = covariates[risk_mask]

        # Weighted mean of covariates in the risk set.
        x_bar = (w[:, None] * x_risk).sum(axis=0) / w_sum

        # Weighted variance (diagonal of V(t_k)).
        diff = x_risk - x_bar
        var_x = (w[:, None] * diff ** 2).sum(axis=0) / w_sum
        info_diag += var_x

        # Schoenfeld residual.
        residuals[k] = covariates[idx] - x_bar

    return event_times_out, residuals, info_diag


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def schoenfeld_residuals(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    coef: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Schoenfeld residuals for a fitted Cox model.

    Parameters
    ----------
    time
        Observed durations (n,).
    event
        Event indicator, 1 = event, 0 = censored (n,).
    covariates
        Covariate matrix (n, p) or vector (n,) for a single covariate.
    coef
        Fitted Cox regression coefficients (p,).

    Returns
    -------
    event_times : ndarray, shape (d,)
        Sorted uncensored event times (d = number of events).
    residuals : ndarray, shape (d, p)
        Schoenfeld residual at each event time for each covariate.
    """
    event_times, residuals, _ = _compute_residuals_and_info(
        time, event, covariates, coef
    )
    return event_times, residuals


def scaled_schoenfeld_residuals(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    coef: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute scaled Schoenfeld residuals (Grambsch & Therneau 1994).

    The scaled residuals estimate the time-varying coefficient beta(t).
    Under PH they should be constant over time; a trend indicates
    violation.

    Parameters
    ----------
    time, event, covariates, coef
        As in :func:`schoenfeld_residuals`.

    Returns
    -------
    event_times : ndarray, shape (d,)
        Sorted uncensored event times.
    residuals : ndarray, shape (d, p)
        Raw Schoenfeld residuals.
    scaled : ndarray, shape (d, p)
        Scaled Schoenfeld residuals:
        r*_j(t_k) = beta_j + d * r_j(t_k) / I_jj.
    """
    event_times, residuals, info_diag = _compute_residuals_and_info(
        time, event, covariates, coef
    )
    coef = np.asarray(coef, dtype=float).ravel()
    d = len(event_times)
    p = residuals.shape[1]

    scaled = np.zeros_like(residuals)
    for j in range(p):
        if info_diag[j] > 0:
            scaled[:, j] = coef[j] + d * residuals[:, j] / info_diag[j]
        else:
            scaled[:, j] = coef[j]

    return event_times, residuals, scaled


def ph_test(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
    coef: np.ndarray,
    time_transform: str = "rank",
    covariate_names: list[str] | None = None,
) -> PHTestResult:
    """Test the proportional hazards assumption.

    Correlates the scaled Schoenfeld residuals for each covariate with
    a transformation of event time.  Under PH the correlation should be
    zero; a significant result indicates that the covariate's effect
    changes over time.

    Parameters
    ----------
    time
        Observed durations (n,).
    event
        Event indicator, 1 = event, 0 = censored (n,).
    covariates
        Covariate matrix (n, p) or vector (n,) for a single covariate.
    coef
        Fitted Cox model coefficients (p,).
    time_transform
        Transformation applied to event times before testing
        correlation.  One of ``'identity'``, ``'log'``, or ``'rank'``
        (default).
    covariate_names
        Optional labels for each covariate.  If *None*, defaults to
        ``['x0', 'x1', ...]``.

    Returns
    -------
    PHTestResult
        Per-covariate and global test statistics, plus the residual
        arrays for further inspection or plotting.
    """
    event_times, residuals, scaled = scaled_schoenfeld_residuals(
        time, event, covariates, coef
    )

    d, p = scaled.shape

    if covariate_names is None:
        covariate_names = [f"x{j}" for j in range(p)]
    if len(covariate_names) != p:
        raise ValueError(
            f"covariate_names has {len(covariate_names)} entries "
            f"but there are {p} covariates"
        )

    # Transform event times.
    g = _apply_time_transform(event_times, time_transform)

    covariate_results: list[CovariateTestResult] = []
    global_chi_sq = 0.0
    n_testable = 0

    for j in range(p):
        # Need at least 3 events and non-constant arrays to
        # compute a meaningful Pearson correlation.
        if d < 3 or np.std(g) < 1e-15 or np.std(scaled[:, j]) < 1e-15:
            covariate_results.append(
                CovariateTestResult(
                    name=covariate_names[j],
                    correlation=np.nan,
                    chi_sq=np.nan,
                    df=1,
                    p_value=np.nan,
                )
            )
            continue

        rho, _ = stats.pearsonr(g, scaled[:, j])
        chi_sq_j = rho ** 2 * d
        p_val = float(1.0 - stats.chi2.cdf(chi_sq_j, df=1))

        covariate_results.append(
            CovariateTestResult(
                name=covariate_names[j],
                correlation=float(rho),
                chi_sq=float(chi_sq_j),
                df=1,
                p_value=p_val,
            )
        )
        global_chi_sq += chi_sq_j
        n_testable += 1

    # Global test.
    if n_testable > 0:
        global_df = n_testable
        global_p_value = float(
            1.0 - stats.chi2.cdf(global_chi_sq, df=global_df)
        )
    else:
        global_df = p
        global_p_value = np.nan

    return PHTestResult(
        covariate_results=covariate_results,
        global_chi_sq=float(global_chi_sq),
        global_df=global_df,
        global_p_value=global_p_value,
        event_times=event_times,
        schoenfeld_residuals=residuals,
        scaled_schoenfeld_residuals=scaled,
        time_transform=time_transform,
    )
