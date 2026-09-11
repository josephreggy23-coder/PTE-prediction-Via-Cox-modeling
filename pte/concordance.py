"""Harrell's concordance index for survival model discrimination.

Concordance (the C-index) answers: given two randomly chosen subjects,
how often does the model assign a higher risk to the one who experienced
the event first?  It is the survival-analysis analogue of the area under
the ROC curve.

Mathematical definition
-----------------------
A pair of subjects (i, j) is *concordant* if the subject with the
shorter observed event time has the higher predicted risk score.
Formally, for all *usable* pairs where the ordering of event times
is unambiguous:

    C = (# concordant pairs + 0.5 * # tied predictions) / (# usable pairs)

A pair is usable if both experienced the event, or the one with the
shorter time experienced the event (so we know the ordering).  A pair
is *not* usable if the shorter time is censored, because we cannot
tell which subject truly had the earlier event.

Specifically, for subjects i and j with observed times T_i < T_j:

- If delta_i = 1 (subject i had the event):
  The pair is usable.  It is concordant if risk_i > risk_j,
  discordant if risk_i < risk_j, and tied if risk_i = risk_j.

- If delta_i = 0 (subject i was censored):
  The pair is not usable (we don't know whether i's event would
  have preceded j's).

Interpretation: C = 0.5 is random, C = 1.0 is perfect discrimination,
C < 0.5 means the model ranks risk backwards.

Connection to Kendall's tau
----------------------------
The C-index is related to the Somers' D statistic and Kendall's tau:

    D = 2 * (C - 0.5)
    tau_a = D * (usable pairs / total pairs)

Somers' D ranges from -1 to +1 and can be interpreted as the excess
concordance probability minus discordance probability.

References
----------
Harrell FE, Califf RM, Pryor DB, Lee KL, Rosati RA. Evaluating the
yield of medical tests. JAMA, 1982; 247(18):2543-2546.

Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in
developing models, evaluating assumptions and adequacy, and measuring
and reducing errors. Statistics in Medicine, 1996; 15(4):361-387.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConcordanceResult:
    """Result of a concordance index computation.

    Attributes
    ----------
    c_index
        Concordance probability in [0, 1].
    concordant
        Number of concordant pairs.
    discordant
        Number of discordant pairs.
    tied_risk
        Number of pairs with equal predicted risk but different event times.
    usable
        Total usable pairs (concordant + discordant + tied_risk).
    somers_d
        Somers' D = 2 * (C - 0.5), ranging from -1 to +1.
    """

    c_index: float
    concordant: int
    discordant: int
    tied_risk: int
    usable: int

    @property
    def somers_d(self) -> float:
        """Somers' D: the concordance-discordance difference, normalized."""
        return 2.0 * (self.c_index - 0.5)


def concordance_index(
    time: np.ndarray,
    event: np.ndarray,
    risk_score: np.ndarray,
) -> ConcordanceResult:
    """Compute Harrell's concordance index.

    Parameters
    ----------
    time
        Observed durations (time to event or censoring).
    event
        Event indicator (1 = event observed, 0 = censored).
    risk_score
        Model-predicted risk scores.  Higher values should correspond
        to shorter survival times (higher hazard).  For a Cox model
        this is the linear predictor ``X @ beta``.

    Returns
    -------
    ConcordanceResult
        Contains the C-index and pair-level counts.

    Notes
    -----
    The naive pairwise comparison is O(n^2), which is acceptable for
    typical clinical cohorts (n < 10,000).  For very large datasets,
    an O(n log n) algorithm based on sorting and a modified merge
    count could replace this.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk_score = np.asarray(risk_score, dtype=float)

    n = len(time)
    if n != len(event) or n != len(risk_score):
        raise ValueError(
            "time, event, and risk_score must have the same length"
        )

    # Drop missing values.
    ok = np.isfinite(time) & np.isfinite(event) & np.isfinite(risk_score)
    time = time[ok]
    event = event[ok]
    risk_score = risk_score[ok]
    n = len(time)

    if n < 2:
        return ConcordanceResult(np.nan, 0, 0, 0, 0)

    concordant = 0
    discordant = 0
    tied_risk = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Determine which has the shorter time.
            if time[i] < time[j]:
                shorter, longer = i, j
            elif time[j] < time[i]:
                shorter, longer = j, i
            else:
                # Tied times: usable only if both are events.
                if event[i] == 1 and event[j] == 1:
                    # With tied times and both events, only tied risk
                    # contributes (neither can be concordant).
                    if risk_score[i] == risk_score[j]:
                        tied_risk += 1
                continue

            # The pair is usable only if the shorter-time subject had
            # an event (otherwise we don't know the true ordering).
            if event[shorter] != 1:
                continue

            # Compare risk scores.
            if risk_score[shorter] > risk_score[longer]:
                concordant += 1
            elif risk_score[shorter] < risk_score[longer]:
                discordant += 1
            else:
                tied_risk += 1

    usable = concordant + discordant + tied_risk
    if usable == 0:
        return ConcordanceResult(np.nan, 0, 0, 0, 0)

    c_index = (concordant + 0.5 * tied_risk) / usable
    return ConcordanceResult(
        c_index=float(c_index),
        concordant=concordant,
        discordant=discordant,
        tied_risk=tied_risk,
        usable=usable,
    )


def concordance_bootstrap_ci(
    time: np.ndarray,
    event: np.ndarray,
    risk_score: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap confidence interval for the concordance index.

    Parameters
    ----------
    time, event, risk_score
        As in ``concordance_index``.
    n_bootstrap
        Number of bootstrap resamples.
    alpha
        Significance level (e.g., 0.05 for a 95% CI).
    seed
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``c_index``, ``ci_low``, ``ci_high``, ``se``,
        ``n_bootstrap_ok``.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk_score = np.asarray(risk_score, dtype=float)

    point = concordance_index(time, event, risk_score).c_index

    rng = np.random.default_rng(seed)
    n = len(time)
    c_values = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        result = concordance_index(time[idx], event[idx], risk_score[idx])
        if np.isfinite(result.c_index):
            c_values.append(result.c_index)

    c_values = np.array(c_values)
    return {
        "c_index": float(point),
        "ci_low": float(np.percentile(c_values, 100 * alpha / 2))
        if len(c_values) > 0
        else np.nan,
        "ci_high": float(np.percentile(c_values, 100 * (1 - alpha / 2)))
        if len(c_values) > 0
        else np.nan,
        "se": float(np.std(c_values)) if len(c_values) > 0 else np.nan,
        "n_bootstrap_ok": len(c_values),
    }
