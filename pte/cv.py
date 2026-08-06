"""Leave-one-fish-out cross-validation and its summary statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from . import config as C
from .pipeline import FoldPipeline


@dataclass
class CVResult:
    features: list
    risk: np.ndarray            # held-out linear predictor, one per larva
    pred_median_s: np.ndarray   # held-out predicted median latency, seconds
    time: np.ndarray
    event: np.ndarray
    index: np.ndarray
    chosen_penalizer: np.ndarray
    chosen_l1_ratio: np.ndarray
    selection_counts: dict

    @property
    def c_index(self) -> float:
        # Higher linear predictor = higher hazard = shorter latency, so the
        # risk score is negated to align with concordance on survival time.
        return float(concordance_index(self.time, -self.risk, self.event))


def leave_one_fish_out(df: pd.DataFrame, features: list,
                       use_clutch_strata: bool = True,
                       penalizer_grid=None, l1_grid=None,
                       seed: int = C.SEED, tag: str = "loo",
                       compute_medians: bool = True) -> CVResult:
    """One fold per larva. Nothing is fitted on the held-out larva."""
    idx = np.asarray(df.index)
    n = len(idx)

    risk = np.empty(n)
    med = np.full(n, np.nan)
    pens = np.empty(n)
    l1s = np.empty(n)
    sel_counts = {f: 0 for f in features}

    for i, held in enumerate(idx):
        train = [j for j in idx if j != held]
        pipe = FoldPipeline(
            features=features,
            penalizer_grid=list(penalizer_grid or C.PENALIZER_GRID),
            l1_grid=list(l1_grid or C.L1_RATIO_GRID),
            use_clutch_strata=use_clutch_strata,
            seed=seed, tag=tag,
        ).fit(df, train)

        risk[i] = pipe.predict(df, [held])[0]
        if compute_medians:
            med[i] = pipe.predict_median_time(df, [held])[0]
        pens[i] = pipe.best_penalizer_
        l1s[i] = pipe.best_l1_ratio_
        for f in pipe.model_.selected:
            sel_counts[f] += 1

    return CVResult(
        features=list(features), risk=risk, pred_median_s=med,
        time=df[C.DURATION_COL].to_numpy(float),
        event=df[C.EVENT_COL].to_numpy(float),
        index=idx, chosen_penalizer=pens, chosen_l1_ratio=l1s,
        selection_counts=sel_counts,
    )


# --------------------------------------------------------------------
# Interval estimation
# --------------------------------------------------------------------
def c_index_ci(time, risk, event, n_boot: int = C.N_BOOTSTRAP,
               seed: int = C.BOOTSTRAP_SEED, alpha: float = C.ALPHA,
               clusters=None):
    """Percentile bootstrap CI for Harrell's C.

    If `clusters` is given (clutch ids), the bootstrap resamples whole
    clutches rather than individual larvae, which is the substitute for
    the frailty term's variance correction.
    """
    rng = np.random.default_rng(seed)
    time = np.asarray(time, float)
    risk = np.asarray(risk, float)
    event = np.asarray(event, float)
    n = len(time)
    point = float(concordance_index(time, -risk, event))

    boots = []
    if clusters is not None:
        clusters = np.asarray(clusters)
        groups = [np.flatnonzero(clusters == g) for g in np.unique(clusters)]

    for _ in range(n_boot):
        if clusters is None:
            s = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(groups), len(groups))
            s = np.concatenate([groups[k] for k in pick])
        if event[s].sum() < 2:
            continue
        try:
            boots.append(concordance_index(time[s], -risk[s], event[s]))
        except ZeroDivisionError:
            continue

    if len(boots) < 50:
        return point, float("nan"), float("nan"), len(boots)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi), len(boots)
