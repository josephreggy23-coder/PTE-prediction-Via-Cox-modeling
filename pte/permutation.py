"""Permutation null for the cross-validated concordance index.

The outcome pair (time, event) is shuffled across larvae, keeping the
censoring pattern attached to the time it belongs to, and then the
ENTIRE pipeline is refitted: fold-local scaling, inner-loop
hyperparameter tuning, elastic-net selection, and leave-one-fish-out
prediction. Nothing is cached from the observed fit.

This is the expensive part of the analysis. It is why `coxnet` exists.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config as C
from . import cv

_WORKER = {}


def _init(df_records, columns, features, use_strata):
    _WORKER["df"] = pd.DataFrame(df_records, columns=columns)
    _WORKER["features"] = features
    _WORKER["use_strata"] = use_strata


def _one(seed: int) -> float:
    df = _WORKER["df"].copy()
    rng = np.random.default_rng(seed)

    # Shuffle the (time, event) pair as a unit: a censored larva stays
    # censored, it simply gets attached to a different set of covariates.
    perm = rng.permutation(len(df))
    df[C.DURATION_COL] = df[C.DURATION_COL].to_numpy()[perm]
    df[C.EVENT_COL] = df[C.EVENT_COL].to_numpy()[perm]

    res = cv.leave_one_fish_out(
        df, _WORKER["features"], use_clutch_strata=_WORKER["use_strata"],
        seed=C.SEED, tag="perm", compute_medians=False,
    )
    return res.c_index


def run(df: pd.DataFrame, features: list, observed_c: float,
        n_iter: int = C.N_PERMUTATIONS, seed: int = C.PERMUTATION_SEED,
        use_clutch_strata: bool = True, n_workers: int | None = None,
        label: str = "M1") -> dict:
    seeds = list(np.random.default_rng(seed).integers(0, 2**31 - 1, n_iter))
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)

    args = (df.to_dict("records"), list(df.columns), list(features),
            use_clutch_strata)

    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init, initargs=args) as ex:
        null = list(ex.map(_one, seeds, chunksize=4))

    null = np.asarray(null, dtype=float)
    # Add-one (Phipson & Smyth): a permutation p-value is never exactly 0.
    p = float((1 + np.sum(null >= observed_c)) / (1 + len(null)))

    return {
        "label": label,
        "n_permutations": int(len(null)),
        "observed_c_index": float(observed_c),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "null_median": float(np.median(null)),
        "null_q95": float(np.percentile(null, 95)),
        "null_q99": float(np.percentile(null, 99)),
        "null_max": float(null.max()),
        "p_value": p,
        "p_value_method": "add-one permutation p (Phipson & Smyth 2010)",
        "z_vs_null": float((observed_c - null.mean()) / null.std(ddof=1)),
        "exceeds_null_95th": bool(observed_c > np.percentile(null, 95)),
        "seed": seed,
        "refit_end_to_end": True,
        "null_distribution": null.tolist(),
    }
