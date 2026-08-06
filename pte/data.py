"""Loading and validation.

The invariants enforced here are the ones that, if silently violated,
would turn a survival analysis into a misspecified classification:
censored larvae must survive the pipeline intact and must never be
recoded as a negative class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


class DataValidationError(RuntimeError):
    pass


def load() -> pd.DataFrame:
    """Read the single-sheet table and validate it before returning."""
    df = pd.read_excel(C.DATA_FILE, sheet_name=C.DATA_SHEET)
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    n = len(df)

    if df[C.SUBJECT_COL].nunique() != n:
        raise DataValidationError(
            f"{C.SUBJECT_COL} is not unique: {df[C.SUBJECT_COL].nunique()} ids "
            f"for {n} rows. One row per larva is assumed everywhere."
        )

    needed = (
        [C.DURATION_COL, C.EVENT_COL, C.CENSOR_TIME_COL, C.CLUTCH_COL,
         C.STRATUM_COL, C.LABEL_COL, C.PRESSURE]
        + C.LFP_FEATURES + C.BEH_FEATURES + C.QC_COLUMNS
    )
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")

    model_cols = [C.PRESSURE] + C.LFP_FEATURES + C.BEH_FEATURES
    nulls = df[model_cols + [C.DURATION_COL, C.EVENT_COL]].isna().sum()
    if nulls.any():
        raise DataValidationError(f"nulls present:\n{nulls[nulls > 0]}")

    if not set(df[C.EVENT_COL].unique()) <= {0, 1}:
        raise DataValidationError("event must be coded 0/1")

    if (df[C.DURATION_COL] <= 0).any():
        raise DataValidationError("non-positive follow-up time")

    # A censored row carries the censoring time as its recorded duration.
    cens = df[df[C.EVENT_COL] == 0]
    if len(cens) and not np.allclose(
        cens[C.DURATION_COL].to_numpy(), cens[C.CENSOR_TIME_COL].to_numpy()
    ):
        raise DataValidationError(
            "censored rows do not carry censor_time_s as their duration"
        )

    # Events must not sit at or beyond the censoring horizon.
    ev = df[df[C.EVENT_COL] == 1]
    if (ev[C.DURATION_COL] >= ev[C.CENSOR_TIME_COL]).any():
        raise DataValidationError("an event time reaches the censoring horizon")


def design_summary(df: pd.DataFrame) -> dict:
    """Facts about the cohort that belong in the results file."""
    n = len(df)
    n_events = int(df[C.EVENT_COL].sum())
    n_censored = n - n_events
    p_m1 = len(C.M1_FEATURES)

    by_stratum = (
        df.groupby(C.STRATUM_COL)[C.EVENT_COL]
        .agg(n="count", events="sum")
        .assign(censored=lambda d: d.n - d.events)
    )
    by_clutch = (
        df.groupby(C.CLUTCH_COL)[C.EVENT_COL]
        .agg(n="count", events="sum")
        .assign(censored=lambda d: d.n - d.events)
    )

    epv_m1 = n_events / p_m1
    epv_m0 = n_events / len(C.M0_FEATURES)

    # Strata in which max_pressure_kPa has no within-stratum variance.
    degenerate = [
        int(h)
        for h, g in df.groupby(C.STRATUM_COL)
        if float(g[C.PRESSURE].std(ddof=0)) < 1e-12
    ]

    return {
        "n_larvae": n,
        "n_events": n_events,
        "n_censored": n_censored,
        "censoring_is_administrative": bool(
            df.loc[df[C.EVENT_COL] == 0, C.DURATION_COL].nunique() <= 1
        ),
        "censor_horizon_s": float(df[C.CENSOR_TIME_COL].iloc[0]),
        "n_clutches": int(df[C.CLUTCH_COL].nunique()),
        "n_injury_strata": int(df[C.STRATUM_COL].nunique()),
        "n_tied_event_times": int(
            df.loc[df[C.EVENT_COL] == 1, C.DURATION_COL].duplicated().sum()
        ),
        "n_predictors_M0": len(C.M0_FEATURES),
        "n_predictors_M1": p_m1,
        "events_per_predictor_M0": round(epv_m0, 3),
        "events_per_predictor_M1": round(epv_m1, 3),
        "epv_threshold": C.EPV_THRESHOLD,
        "epv_M1_below_threshold": bool(epv_m1 < C.EPV_THRESHOLD),
        "epv_flag": (
            f"WARNING: M1 events-per-predictor = {epv_m1:.2f} "
            f"(< {C.EPV_THRESHOLD:g}). Coefficient estimates are "
            f"variance-limited; treat individual hazard ratios as "
            f"directional evidence, not precise effect sizes."
            if epv_m1 < C.EPV_THRESHOLD
            else f"OK: M1 events-per-predictor = {epv_m1:.2f}"
        ),
        "by_injury_stratum": by_stratum.reset_index().to_dict("records"),
        "by_clutch": by_clutch.reset_index().to_dict("records"),
        "strata_with_constant_pressure": degenerate,
        "constant_columns_dropped": C.CONSTANT_COLUMNS,
        "qc_columns_excluded_from_predictors": C.QC_COLUMNS,
    }


def qc_summary(df: pd.DataFrame) -> dict:
    """QC channels are reported, never modelled."""
    out = {}
    for c in C.QC_COLUMNS:
        out[c] = {
            "min": float(df[c].min()),
            "median": float(df[c].median()),
            "max": float(df[c].max()),
            "mean": float(df[c].mean()),
        }
    return out
