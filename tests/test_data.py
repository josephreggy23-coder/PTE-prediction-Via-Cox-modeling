"""Survival-data invariants.

The failure mode these guard against is the one that quietly turns a
survival analysis into a misspecified classification: a censored larva
being dropped, or being scored as though it had been observed not to
seize.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from pte import config as C
from pte import cv, data
from pte.data import DataValidationError


@pytest.fixture(scope="module")
def df():
    return data.load()


def test_one_row_per_larva(df):
    assert len(df) == 100
    assert df[C.SUBJECT_COL].is_unique


def test_no_censored_row_is_dropped(df):
    n_censored = int((df[C.EVENT_COL] == 0).sum())
    assert n_censored > 0, "fixture has no censored larvae to protect"

    res = cv.leave_one_fish_out(df, C.M0_FEATURES,
                                penalizer_grid=[0.1], l1_grid=[0.5],
                                tag="census", compute_medians=False)
    assert len(res.risk) == len(df), "a larva vanished during CV"
    assert int((res.event == 0).sum()) == n_censored, \
        "censored larvae were lost between the table and the CV output"


def test_censored_larvae_are_not_a_negative_class(df):
    """A censored larva must contribute as a risk-set member, not a label.

    Operationally: recoding censored rows as observed non-events changes
    the fitted model. If it did not, censoring would be being ignored.
    """
    from pte import coxnet

    X = df[C.M1_FEATURES].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    t = df[C.DURATION_COL].to_numpy(float)
    e = df[C.EVENT_COL].to_numpy(float)

    proper = coxnet.fit(X, t, e, C.M1_FEATURES, 0.03, 0.5)
    as_negative = coxnet.fit(X, t, np.ones_like(e), C.M1_FEATURES, 0.03, 0.5)

    assert not np.allclose(proper.beta, as_negative.beta, atol=1e-6), (
        "treating censored larvae as observed events leaves the fit "
        "unchanged, which means censoring is not entering the likelihood"
    )


def test_censoring_is_administrative(df):
    cens = df[df[C.EVENT_COL] == 0]
    assert cens[C.DURATION_COL].nunique() == 1
    np.testing.assert_allclose(cens[C.DURATION_COL].to_numpy(),
                               cens[C.CENSOR_TIME_COL].to_numpy())


def test_no_event_reaches_the_horizon(df):
    ev = df[df[C.EVENT_COL] == 1]
    assert (ev[C.DURATION_COL] < ev[C.CENSOR_TIME_COL]).all()


def test_qc_columns_are_never_predictors():
    for c in C.QC_COLUMNS:
        assert c not in C.M0_FEATURES
        assert c not in C.M1_FEATURES


def test_condition_is_never_a_predictor():
    assert C.LABEL_COL not in C.M1_FEATURES
    assert C.STRATUM_COL not in C.M1_FEATURES


def test_behavioural_features_stay_out_of_survival_models():
    for f in C.BEH_FEATURES:
        assert f not in C.M1_FEATURES


def test_validation_rejects_dropped_censoring(df):
    bad = df.copy()
    bad.loc[bad[C.EVENT_COL] == 0, C.DURATION_COL] = 900.0
    with pytest.raises(DataValidationError):
        data.validate(bad)


def test_validation_rejects_duplicate_larvae(df):
    bad = df.copy()
    bad.loc[bad.index[0], C.SUBJECT_COL] = bad.loc[bad.index[1], C.SUBJECT_COL]
    with pytest.raises(DataValidationError):
        data.validate(bad)


def test_events_per_predictor_is_computed_and_flagged(df):
    s = data.design_summary(df)
    assert s["events_per_predictor_M1"] == pytest.approx(
        s["n_events"] / s["n_predictors_M1"], abs=1e-3)   # reported to 3 dp
    if s["events_per_predictor_M1"] < C.EPV_THRESHOLD:
        assert s["epv_M1_below_threshold"]
        assert "WARNING" in s["epv_flag"]
