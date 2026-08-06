"""Tests that fail if any transform touches data outside the training fold.

Three independent checks, because a single assertion is easy to satisfy
accidentally:

  1. STRUCTURAL  -- the pipeline refuses to score rows it was fitted on.
  2. CANARY      -- corrupting the held-out rows to absurd values must not
                    move a single training-fold constant. If any statistic
                    were pooled across the split, this fails loudly.
  3. AUDIT       -- a real leave-one-fish-out run leaves an audit trail
                    with no fit/predict overlap anywhere in it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from pte import config as C
from pte import cv, data, pipeline
from pte.pipeline import FoldPipeline, LeakageError


@pytest.fixture(scope="module")
def df():
    return data.load()


# --------------------------------------------------------------------
# 1. Structural: the guard exists and bites
# --------------------------------------------------------------------
def test_pipeline_refuses_to_score_its_own_training_rows(df):
    idx = list(df.index)
    train, held = idx[:-1], idx[-1:]
    pipe = FoldPipeline(features=C.M1_FEATURES, tag="test").fit(df, train)

    pipe.predict(df, held)                       # legitimate: disjoint

    with pytest.raises(LeakageError):
        pipe.predict(df, [train[0]])             # its own training row

    with pytest.raises(LeakageError):
        pipe.predict(df, held + [train[5]])      # partial overlap


def test_overlap_is_allowed_only_when_explicitly_requested(df):
    idx = list(df.index)
    pipe = FoldPipeline(features=C.M1_FEATURES, tag="test").fit(df, idx)
    # The final full-data model is in-sample by definition and must say so.
    pipe.predict(df, idx, allow_overlap=True)
    with pytest.raises(LeakageError):
        pipe.predict(df, idx)


# --------------------------------------------------------------------
# 2. Canary: held-out data cannot influence anything learned
# --------------------------------------------------------------------
def test_corrupting_held_out_rows_changes_nothing_learned(df):
    """The decisive test.

    If ANY scaling constant, hyperparameter, or selected feature were
    computed on the full table rather than the training fold, replacing
    the held-out larvae's covariates with absurd values would move it.
    """
    idx = list(df.index)
    held = idx[:10]
    train = idx[10:]

    clean = FoldPipeline(features=C.M1_FEATURES, seed=C.SEED,
                         tag="canary").fit(df, train).fingerprint()

    poisoned = df.copy()
    cols = C.M1_FEATURES
    rng = np.random.default_rng(0)
    poisoned.loc[held, cols] = rng.normal(1e6, 1e5, size=(len(held), len(cols)))
    poisoned.loc[held, C.DURATION_COL] = 1e7
    poisoned.loc[held, C.EVENT_COL] = 1

    after = FoldPipeline(features=C.M1_FEATURES, seed=C.SEED,
                         tag="canary").fit(poisoned, train).fingerprint()

    assert clean["mean"] == after["mean"], "scaler mean saw held-out rows"
    assert clean["scale"] == after["scale"], "scaler scale saw held-out rows"
    assert clean["penalizer"] == after["penalizer"], \
        "hyperparameter tuning saw held-out rows"
    assert clean["l1_ratio"] == after["l1_ratio"], \
        "hyperparameter tuning saw held-out rows"
    assert clean["selected"] == after["selected"], \
        "feature selection saw held-out rows"
    assert clean["beta"] == after["beta"], \
        "coefficient estimation saw held-out rows"


def test_canary_would_actually_fire(df):
    """Guard against a vacuous canary.

    If the same corruption is applied to a TRAINING row instead, the
    fingerprint must change -- otherwise test_corrupting_held_out_rows
    proves nothing.
    """
    idx = list(df.index)
    held, train = idx[:10], idx[10:]

    clean = FoldPipeline(features=C.M1_FEATURES, seed=C.SEED,
                         tag="canary").fit(df, train).fingerprint()

    poisoned = df.copy()
    rng = np.random.default_rng(0)
    poisoned.loc[train[:10], C.M1_FEATURES] = rng.normal(
        1e6, 1e5, size=(10, len(C.M1_FEATURES)))

    after = FoldPipeline(features=C.M1_FEATURES, seed=C.SEED,
                         tag="canary").fit(poisoned, train).fingerprint()

    assert clean["mean"] != after["mean"], (
        "corrupting TRAINING rows did not change the scaler -- the canary "
        "test is vacuous"
    )


def test_scaling_uses_training_fold_moments_only(df):
    idx = list(df.index)
    train = idx[:80]
    pipe = FoldPipeline(features=C.M1_FEATURES, tag="test").fit(df, train)

    expected_mean = df.loc[train, C.M1_FEATURES].to_numpy(float).mean(axis=0)
    expected_sd = df.loc[train, C.M1_FEATURES].to_numpy(float).std(axis=0,
                                                                   ddof=0)
    np.testing.assert_allclose(pipe.mean_, expected_mean, rtol=1e-12)
    np.testing.assert_allclose(pipe.scale_, expected_sd, rtol=1e-12)

    full_mean = df[C.M1_FEATURES].to_numpy(float).mean(axis=0)
    assert not np.allclose(pipe.mean_, full_mean), (
        "training-fold mean equals the full-data mean; scaling is pooled"
    )


# --------------------------------------------------------------------
# 3. Audit: a real CV run leaves a clean trail
# --------------------------------------------------------------------
def test_leave_one_fish_out_audit_trail_is_clean(df):
    pipeline.reset_audit()
    small = df.iloc[:30].reset_index(drop=True)
    cv.leave_one_fish_out(small, C.M1_FEATURES,
                          penalizer_grid=[0.1], l1_grid=[0.5],
                          tag="audit", compute_medians=False)

    assert len(pipeline.AUDIT_LOG) == len(small), \
        "one fit/predict pair per larva expected"
    assert pipeline.audit_violations() == [], "leakage detected in the CV run"

    for rec in pipeline.AUDIT_LOG:
        assert len(rec.predict_index) == 1
        assert len(rec.fit_index) == len(small) - 1
        assert not (set(rec.fit_index) & set(rec.predict_index))


def test_permuted_outcomes_collapse_performance(df):
    """A pipeline that leaks would still score well on shuffled outcomes."""
    small = df.iloc[:60].reset_index(drop=True)
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(small))
    shuffled = small.copy()
    shuffled[C.DURATION_COL] = small[C.DURATION_COL].to_numpy()[perm]
    shuffled[C.EVENT_COL] = small[C.EVENT_COL].to_numpy()[perm]

    res = cv.leave_one_fish_out(shuffled, C.M1_FEATURES,
                                penalizer_grid=[0.1], l1_grid=[0.5],
                                tag="shuffled", compute_medians=False)
    assert 0.25 < res.c_index < 0.75, (
        f"concordance {res.c_index:.3f} on shuffled outcomes is not near "
        f"chance; the pipeline is recovering signal that should not exist"
    )
