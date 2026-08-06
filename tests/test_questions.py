"""Tests for the reporting logic of the eight biological questions.

These guard the interpretive layer rather than the arithmetic. Both
bugs they pin down were found in real output: a shrunk-to-nothing
coefficient being reported as pointing "as expected", and an inverted
R-squared interval for a negative correlation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from pte import config as C
from pte import data
from pte import questions as Q


@pytest.fixture(scope="module")
def df():
    return data.load()


def _hr_table(coefs: dict):
    """Minimal elastic-net HR table for the direction checker."""
    rows = []
    for f, c in coefs.items():
        rows.append({
            "feature": f,
            "coef_log_hr_per_sd": c,
            "hazard_ratio_per_sd": float(np.exp(c)),
            "hr_ci_low": float(np.exp(c - 0.2)),
            "hr_ci_high": float(np.exp(c + 0.2)),
            "selected": abs(c) >= C.NEGLIGIBLE_LOG_HR,
        })
    return pd.DataFrame(rows)


def _unpen_table(hrs: dict):
    return pd.DataFrame([
        {"feature": f, "hazard_ratio_per_sd": h,
         "hr_ci_low": h * 0.8, "hr_ci_high": h * 1.2, "p": 0.5}
        for f, h in hrs.items()
    ])


# --------------------------------------------------------------------
# Question 2: a deleted coefficient has no direction
# --------------------------------------------------------------------
def test_negligible_coefficients_are_not_counted_as_agreeing():
    """The bug this pins: HR = 1.000004 is not evidence of correct biology."""
    hr = _hr_table({
        "lfp_ap_slope": 1e-6,
        "lfp_variance_uV2": -2e-6,
        "lfp_autocorr_lag1": 3e-6,
        "lfp_discharge_rate_per_min": 1e-7,
    })
    unpen = _unpen_table({
        "lfp_ap_slope": 0.46, "lfp_variance_uV2": 0.78,
        "lfp_autocorr_lag1": 0.98, "lfp_discharge_rate_per_min": 1.04,
    })
    out = Q.q2_direction_check(hr, unpen)

    assert out["n_retained"] == 0
    assert out["n_shrunk_to_negligible"] == 4
    assert "no direction to check" in out["answer"]
    assert "as expected" not in out["answer"].split("unpenalized")[0]
    # The unpenalized fallback must still report the real inversions.
    assert set(out["inverted_features_unpenalized"]) == {
        "lfp_variance_uV2", "lfp_autocorr_lag1"}


def test_genuine_inversion_is_flagged():
    hr = _hr_table({
        "lfp_ap_slope": -0.8,          # expected negative -> correct
        "lfp_variance_uV2": -0.5,      # expected positive -> INVERTED
        "lfp_autocorr_lag1": 0.4,      # expected positive -> correct
        "lfp_discharge_rate_per_min": 0.3,
    })
    unpen = _unpen_table({
        "lfp_ap_slope": 0.45, "lfp_variance_uV2": 0.60,
        "lfp_autocorr_lag1": 1.5, "lfp_discharge_rate_per_min": 1.3,
    })
    out = Q.q2_direction_check(hr, unpen)

    assert out["n_retained"] == 4
    assert out["inverted_features"] == ["lfp_variance_uV2"]
    assert "WRONG way" in out["answer"]


def test_all_correct_directions_report_no_inversion():
    hr = _hr_table({
        "lfp_ap_slope": -0.8, "lfp_variance_uV2": 0.5,
        "lfp_autocorr_lag1": 0.4, "lfp_discharge_rate_per_min": 0.3,
    })
    unpen = _unpen_table({
        "lfp_ap_slope": 0.45, "lfp_variance_uV2": 1.6,
        "lfp_autocorr_lag1": 1.5, "lfp_discharge_rate_per_min": 1.3,
    })
    out = Q.q2_direction_check(hr, unpen)
    assert out["n_inverted"] == 0
    assert "no inversions" in out["answer"]


# --------------------------------------------------------------------
# Question 3: agreement is only meaningful if both survive
# --------------------------------------------------------------------
def test_critical_slowing_pair_shrunk_out_is_not_agreement(df):
    hr = _hr_table({"lfp_variance_uV2": 1e-6, "lfp_autocorr_lag1": -1e-6})
    unpen = _unpen_table({"lfp_variance_uV2": 0.78, "lfp_autocorr_lag1": 0.98})
    out = Q.q3_critical_slowing_agreement(df, hr, unpen)

    assert out["both_shrunk_to_negligible"]
    assert not out["signs_agree"]
    assert "cannot say whether they agree" in out["answer"]
    # Falls through to the unpenalized fit, where both invert together.
    assert out["unpenalized_signs_agree"]
    assert not out["unpenalized_both_match_expected"]


# --------------------------------------------------------------------
# Question 6: R-squared intervals must not invert
# --------------------------------------------------------------------
def test_r2_interval_is_ordered_even_for_negative_correlations(df):
    out = Q.q6_behaviour_vs_dose(df)
    for row in out["table"]:
        lo, hi = row["r2_ci"]
        assert lo <= hi, (
            f"{row['feature']}: R^2 interval [{lo:.3f}, {hi:.3f}] is inverted"
        )
        assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
        assert lo <= row["r2_on_pressure"] <= hi, (
            f"{row['feature']}: point estimate outside its own interval"
        )


def test_negative_correlation_is_present_to_make_that_test_meaningful(df):
    """Guard against a vacuous interval test: one feature must be negative."""
    out = Q.q6_behaviour_vs_dose(df)
    signs = [np.sign(r["pearson_r"]) for r in out["table"]]
    assert -1 in signs, "no negatively correlated behavioural feature"


def test_premise_check_reports_the_actual_direction(df):
    """Injury is assumed to suppress locomotion; the data must be believed."""
    out = Q.q6_behaviour_vs_dose(df)
    speed = next(r for r in out["table"]
                 if r["feature"] == "beh_baseline_speed_mm_s")
    assert out["premise_injury_suppresses_locomotion_holds"] == (
        speed["slope_per_kPa"] < 0)
    assert "premise" in out["premise_note"].lower()


# --------------------------------------------------------------------
# Question 7: a constant risk score is undefined, not chance
# --------------------------------------------------------------------
def test_constant_risk_stratum_is_reported_as_undefined(df):
    out = Q.q7_leave_one_stratum_out(df)
    for rec in out["per_held_out_stratum"]:
        if rec.get("risk_score_constant_within_stratum"):
            assert rec["c_index"] is None, (
                "a stratum where every larva scored identically was assigned "
                "a concordance index; 0.5 there is an artifact of ties, not a "
                "measurement"
            )
            assert "undefined" in rec["reason"]
