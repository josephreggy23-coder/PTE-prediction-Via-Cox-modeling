"""Reported inference: hazard ratios, CIs, Schoenfeld, AFT fallback.

Everything a reader sees as an effect estimate is produced here by
lifelines, not by the fast solver. The fast solver drives the
cross-validation and permutation machinery; this module drives the
numbers that go in the results table.

Two data-specific decisions are enforced here and logged in the output:

1. `lifelines` cannot combine `strata` with `cluster_col` (it raises
   KeyError). Clutch is therefore handled as a stratified baseline
   hazard -- each clutch gets its own baseline seizure threshold, which
   is the biologically meaningful part of a frailty term -- and the
   variance correction that the frailty term would have supplied is
   obtained by bootstrapping whole clutches.

2. lfp_power_{low,mid,high}_norm are a closed composition (they sum to
   1.000 +/- 0.0007). Any unpenalized fit on all three is chasing
   rounding noise, so unpenalized fits drop one band as the reference.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.statistics import proportional_hazard_test
from scipy import stats

from . import config as C

COMPOSITIONAL_BANDS = [
    "lfp_power_low_norm", "lfp_power_mid_norm", "lfp_power_high_norm",
]
REFERENCE_BAND = "lfp_power_high_norm"

FRAILTY_NOTE = (
    "Requested: clutch_id as a frailty term. lifelines 0.30.3 provides no "
    "gamma-frailty Cox model, and its `cluster_col` sandwich estimator "
    "cannot be combined with `strata` (raises KeyError). Substitution "
    "applied and logged: clutch_id enters as a stratified baseline hazard "
    "(separate baseline seizure threshold per clutch), and all reported "
    "intervals use a clutch-level bootstrap so that the unit of "
    "resampling is the biological replicate, not the individual larva."
)


def reference_free(features: list) -> list:
    """Drop the compositional reference band for unpenalized fits."""
    if all(b in features for b in COMPOSITIONAL_BANDS):
        return [f for f in features if f != REFERENCE_BAND]
    return list(features)


# --------------------------------------------------------------------
# Design assembly
# --------------------------------------------------------------------
def _frame(df, features, standardize=True):
    X = df[features].astype(float).copy()
    sd = X.std(ddof=0).replace(0.0, 1.0)
    mu = X.mean()
    if standardize:
        X = (X - mu) / sd
    X[C.DURATION_COL] = df[C.DURATION_COL].to_numpy(float)
    X[C.EVENT_COL] = df[C.EVENT_COL].to_numpy(int)
    X[C.CLUTCH_COL] = df[C.CLUTCH_COL].to_numpy()
    return X, mu, sd


def fit_cox(df, features, penalizer=0.0, l1_ratio=0.0, strata=True,
            standardize=True):
    """Stratified Cox fit via lifelines. penalizer=0 gives the MLE."""
    X, mu, sd = _frame(df, features, standardize)
    weights = np.array(
        [0.0 if f in C.UNPENALIZED else 1.0 for f in features]
    )
    pen = penalizer * weights if penalizer > 0 else 0.0
    cph = CoxPHFitter(penalizer=pen, l1_ratio=l1_ratio)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(X, C.DURATION_COL, C.EVENT_COL,
                strata=C.CLUTCH_COL if strata else None)
    return cph, mu, sd


# --------------------------------------------------------------------
# Hazard ratios
# --------------------------------------------------------------------
def hazard_ratios(df, features, penalizer, l1_ratio,
                  n_boot=C.N_BOOTSTRAP, seed=C.BOOTSTRAP_SEED,
                  alpha=C.ALPHA):
    """Elastic-net hazard ratios with clutch-level bootstrap CIs.

    Coefficients are on standardized predictors, so each hazard ratio is
    the multiplicative change in seizure hazard per one standard
    deviation of that feature across the cohort. The SD in raw units is
    reported alongside so the effect can be read biologically.
    """
    cph, mu, sd = fit_cox(df, features, penalizer, l1_ratio, strata=True)
    point = cph.params_.reindex(features).to_numpy()

    rng = np.random.default_rng(seed)
    clutches = df[C.CLUTCH_COL].to_numpy()
    groups = [np.flatnonzero(clutches == g) for g in np.unique(clutches)]
    idx = np.asarray(df.index)

    draws = np.full((n_boot, len(features)), np.nan)
    ok = 0
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[k] for k in pick])
        sub = df.iloc[rows].reset_index(drop=True)
        # Resampled clutches repeat; relabel so each draw is its own stratum.
        sub[C.CLUTCH_COL] = np.concatenate(
            [[f"{k}_{r}"] * len(groups[k]) for r, k in enumerate(pick)]
        )
        if sub[C.EVENT_COL].sum() < 5:
            continue
        try:
            f, _, _ = fit_cox(sub, features, penalizer, l1_ratio, strata=True)
            draws[b] = f.params_.reindex(features).to_numpy()
            ok += 1
        except Exception:
            continue

    lo = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)

    rows = []
    for j, f in enumerate(features):
        # A coefficient can be numerically non-zero and still be shrunk to
        # nothing (|log HR| ~ 1e-5). Calling that "selected" would let a
        # deleted feature masquerade as a retained one in the results file.
        negligible = abs(float(point[j])) < C.NEGLIGIBLE_LOG_HR
        rows.append({
            "feature": f,
            "coef_log_hr_per_sd": float(point[j]),
            "hazard_ratio_per_sd": float(np.exp(point[j])),
            "hr_ci_low": float(np.exp(lo[j])),
            "hr_ci_high": float(np.exp(hi[j])),
            "sd_raw_units": float(sd[f]),
            "selected": bool(not negligible),
            "exactly_zero": bool(point[j] == 0.0),
            "shrunk_to_negligible": bool(negligible),
            "bootstrap_prob_retained": float(
                np.nanmean(np.abs(draws[:, j]) >= C.NEGLIGIBLE_LOG_HR)),
        })
    return pd.DataFrame(rows), {"n_bootstrap_ok": ok, "n_bootstrap": n_boot}


def hazard_ratios_unpenalized(df, features):
    """Maximum-partial-likelihood HRs with Wald CIs, as a sensitivity check."""
    feats = reference_free(features)
    cph, mu, sd = fit_cox(df, feats, penalizer=0.0, strata=True)
    s = cph.summary
    rows = []
    for f in feats:
        rows.append({
            "feature": f,
            "hazard_ratio_per_sd": float(s.loc[f, "exp(coef)"]),
            "hr_ci_low": float(s.loc[f, "exp(coef) lower 95%"]),
            "hr_ci_high": float(s.loc[f, "exp(coef) upper 95%"]),
            "p": float(s.loc[f, "p"]),
        })
    return pd.DataFrame(rows), cph


# --------------------------------------------------------------------
# Likelihood ratio test
# --------------------------------------------------------------------
def likelihood_ratio_test(df, features_null, features_full):
    """Nested LRT on unpenalized stratified Cox fits.

    Penalized fits have no valid likelihood-ratio reference
    distribution, so both models are refitted unpenalized, with the
    compositional reference band dropped from the full model.
    """
    f_null = reference_free(features_null)
    f_full = reference_free(features_full)
    m0, _, _ = fit_cox(df, f_null, penalizer=0.0, strata=True)
    m1, _, _ = fit_cox(df, f_full, penalizer=0.0, strata=True)

    stat = 2.0 * (m1.log_likelihood_ - m0.log_likelihood_)
    dfree = len(f_full) - len(f_null)
    p = float(stats.chi2.sf(stat, dfree)) if stat > 0 else 1.0
    return {
        "loglik_M0": float(m0.log_likelihood_),
        "loglik_M1": float(m1.log_likelihood_),
        "lr_statistic": float(stat),
        "df": int(dfree),
        "p_value": p,
        "features_M0": f_null,
        "features_M1": f_full,
        "note": (
            f"Unpenalized stratified Cox fits. {REFERENCE_BAND} dropped from "
            f"M1 as the compositional reference (the three normalized power "
            f"bands sum to 1, so all three cannot be identified jointly "
            f"without a penalty)."
        ),
    }


# --------------------------------------------------------------------
# Proportional hazards
# --------------------------------------------------------------------
def schoenfeld(df, features, alpha=C.PH_ALPHA):
    """Schoenfeld residual test of the proportional hazards assumption."""
    feats = reference_free(features)
    cph, _, _ = fit_cox(df, feats, penalizer=0.0, strata=True)
    training = _frame(df, feats, standardize=True)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = proportional_hazard_test(cph, training, time_transform="rank")
    tab = res.summary.reset_index()
    per_feature = {
        str(r.iloc[0]): {"test_statistic": float(r["test_statistic"]),
                         "p": float(r["p"])}
        for _, r in tab.iterrows()
    }
    pvals = np.array([v["p"] for v in per_feature.values()])
    # Global test: sum of per-covariate chi-square(1) statistics.
    global_stat = float(np.sum([v["test_statistic"]
                                for v in per_feature.values()]))
    global_p = float(stats.chi2.sf(global_stat, len(per_feature)))
    violated = [k for k, v in per_feature.items() if v["p"] < alpha]

    return {
        "per_feature": per_feature,
        "global_statistic": global_stat,
        "global_df": len(per_feature),
        "global_p": global_p,
        "alpha": alpha,
        "violated_features": violated,
        "assumption_violated": bool(global_p < alpha),
        "min_feature_p": float(pvals.min()) if len(pvals) else float("nan"),
        "time_transform": "rank",
    }


# --------------------------------------------------------------------
# Weibull AFT fallback
# --------------------------------------------------------------------
def weibull_aft(df, features):
    """Weibull accelerated failure time refit.

    Reported as time ratios: a value below 1 means the feature SHORTENS
    latency to stage III, which is the AFT counterpart of a hazard ratio
    above 1.
    """
    feats = reference_free(features)
    X, mu, sd = _frame(df, feats, standardize=True)
    X = X.drop(columns=[C.CLUTCH_COL])
    aft = WeibullAFTFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aft.fit(X, C.DURATION_COL, C.EVENT_COL)
    s = aft.summary.loc["lambda_"]
    rows = []
    for f in feats:
        rows.append({
            "feature": f,
            "time_ratio_per_sd": float(np.exp(s.loc[f, "coef"])),
            "tr_ci_low": float(np.exp(s.loc[f, "coef lower 95%"])),
            "tr_ci_high": float(np.exp(s.loc[f, "coef upper 95%"])),
            "p": float(s.loc[f, "p"]),
            "implied_hazard_direction": (
                "higher hazard" if s.loc[f, "coef"] < 0 else "lower hazard"
            ),
        })
    return pd.DataFrame(rows), aft
