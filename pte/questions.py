"""The eight biological questions.

Each function returns a dict whose "answer" key is a sentence with the
supporting number in it, plus the numbers themselves for the results
file. The sentence is the deliverable; the rest is the evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from scipy import stats

from . import config as C
from . import cv, inference
from .pipeline import FoldPipeline


def _fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else f"{x:.{d}f}"


# ====================================================================
# 1. Does the brain's own signal add anything to knowing how hard the
#    fish was hit?
# ====================================================================
def q1_lfp_adds_over_pressure(df, cv_m0: cv.CVResult, cv_m1: cv.CVResult,
                              null_m0=None, null_m1=None):
    clutches = df[C.CLUTCH_COL].to_numpy()
    t, e = cv_m0.time, cv_m0.event

    c0, lo0, hi0, _ = cv.c_index_ci(t, cv_m0.risk, e, clusters=clutches)
    c1, lo1, hi1, _ = cv.c_index_ci(t, cv_m1.risk, e, clusters=clutches)

    # Paired bootstrap of the difference, resampling whole clutches.
    rng = np.random.default_rng(C.BOOTSTRAP_SEED + 1)
    groups = [np.flatnonzero(clutches == g) for g in np.unique(clutches)]
    diffs = []
    for _ in range(C.N_BOOTSTRAP):
        pick = rng.integers(0, len(groups), len(groups))
        s = np.concatenate([groups[k] for k in pick])
        if e[s].sum() < 2:
            continue
        try:
            diffs.append(concordance_index(t[s], -cv_m1.risk[s], e[s])
                         - concordance_index(t[s], -cv_m0.risk[s], e[s]))
        except ZeroDivisionError:
            continue
    d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])
    d_point = c1 - c0
    p_boot = float(2 * min(np.mean(np.asarray(diffs) <= 0),
                           np.mean(np.asarray(diffs) >= 0)))

    lrt = inference.likelihood_ratio_test(df, C.M0_FEATURES, C.M1_FEATURES)

    crosses = (d_lo <= 0 <= d_hi)
    verdict = "does not add" if (crosses and lrt["p_value"] >= 0.05) else "adds"
    answer = (
        f"The LFP block {verdict} detectable information beyond injury "
        f"severity: cross-validated concordance rises from {c0:.3f} "
        f"(95% CI {lo0:.3f}-{hi0:.3f}) with pressure alone to {c1:.3f} "
        f"({lo1:.3f}-{hi1:.3f}) with pressure plus LFP, a change of "
        f"{d_point:+.3f} (95% CI {d_lo:+.3f} to {d_hi:+.3f}), and the "
        f"nested likelihood ratio test gives chi2({lrt['df']}) = "
        f"{lrt['lr_statistic']:.2f}, p = {lrt['p_value']:.3f}."
    )

    return {
        "answer": answer,
        "M0_c_index": c0, "M0_ci": [lo0, hi0],
        "M1_c_index": c1, "M1_ci": [lo1, hi1],
        "delta_c_index": d_point, "delta_ci": [float(d_lo), float(d_hi)],
        "delta_bootstrap_p": p_boot,
        "delta_ci_crosses_zero": bool(crosses),
        "likelihood_ratio_test": lrt,
        "permutation_null_M0": null_m0,
        "permutation_null_M1": null_m1,
        "bootstrap_unit": "clutch",
    }


# ====================================================================
# 2. Do the retained features point the right way?
# ====================================================================
def _sign_of(log_hr: float) -> int:
    """Direction of an effect, with shrunk-to-nothing treated as no direction."""
    if not np.isfinite(log_hr) or abs(log_hr) < C.NEGLIGIBLE_LOG_HR:
        return 0
    return 1 if log_hr > 0 else -1


def q2_direction_check(hr_table: pd.DataFrame, hr_unpen: pd.DataFrame):
    rows = []
    for feat, expected in C.EXPECTED_DIRECTION.items():
        r = hr_table[hr_table.feature == feat]
        if r.empty:
            continue
        r = r.iloc[0]
        hr = float(r.hazard_ratio_per_sd)
        observed = _sign_of(float(r.coef_log_hr_per_sd))

        u = hr_unpen[hr_unpen.feature == feat]
        hr_u = float(u.iloc[0].hazard_ratio_per_sd) if not u.empty else np.nan
        obs_u = _sign_of(np.log(hr_u)) if np.isfinite(hr_u) else 0

        # "Retained" means the penalty left a coefficient big enough to
        # carry a claim, not merely one that is numerically non-zero.
        retained = observed != 0
        inverted = retained and observed != expected
        inverted_u = obs_u != 0 and obs_u != expected

        rows.append({
            "feature": feat,
            "expected_direction": "HR > 1" if expected > 0 else "HR < 1",
            "expected_sign": expected,
            "rationale": C.DIRECTION_RATIONALE[feat],
            "hazard_ratio_per_sd": hr,
            "hr_ci_low": float(r.hr_ci_low),
            "hr_ci_high": float(r.hr_ci_high),
            "observed_sign": observed,
            "retained_by_elastic_net": bool(retained),
            "shrunk_to_negligible": bool(not retained),
            "inverted": bool(inverted),
            "ci_excludes_1": bool(r.hr_ci_low > 1 or r.hr_ci_high < 1),
            "unpenalized_hazard_ratio_per_sd": hr_u,
            "unpenalized_sign": obs_u,
            "unpenalized_inverted": bool(inverted_u),
            "unpenalized_sign_agrees_with_penalized": bool(obs_u == observed),
        })

    tab = pd.DataFrame(rows)
    retained = tab[tab.retained_by_elastic_net]
    inverted = tab[tab.inverted]
    inv_u = tab[tab.unpenalized_inverted]
    ok_u = tab[(tab.unpenalized_sign != 0)
               & (~tab.unpenalized_inverted.astype(bool))]

    def _bits(frame, col="hazard_ratio_per_sd"):
        return ", ".join(
            f"{r.feature} (HR {r[col]:.2f} per SD, expected "
            f"{r.expected_direction})" for _, r in frame.iterrows()
        )

    if retained.empty:
        # The honest reading: the penalty answered the question by
        # deleting the features, so the direction check has to fall back
        # to the unpenalized sensitivity fit and say so.
        answer = (
            f"The elastic net shrank all {len(tab)} mechanistic LFP "
            f"coefficients to effectively zero (every |log HR| < "
            f"{C.NEGLIGIBLE_LOG_HR}, i.e. hazard ratios inside 0.99-1.01 per "
            f"SD), so the penalized model retains no direction to check. In "
            f"the unpenalized sensitivity fit, {len(inv_u)} of "
            f"{len(inv_u) + len(ok_u)} point the WRONG way"
            + (f" ({_bits(inv_u, 'unpenalized_hazard_ratio_per_sd')})"
               if len(inv_u) else "")
            + (f", while {_bits(ok_u, 'unpenalized_hazard_ratio_per_sd')} "
               f"point as predicted." if len(ok_u) else ".")
        )
    elif inverted.empty:
        answer = (
            f"All {len(retained)} retained mechanistic features point the way "
            f"the biology predicts, with no inversions: {_bits(retained)}."
        )
    else:
        answer = (
            f"{len(inverted)} of {len(retained)} retained mechanistic features "
            f"point the WRONG way: {_bits(inverted)}. Discrimination built on "
            f"inverted biology is fitting noise, not seizure susceptibility."
        )

    return {
        "answer": answer,
        "table": tab.to_dict("records"),
        "n_checked": int(len(tab)),
        "n_retained": int(len(retained)),
        "n_shrunk_to_negligible": int(len(tab) - len(retained)),
        "n_inverted": int(len(inverted)),
        "inverted_features": inverted.feature.tolist(),
        "n_inverted_unpenalized": int(len(inv_u)),
        "inverted_features_unpenalized": inv_u.feature.tolist(),
        "negligible_log_hr_threshold": C.NEGLIGIBLE_LOG_HR,
        "note": (
            "A hazard ratio within 1% of 1.0 per SD is treated as carrying no "
            "direction. Reading the sign of a coefficient the penalty has "
            "already deleted would manufacture agreement out of rounding."
        ),
    }


# ====================================================================
# 3. Do the two critical slowing features agree with each other?
# ====================================================================
def q3_critical_slowing_agreement(df, hr_table: pd.DataFrame,
                                  hr_unpen: pd.DataFrame):
    a, b = C.CRITICAL_SLOWING_PAIR
    ra = hr_table[hr_table.feature == a].iloc[0]
    rb = hr_table[hr_table.feature == b].iloc[0]
    hra, hrb = float(ra.hazard_ratio_per_sd), float(rb.hazard_ratio_per_sd)

    sa = _sign_of(float(ra.coef_log_hr_per_sd))
    sb = _sign_of(float(rb.coef_log_hr_per_sd))
    agree = (sa == sb) and sa != 0

    ua = hr_unpen[hr_unpen.feature == a]
    ub = hr_unpen[hr_unpen.feature == b]
    hra_u = float(ua.iloc[0].hazard_ratio_per_sd) if not ua.empty else np.nan
    hrb_u = float(ub.iloc[0].hazard_ratio_per_sd) if not ub.empty else np.nan
    sa_u = _sign_of(np.log(hra_u)) if np.isfinite(hra_u) else 0
    sb_u = _sign_of(np.log(hrb_u)) if np.isfinite(hrb_u) else 0
    agree_u = (sa_u == sb_u) and sa_u != 0

    r_pear = float(np.corrcoef(df[a], df[b])[0, 1])
    rho, p_rho = stats.spearmanr(df[a], df[b])

    both_expected = (sa == C.EXPECTED_DIRECTION[a]
                     and sb == C.EXPECTED_DIRECTION[b])
    both_expected_u = (sa_u == C.EXPECTED_DIRECTION[a]
                       and sb_u == C.EXPECTED_DIRECTION[b])

    if sa == 0 and sb == 0:
        verdict = ("were both shrunk out of the penalized model entirely, so "
                   "the penalized fit cannot say whether they agree")
        detail = (
            f" In the unpenalized sensitivity fit they "
            f"{'agree' if agree_u else 'disagree'}: {a} HR {hra_u:.2f} per SD "
            f"and {b} HR {hrb_u:.2f} per SD, "
            + ("both running counter to the critical slowing prediction, "
               "which points to a shared confound rather than a shared "
               "mechanism."
               if agree_u and not both_expected_u else
               "both as the critical slowing account predicts."
               if agree_u else
               "so at least one is tracking an artifact rather than approach "
               "to instability.")
        )
    elif agree and both_expected:
        verdict = ("agree with each other and with the critical slowing "
                   "prediction")
        detail = ""
    elif agree:
        verdict = ("agree with each other but both run counter to the "
                   "critical slowing prediction, which points to a shared "
                   "confound rather than a shared mechanism")
        detail = ""
    else:
        verdict = ("disagree, so at least one of them is picking up an "
                   "artifact rather than approach to instability")
        detail = ""

    answer = (
        f"The two critical slowing indicators {verdict}: {a} gives HR "
        f"{hra:.2f} per SD ({ra.hr_ci_low:.2f}-{ra.hr_ci_high:.2f}) and {b} "
        f"gives HR {hrb:.2f} per SD ({rb.hr_ci_low:.2f}-{rb.hr_ci_high:.2f}), "
        f"while the two features are themselves correlated at r = "
        f"{r_pear:.2f} (Spearman rho = {rho:.2f}, p = {p_rho:.3g}) across "
        f"larvae, so they do co-vary biologically even where the model "
        f"declines to use them." + detail
    )

    return {
        "answer": answer,
        "feature_a": a, "feature_b": b,
        "hr_a": hra, "hr_a_ci": [float(ra.hr_ci_low), float(ra.hr_ci_high)],
        "hr_b": hrb, "hr_b_ci": [float(rb.hr_ci_low), float(rb.hr_ci_high)],
        "sign_a": sa, "sign_b": sb,
        "signs_agree": bool(agree),
        "both_shrunk_to_negligible": bool(sa == 0 and sb == 0),
        "both_match_expected_direction": bool(both_expected),
        "unpenalized_hr_a": hra_u, "unpenalized_hr_b": hrb_u,
        "unpenalized_signs_agree": bool(agree_u),
        "unpenalized_both_match_expected": bool(both_expected_u),
        "feature_correlation_pearson": r_pear,
        "feature_correlation_spearman": float(rho),
        "feature_correlation_p": float(p_rho),
    }


# ====================================================================
# 4. Do two larvae that took the same hit differ predictably?
# ====================================================================
def q4_within_stratum(df, cv_m1: cv.CVResult, min_events=10, min_epv=5.0):
    t, e, risk = cv_m1.time, cv_m1.event, cv_m1.risk
    strata = df[C.STRATUM_COL].to_numpy()

    # --- primary: within-stratum concordance of the global LOO score -----
    # Only larvae that took the same hit are compared, so injury severity
    # is held constant by construction.
    per_stratum = []
    for h in np.unique(strata):
        m = strata == h
        n_ev = int(e[m].sum())
        if n_ev < 2:
            per_stratum.append({"drop_height_cm": int(h), "n": int(m.sum()),
                                "n_events": n_ev, "answerable": False,
                                "reason": "fewer than 2 events"})
            continue
        c, lo, hi, _ = cv.c_index_ci(t[m], risk[m], e[m],
                                     n_boot=C.N_BOOTSTRAP // 2)
        per_stratum.append({
            "drop_height_cm": int(h), "n": int(m.sum()), "n_events": n_ev,
            "c_index_within": float(c), "ci": [float(lo), float(hi)],
            "answerable": bool(n_ev >= min_events),
            "ci_excludes_chance": bool(lo > 0.5),
        })

    # Pooled: all comparable pairs, pooling across strata.
    pooled_num = pooled_den = 0.0
    for h in np.unique(strata):
        m = strata == h
        if e[m].sum() < 2:
            continue
        c = concordance_index(t[m], -risk[m], e[m])
        # weight by number of comparable pairs in the stratum
        npairs = _comparable_pairs(t[m], e[m])
        pooled_num += c * npairs
        pooled_den += npairs
    pooled = pooled_num / pooled_den if pooled_den else float("nan")

    # --- secondary: full refit inside each stratum, as specified ---------
    refits = []
    for h in np.unique(strata):
        sub = df[df[C.STRATUM_COL] == h].reset_index(drop=True)
        n_ev = int(sub[C.EVENT_COL].sum())
        # max_pressure_kPa is constant within a stratum for sham (all 0.0);
        # a constant column cannot carry a coefficient, so it is dropped.
        feats = [f for f in C.M1_FEATURES
                 if float(sub[f].std(ddof=0)) > 1e-12]
        epv = n_ev / max(len(feats), 1)
        rec = {"drop_height_cm": int(h), "n": int(len(sub)),
               "n_events": n_ev, "n_predictors": len(feats),
               "events_per_predictor": round(epv, 2),
               "pressure_dropped_constant": C.PRESSURE not in feats}
        if n_ev < min_events:
            rec.update({"refit_performed": False,
                        "reason": f"only {n_ev} events (< {min_events})"})
        else:
            r = cv.leave_one_fish_out(sub, feats, use_clutch_strata=False,
                                      tag=f"q4_h{h}", compute_medians=False)
            c, lo, hi, _ = cv.c_index_ci(r.time, r.risk, r.event,
                                         n_boot=C.N_BOOTSTRAP // 2)
            rec.update({
                "refit_performed": True,
                "c_index": float(c), "ci": [float(lo), float(hi)],
                "interpretable": bool(epv >= min_epv),
                "caveat": (
                    None if epv >= min_epv else
                    f"events-per-predictor {epv:.2f} is far below {min_epv:g}; "
                    f"this estimate is descriptive only"
                ),
            })
        refits.append(rec)

    ok = [s for s in per_stratum if s.get("answerable")]
    sig = [s for s in ok if s.get("ci_excludes_chance")]
    unanswerable = [s for s in per_stratum if not s.get("answerable")]
    per_txt = ", ".join(f"{s['drop_height_cm']} cm {s['c_index_within']:.3f}"
                        for s in ok)

    if not ok:
        lead = ("No drop-height stratum carries enough events to answer the "
                "question.")
    elif len(sig) == 0:
        lead = (
            f"No: once the hit is held constant, the model cannot tell two "
            f"larvae apart. Within-stratum concordance is {pooled:.3f} "
            f"pair-weighted (chance is 0.500), and not one of the {len(ok)} "
            f"drop-height strata has an interval excluding chance "
            f"(per-stratum: {per_txt})."
        )
    else:
        lead = (
            f"Holding the hit constant, larvae still order correctly more "
            f"often than chance in {len(sig)} of {len(ok)} drop-height strata, "
            f"with a pair-weighted within-stratum concordance of "
            f"{pooled:.3f} (per-stratum: {per_txt})."
        )

    answer = (
        lead
        + (f" Full within-stratum refits are reported but rest on only "
           f"{min(r['n_events'] for r in refits)}-"
           f"{max(r['n_events'] for r in refits)} events against up to "
           f"{max(r['n_predictors'] for r in refits)} predictors, so they are "
           f"descriptive rather than inferential."
           if refits else "")
        + (f" Strata that cannot answer the question at all: "
           f"{[s['drop_height_cm'] for s in unanswerable]}."
           if unanswerable else "")
    )

    return {
        "answer": answer,
        "method_primary": (
            "within-stratum concordance of the global leave-one-fish-out risk "
            "score; only larvae from the same drop height are compared"
        ),
        "per_stratum": per_stratum,
        "pooled_within_stratum_c_index": float(pooled),
        "method_secondary": (
            "independent elastic-net refit inside each stratum, as specified"
        ),
        "within_stratum_refits": refits,
        "min_events_to_report": min_events,
        "min_epv_to_interpret": min_epv,
    }


def _comparable_pairs(t, e):
    n, count = len(t), 0
    for i in range(n):
        for j in range(i + 1, n):
            if (e[i] and t[i] < t[j]) or (e[j] and t[j] < t[i]) \
                    or (e[i] and e[j] and t[i] == t[j]):
                count += 1
    return count


# ====================================================================
# 5. Does predicted susceptibility follow the injury gradient?
# ====================================================================
def q5_injury_gradient(df, cv_m1: cv.CVResult):
    order = ["Sham", "TBI_27cm", "TBI_54cm", "TBI_108cm"]
    heights = {"Sham": 0, "TBI_27cm": 27, "TBI_54cm": 54, "TBI_108cm": 108}
    risk = cv_m1.risk
    cond = df[C.LABEL_COL].to_numpy()

    means, rows = [], []
    for c in order:
        m = cond == c
        v = risk[m]
        se = float(v.std(ddof=1) / np.sqrt(m.sum()))
        means.append(float(v.mean()))
        rows.append({
            "condition": c, "drop_height_cm": heights[c], "n": int(m.sum()),
            "mean_risk_score": float(v.mean()),
            "sd_risk_score": float(v.std(ddof=1)),
            "ci95": [float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)],
            "median_risk_score": float(np.median(v)),
        })

    monotone = all(means[i] < means[i + 1] for i in range(len(means) - 1))
    observed_order = [order[i] for i in np.argsort(means)]

    h = df[C.STRATUM_COL].to_numpy(float)
    rho, p_rho = stats.spearmanr(h, risk)
    kw = stats.kruskal(*[risk[cond == c] for c in order])

    answer = (
        f"Mean predicted susceptibility "
        f"{'follows' if monotone else 'does NOT follow'} the injury gradient: "
        f"the observed ordering is "
        + " < ".join(observed_order)
        + f" (means {', '.join(f'{m:+.3f}' for m in means)}), against the "
        f"expected sham < low < mid < high, with a monotone trend of Spearman "
        f"rho = {rho:.3f} (p = {p_rho:.3g}) between drop height and risk "
        f"score."
    )

    return {
        "answer": answer,
        "expected_order": order,
        "observed_order": observed_order,
        "is_monotone_as_expected": bool(monotone),
        "per_condition": rows,
        "spearman_rho_height_vs_risk": float(rho),
        "spearman_p": float(p_rho),
        "kruskal_statistic": float(kw.statistic),
        "kruskal_p": float(kw.pvalue),
        "note": ("Risk score is the held-out Cox linear predictor; higher "
                 "means higher hazard, i.e. shorter predicted latency."),
    }


# ====================================================================
# 6. Is the behavioral signal just re-reporting injury dose?
# ====================================================================
def q6_behaviour_vs_dose(df, r2_threshold=0.5):
    x = df[C.PRESSURE].to_numpy(float)
    rows = []
    for f in C.BEH_FEATURES:
        y = df[f].to_numpy(float)
        slope, intercept, r, p, se = stats.linregress(x, y)
        r2 = float(r ** 2)
        # Fisher z interval on the correlation -> interval on R^2.
        z = np.arctanh(np.clip(r, -0.999999, 0.999999))
        sez = 1.0 / np.sqrt(len(x) - 3)
        lo_r, hi_r = np.tanh(z - 1.96 * sez), np.tanh(z + 1.96 * sez)
        # Square FIRST, then order. For a negative correlation the lower
        # bound on r maps to the UPPER bound on R^2, so ordering the
        # correlation bounds and squaring them yields an inverted interval.
        sq = sorted((float(lo_r ** 2), float(hi_r ** 2)))
        rows.append({
            "feature": f,
            "r2_on_pressure": r2,
            "r2_ci": [sq[0], sq[1]],
            "pearson_r": float(r),
            "slope_per_kPa": float(slope),
            "p_value": float(p),
            "residual_variance_fraction": float(1 - r2),
            "largely_explained_by_dose": bool(r2 >= r2_threshold),
            "direction_vs_pressure": "increases" if slope > 0 else "decreases",
        })
    tab = pd.DataFrame(rows)
    redundant = tab[tab.largely_explained_by_dose]

    # The brief's premise is that injury suppresses locomotion. Check it.
    speed = tab[tab.feature == "beh_baseline_speed_mm_s"].iloc[0]
    premise_holds = speed.slope_per_kPa < 0

    answer = (
        f"{len(redundant)} of {len(tab)} baseline behavioural features are "
        f"largely a restatement of injury dose (R^2 on max_pressure_kPa alone: "
        + ", ".join(f"{r.feature} {r.r2_on_pressure:.2f}"
                    for _, r in tab.iterrows())
        + f"), leaving only {tab.residual_variance_fraction.min():.0%}-"
        f"{tab.residual_variance_fraction.max():.0%} of their variance "
        f"independent of the hit."
    )

    return {
        "answer": answer,
        "table": tab.to_dict("records"),
        "r2_threshold": r2_threshold,
        "n_largely_explained": int(len(redundant)),
        "premise_injury_suppresses_locomotion_holds": bool(premise_holds),
        "premise_note": (
            "The stated premise is that injury suppresses locomotion. In this "
            "cohort baseline speed RISES with pressure (slope "
            f"{speed.slope_per_kPa:+.5f} mm/s per kPa, r = {speed.pearson_r:.2f}), "
            "so the redundancy finding holds but the sign of the underlying "
            "relationship is opposite to the one assumed."
            if not premise_holds else
            "Baseline speed falls with pressure, consistent with the stated "
            "premise that injury suppresses locomotion."
        ),
    }


# ====================================================================
# 7. Does it hold for an injury level it has never seen?
# ====================================================================
def q7_leave_one_stratum_out(df):
    strata = np.sort(df[C.STRATUM_COL].unique())
    rows = []
    for h in strata:
        train_idx = df.index[df[C.STRATUM_COL] != h].tolist()
        test_idx = df.index[df[C.STRATUM_COL] == h].tolist()

        pipe = FoldPipeline(features=C.M1_FEATURES, use_clutch_strata=True,
                            tag=f"q7_holdout_{h}").fit(df, train_idx)
        risk = pipe.predict(df, test_idx)

        t = df.loc[test_idx, C.DURATION_COL].to_numpy(float)
        e = df.loc[test_idx, C.EVENT_COL].to_numpy(float)
        n_ev = int(e.sum())

        rec = {
            "held_out_drop_height_cm": int(h),
            "n_train": len(train_idx), "n_test": len(test_idx),
            "n_events_test": n_ev,
            "chosen_penalizer": pipe.best_penalizer_,
            "chosen_l1_ratio": pipe.best_l1_ratio_,
            "selected_features": pipe.model_.selected,
            "pressure_out_of_range": bool(
                df.loc[test_idx, C.PRESSURE].max()
                > df.loc[train_idx, C.PRESSURE].max()
                or df.loc[test_idx, C.PRESSURE].min()
                < df.loc[train_idx, C.PRESSURE].min()
            ),
        }
        # A held-out stratum can receive one identical score for every
        # larva: if the only surviving predictor is max_pressure_kPa and
        # that stratum has a single pressure value (sham is 0.0 for all
        # 25), the risk score is constant and concordance is 0.5 by
        # construction, not by measurement.
        degenerate = bool(np.ptp(risk) < 1e-9)
        rec["risk_score_constant_within_stratum"] = degenerate
        if degenerate:
            rec["degenerate_reason"] = (
                f"every held-out larva received the same risk score "
                f"({risk[0]:.4f}); the retained predictors are constant "
                f"within this stratum, so concordance is undefined rather "
                f"than at chance"
            )

        if n_ev >= 2 and not degenerate:
            c, lo, hi, _ = cv.c_index_ci(t, risk, e, n_boot=C.N_BOOTSTRAP // 2)
            rec.update({"c_index": float(c), "ci": [float(lo), float(hi)],
                        "above_chance": bool(lo > 0.5)})
        elif degenerate:
            rec.update({"c_index": None, "reason": rec["degenerate_reason"]})
        else:
            rec.update({"c_index": None,
                        "reason": f"only {n_ev} events in held-out stratum"})
        rows.append(rec)

    done = [r for r in rows if r.get("c_index") is not None]
    above = [r for r in done if r.get("above_chance")]
    skipped = [r for r in rows if r.get("c_index") is None]

    if not done:
        lead = ("No held-out injury level could be evaluated at all")
    elif not above:
        lead = (
            f"No: transferred to an injury level it has never seen, the model "
            f"is indistinguishable from chance in all {len(done)} evaluable "
            f"held-out strata"
        )
    else:
        lead = (
            f"Trained on three injury levels and tested on the fourth, the "
            f"model holds above chance in {len(above)} of {len(done)} "
            f"held-out strata"
        )

    answer = (
        lead
        + (" (" + ", ".join(
            f"{r['held_out_drop_height_cm']} cm: c = {r['c_index']:.3f} "
            f"[{r['ci'][0]:.3f}-{r['ci'][1]:.3f}]" for r in done) + ")"
           if done else "")
        + ", with every hyperparameter and scaling constant learned from the "
          "three training strata only."
        + ("".join(
            f" The {r['held_out_drop_height_cm']} cm stratum could not be "
            f"scored at all: {r['reason']}." for r in skipped))
    )
    return {"answer": answer, "per_held_out_stratum": rows,
            "n_above_chance": len(above), "n_evaluated": len(done),
            "n_not_evaluable": len(skipped)}


# ====================================================================
# 8. Are the predicted latencies right in absolute terms?
# ====================================================================
def q8_calibration(df, cv_m1: cv.CVResult, n_groups=4):
    t, e, risk = cv_m1.time, cv_m1.event, cv_m1.risk
    pred = cv_m1.pred_median_s

    # --- calibration slope in the risk domain (van Houwelingen) ---------
    d = pd.DataFrame({"T": t, "E": e.astype(int), "lp": risk})
    cph = CoxPHFitter().fit(d, "T", "E")
    slope = float(cph.params_["lp"])
    s_lo = float(cph.confidence_intervals_.loc["lp"].iloc[0])
    s_hi = float(cph.confidence_intervals_.loc["lp"].iloc[1])

    # --- calibration in the time domain, seconds ------------------------
    ev = (e == 1) & np.isfinite(pred) & (pred > 0)
    n_inf = int(np.sum(~np.isfinite(pred)))
    if ev.sum() >= 5:
        lo_obs = np.log(t[ev])
        lo_pred = np.log(pred[ev])
        tslope, tintercept, tr, tp, tse = stats.linregress(lo_pred, lo_obs)
        bias = float(np.mean(pred[ev] - t[ev]))
        mae = float(np.mean(np.abs(pred[ev] - t[ev])))
        medae = float(np.median(np.abs(pred[ev] - t[ev])))
    else:
        tslope = tintercept = tr = tp = bias = mae = medae = float("nan")

    # --- grouped calibration curve --------------------------------------
    groups = pd.qcut(risk, n_groups, labels=False, duplicates="drop")
    curve = []
    kmf = KaplanMeierFitter()
    for g in np.unique(groups):
        m = groups == g
        kmf.fit(t[m], e[m])
        obs_med = float(kmf.median_survival_time_)
        finite = np.isfinite(pred[m])
        curve.append({
            "risk_group": int(g) + 1,
            "n": int(m.sum()),
            "n_events": int(e[m].sum()),
            "mean_risk_score": float(risk[m].mean()),
            "predicted_median_s": (float(np.median(pred[m][finite]))
                                   if finite.any() else None),
            "observed_km_median_s": obs_med if np.isfinite(obs_med) else None,
        })

    answer = (
        f"Ranking is better than absolute timing: the calibration slope in the "
        f"risk domain is {slope:.3f} (95% CI {s_lo:.3f}-{s_hi:.3f}, where 1.0 "
        f"is perfect), while in the time domain regressing observed on "
        f"predicted log-latency gives slope {_fmt(tslope)} and intercept "
        f"{_fmt(tintercept)}, with predictions off by a median of "
        f"{_fmt(medae, 0)} s and a mean signed bias of {_fmt(bias, 0)} s "
        f"against a {int(df[C.CENSOR_TIME_COL].iloc[0])} s observation window."
    )

    return {
        "answer": answer,
        "calibration_slope_risk_domain": slope,
        "calibration_slope_ci": [s_lo, s_hi],
        "calibration_slope_target": 1.0,
        "time_domain_slope": float(tslope),
        "time_domain_intercept": float(tintercept),
        "time_domain_r": float(tr),
        "time_domain_p": float(tp),
        "time_domain_note": (
            "OLS of log(observed latency) on log(predicted median latency), "
            "events only. Slope 1 and intercept 0 would be perfect."
        ),
        "mean_signed_bias_s": bias,
        "mean_absolute_error_s": mae,
        "median_absolute_error_s": medae,
        "n_events_used": int(ev.sum()),
        "n_predicted_beyond_horizon": n_inf,
        "observation_window_s": float(df[C.CENSOR_TIME_COL].iloc[0]),
        "curve": curve,
    }
