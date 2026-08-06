"""Writers for the two output artefacts: results JSON and metrics CSV."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C


def _packages() -> dict:
    import importlib
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("numpy", "pandas", "scipy", "lifelines", "matplotlib",
              "openpyxl"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:
            out[m] = "not installed"
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=C.ROOT, capture_output=True,
            text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        return None


def config_block() -> dict:
    return {
        "seeds": {
            "main": C.SEED,
            "permutation": C.PERMUTATION_SEED,
            "bootstrap": C.BOOTSTRAP_SEED,
        },
        "model": {
            "family": "elastic-net regularized Cox proportional hazards",
            "penalizer_grid": C.PENALIZER_GRID,
            "l1_ratio_grid": C.L1_RATIO_GRID,
            "unpenalized_covariates": sorted(C.UNPENALIZED),
            "inner_cv_folds": C.INNER_FOLDS,
            "inner_cv_criterion": "cross-validated partial likelihood",
            "outer_cv": "leave-one-fish-out (100 folds)",
            "clutch_handling": "stratified baseline hazard + clutch bootstrap",
        },
        "features": {
            "M0": C.M0_FEATURES,
            "M1": C.M1_FEATURES,
            "behavioural_not_in_survival_models": C.BEH_FEATURES,
            "qc_excluded": C.QC_COLUMNS,
            "constant_dropped": C.CONSTANT_COLUMNS,
        },
        "inference": {
            "n_permutations": C.N_PERMUTATIONS,
            "n_bootstrap": C.N_BOOTSTRAP,
            "alpha": C.ALPHA,
            "ph_alpha": C.PH_ALPHA,
            "epv_threshold": C.EPV_THRESHOLD,
        },
        "expected_directions": {
            k: ("HR > 1" if v > 0 else "HR < 1")
            for k, v in C.EXPECTED_DIRECTION.items()
        },
    }


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_json(results: dict, path):
    payload = dict(results)
    payload["provenance"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "package_versions": _packages(),
        "config": config_block(),
        "data_file": C.DATA_FILE.name,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_jsonable)
    return path


# --------------------------------------------------------------------
# Flat metrics table
# --------------------------------------------------------------------
def _row(question, metric, value, lo=None, hi=None, unit="", note=""):
    return {
        "question": question, "metric": metric, "value": value,
        "ci_low": lo, "ci_high": hi, "unit": unit, "note": note,
    }


def build_metrics_table(r: dict) -> pd.DataFrame:
    rows = []
    d = r["cohort"]

    rows += [
        _row("cohort", "n_larvae", d["n_larvae"], unit="larvae"),
        _row("cohort", "n_events", d["n_events"], unit="larvae",
             note="reached stage III"),
        _row("cohort", "n_censored", d["n_censored"], unit="larvae",
             note=f"administrative at {d['censor_horizon_s']:.0f} s"),
        _row("cohort", "events_per_predictor_M1",
             d["events_per_predictor_M1"],
             note=d["epv_flag"]),
    ]

    q1 = r["q1_lfp_adds_over_pressure"]
    rows += [
        _row("q1", "c_index_M0_pressure_only", q1["M0_c_index"],
             q1["M0_ci"][0], q1["M0_ci"][1], note="leave-one-fish-out"),
        _row("q1", "c_index_M1_pressure_plus_lfp", q1["M1_c_index"],
             q1["M1_ci"][0], q1["M1_ci"][1], note="leave-one-fish-out"),
        _row("q1", "delta_c_index", q1["delta_c_index"],
             q1["delta_ci"][0], q1["delta_ci"][1], note="M1 - M0, paired"),
        _row("q1", "likelihood_ratio_chi2",
             q1["likelihood_ratio_test"]["lr_statistic"],
             note=f"df = {q1['likelihood_ratio_test']['df']}"),
        _row("q1", "likelihood_ratio_p",
             q1["likelihood_ratio_test"]["p_value"]),
    ]
    for key, lab in (("permutation_null_M0", "M0"),
                     ("permutation_null_M1", "M1")):
        if q1.get(key):
            n = q1[key]
            rows += [
                _row("q1", f"permutation_p_{lab}", n["p_value"],
                     note=f"{n['n_permutations']} end-to-end refits"),
                _row("q1", f"permutation_null_mean_{lab}", n["null_mean"]),
                _row("q1", f"permutation_null_95th_{lab}", n["null_q95"]),
            ]

    for t in r["q2_direction_check"]["table"]:
        if not t["retained_by_elastic_net"]:
            verdict = "SHRUNK TO ZERO by the penalty; no direction to check"
        elif t["inverted"]:
            verdict = "INVERTED"
        else:
            verdict = "as expected"
        rows.append(_row(
            "q2", f"HR_{t['feature']}", t["hazard_ratio_per_sd"],
            t["hr_ci_low"], t["hr_ci_high"], unit="per SD",
            note=f"elastic net; expected {t['expected_direction']}; {verdict}",
        ))
        # The unpenalized fit is where the direction is actually readable
        # once the penalty has deleted the block, so carry it in the table.
        rows.append(_row(
            "q2", f"HR_unpenalized_{t['feature']}",
            t["unpenalized_hazard_ratio_per_sd"], unit="per SD",
            note=(f"unpenalized sensitivity fit; expected "
                  f"{t['expected_direction']}; "
                  f"{'INVERTED' if t['unpenalized_inverted'] else 'as expected'}"),
        ))

    q3 = r["q3_critical_slowing_agreement"]
    rows += [
        _row("q3", "signs_agree", int(q3["signs_agree"]),
             note=f"{q3['feature_a']} vs {q3['feature_b']}"),
        _row("q3", "feature_correlation_pearson",
             q3["feature_correlation_pearson"]),
        _row("q3", "both_match_expected_direction",
             int(q3["both_match_expected_direction"])),
    ]

    q4 = r["q4_within_stratum"]
    rows.append(_row("q4", "pooled_within_stratum_c_index",
                     q4["pooled_within_stratum_c_index"],
                     note="global risk score, same-hit pairs only"))
    for s in q4["per_stratum"]:
        if "c_index_within" in s:
            rows.append(_row(
                "q4", f"c_index_within_{s['drop_height_cm']}cm",
                s["c_index_within"], s["ci"][0], s["ci"][1],
                note=f"{s['n_events']} events / {s['n']} larvae"))
    for s in q4["within_stratum_refits"]:
        rows.append(_row(
            "q4", f"refit_c_index_{s['drop_height_cm']}cm",
            s.get("c_index"),
            (s.get("ci") or [None, None])[0], (s.get("ci") or [None, None])[1],
            note=(s.get("caveat") or s.get("reason")
                  or f"EPV = {s['events_per_predictor']}")))

    q5 = r["q5_injury_gradient"]
    for c in q5["per_condition"]:
        rows.append(_row("q5", f"mean_risk_{c['condition']}",
                         c["mean_risk_score"], c["ci95"][0], c["ci95"][1],
                         note=f"n = {c['n']}"))
    rows += [
        _row("q5", "ordering_is_monotone", int(q5["is_monotone_as_expected"]),
             note=" < ".join(q5["observed_order"])),
        _row("q5", "spearman_rho_height_vs_risk",
             q5["spearman_rho_height_vs_risk"],
             note=f"p = {q5['spearman_p']:.3g}"),
    ]

    for t in r["q6_behaviour_vs_dose"]["table"]:
        rows.append(_row(
            "q6", f"R2_on_pressure_{t['feature']}", t["r2_on_pressure"],
            t["r2_ci"][0], t["r2_ci"][1],
            note=("largely explained by dose"
                  if t["largely_explained_by_dose"] else "independent"),
        ))

    for s in r["q7_leave_one_stratum_out"]["per_held_out_stratum"]:
        rows.append(_row(
            "q7", f"c_index_holdout_{s['held_out_drop_height_cm']}cm",
            s.get("c_index"),
            (s.get("ci") or [None, None])[0], (s.get("ci") or [None, None])[1],
            note=s.get("reason") or f"{s['n_events_test']} events held out"))

    q8 = r["q8_calibration"]
    rows += [
        _row("q8", "calibration_slope_risk_domain",
             q8["calibration_slope_risk_domain"],
             q8["calibration_slope_ci"][0], q8["calibration_slope_ci"][1],
             note="target 1.000"),
        _row("q8", "calibration_slope_time_domain", q8["time_domain_slope"],
             note="log observed on log predicted"),
        _row("q8", "calibration_intercept_time_domain",
             q8["time_domain_intercept"], note="target 0"),
        _row("q8", "median_absolute_error", q8["median_absolute_error_s"],
             unit="s"),
        _row("q8", "mean_absolute_error", q8["mean_absolute_error_s"],
             unit="s"),
        _row("q8", "mean_signed_bias", q8["mean_signed_bias_s"], unit="s",
             note="positive = predicted later than observed"),
    ]

    ph = r["diagnostics"]["proportional_hazards"]
    rows += [
        _row("diagnostics", "schoenfeld_global_p", ph["global_p"],
             note=f"df = {ph['global_df']}; "
                  f"violated: {ph['violated_features'] or 'none'}"),
        _row("diagnostics", "ph_assumption_violated",
             int(ph["assumption_violated"]),
             note=r["diagnostics"]["model_switch"]["decision"]),
    ]

    return pd.DataFrame(rows)


def write_csv(r: dict, path):
    tab = build_metrics_table(r)
    tab.to_csv(path, index=False)
    return path
