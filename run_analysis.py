"""Entry point: run the whole analysis and write the two output artefacts.

    python run_analysis.py                 # full run, 1000 permutations
    python run_analysis.py --permutations 50   # quick smoke run
    python run_analysis.py --no-permutations
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")

from pte import config as C          # noqa: E402
from pte import cv, data, inference, permutation, pipeline, plots, report  # noqa: E402
from pte import questions as Q       # noqa: E402


def _log(msg, t0=None):
    stamp = f"[{time.strftime('%H:%M:%S')}]"
    extra = f"  ({time.time() - t0:.1f}s)" if t0 else ""
    print(f"{stamp} {msg}{extra}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=C.N_PERMUTATIONS)
    ap.add_argument("--no-permutations", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=C.N_BOOTSTRAP)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)

    C.N_BOOTSTRAP = args.bootstrap
    C.RESULTS_DIR.mkdir(exist_ok=True)
    n_perm = 0 if args.no_permutations else args.permutations

    t_start = time.time()
    pipeline.reset_audit()

    # ---------------------------------------------------------- data
    _log("loading and validating data")
    df = data.load()
    cohort = data.design_summary(df)
    _log(f"  n = {cohort['n_larvae']}, events = {cohort['n_events']}, "
         f"censored = {cohort['n_censored']}")
    _log(f"  {cohort['epv_flag']}")

    # ------------------------------------------------- cross-validation
    _log("leave-one-fish-out CV, M0 (pressure only)")
    t0 = time.time()
    cv_m0 = cv.leave_one_fish_out(df, C.M0_FEATURES, tag="M0")
    _log(f"  c-index = {cv_m0.c_index:.4f}", t0)

    _log("leave-one-fish-out CV, M1 (pressure + LFP)")
    t0 = time.time()
    cv_m1 = cv.leave_one_fish_out(df, C.M1_FEATURES, tag="M1")
    _log(f"  c-index = {cv_m1.c_index:.4f}", t0)

    modal_pen = Counter(cv_m1.chosen_penalizer).most_common(1)[0][0]
    modal_l1 = Counter(cv_m1.chosen_l1_ratio).most_common(1)[0][0]
    _log(f"  modal hyperparameters across folds: penalizer = {modal_pen}, "
         f"l1_ratio = {modal_l1}")

    # ----------------------------------------------- reported inference
    _log("final model, hazard ratios with clutch-bootstrap CIs")
    t0 = time.time()
    hr_table, boot_info = inference.hazard_ratios(
        df, C.M1_FEATURES, modal_pen, modal_l1, n_boot=C.N_BOOTSTRAP)
    hr_unpen, _ = inference.hazard_ratios_unpenalized(df, C.M1_FEATURES)
    _log(f"  {boot_info['n_bootstrap_ok']}/{boot_info['n_bootstrap']} "
         f"bootstrap refits converged", t0)

    _log("proportional hazards check (Schoenfeld residuals)")
    ph = inference.schoenfeld(df, C.M1_FEATURES)
    _log(f"  global p = {ph['global_p']:.4f}; "
         f"violated: {ph['violated_features'] or 'none'}")

    aft_table, aft_model, switch = None, None, {}
    if ph["assumption_violated"]:
        _log("  PH VIOLATED -> refitting as Weibull AFT (logged switch)")
        aft_table, aft_model = inference.weibull_aft(df, C.M1_FEATURES)
        switch = {
            "switched": True,
            "from": "Cox proportional hazards",
            "to": "Weibull accelerated failure time",
            "trigger": f"Schoenfeld global p = {ph['global_p']:.4f} "
                       f"< {C.PH_ALPHA}",
            "decision": "SWITCHED to Weibull AFT",
        }
    else:
        _log("  PH holds globally -> Cox retained; "
             "Weibull AFT still fitted as a sensitivity check")
        aft_table, aft_model = inference.weibull_aft(df, C.M1_FEATURES)
        switch = {
            "switched": False,
            "model_used": "Cox proportional hazards",
            "trigger": f"Schoenfeld global p = {ph['global_p']:.4f} "
                       f">= {C.PH_ALPHA}",
            "decision": "NOT switched; Cox retained",
            "note": (
                "Per-covariate flags: "
                f"{ph['violated_features'] or 'none'}. The Weibull AFT fit is "
                "reported alongside as a sensitivity analysis only."
            ),
        }

    # ------------------------------------------------------ permutation
    null_m0 = null_m1 = None
    if n_perm > 0:
        _log(f"permutation null, M1, {n_perm} end-to-end refits "
             f"(this is the slow step)")
        t0 = time.time()
        null_m1 = permutation.run(df, C.M1_FEATURES, cv_m1.c_index,
                                  n_iter=n_perm, n_workers=args.workers,
                                  label="M1")
        _log(f"  null mean = {null_m1['null_mean']:.4f}, "
             f"p = {null_m1['p_value']:.4f}", t0)

        _log(f"permutation null, M0, {n_perm} end-to-end refits")
        t0 = time.time()
        null_m0 = permutation.run(df, C.M0_FEATURES, cv_m0.c_index,
                                  n_iter=n_perm, n_workers=args.workers,
                                  label="M0")
        _log(f"  null mean = {null_m0['null_mean']:.4f}, "
             f"p = {null_m0['p_value']:.4f}", t0)

    # ------------------------------------------------- the 8 questions
    _log("answering the eight biological questions")
    results = {"cohort": cohort, "qc": data.qc_summary(df)}

    results["q1_lfp_adds_over_pressure"] = Q.q1_lfp_adds_over_pressure(
        df, cv_m0, cv_m1, null_m0, null_m1)
    results["q2_direction_check"] = Q.q2_direction_check(hr_table, hr_unpen)
    results["q3_critical_slowing_agreement"] = Q.q3_critical_slowing_agreement(
        df, hr_table, hr_unpen)
    results["q4_within_stratum"] = Q.q4_within_stratum(df, cv_m1)
    results["q5_injury_gradient"] = Q.q5_injury_gradient(df, cv_m1)
    results["q6_behaviour_vs_dose"] = Q.q6_behaviour_vs_dose(df)
    results["q7_leave_one_stratum_out"] = Q.q7_leave_one_stratum_out(df)
    results["q8_calibration"] = Q.q8_calibration(df, cv_m1)

    # ------------------------------------------------------ diagnostics
    violations = pipeline.audit_violations()
    results["diagnostics"] = {
        "proportional_hazards": ph,
        "model_switch": switch,
        "weibull_aft_sensitivity": aft_table.to_dict("records"),
        "frailty_substitution": inference.FRAILTY_NOTE,
        "hazard_ratios_elastic_net": hr_table.to_dict("records"),
        "hazard_ratios_unpenalized": hr_unpen.to_dict("records"),
        "hyperparameters": {
            "modal_penalizer": float(modal_pen),
            "modal_l1_ratio": float(modal_l1),
            "penalizer_by_fold": cv_m1.chosen_penalizer.tolist(),
            "l1_ratio_by_fold": cv_m1.chosen_l1_ratio.tolist(),
            "selection_frequency_across_folds": cv_m1.selection_counts,
        },
        "leakage_audit": {
            "n_fit_predict_pairs_recorded": len(pipeline.AUDIT_LOG),
            "n_violations": len(violations),
            "clean": len(violations) == 0,
            "description": (
                "Every FoldPipeline fit/predict pair is recorded. A violation "
                "is any prediction whose rows intersect its own training "
                "fold. See tests/test_leakage.py for the sentinel canary."
            ),
        },
        "compositional_constraint": {
            "features": inference.COMPOSITIONAL_BANDS,
            "sum_min": float(df[inference.COMPOSITIONAL_BANDS].sum(1).min()),
            "sum_max": float(df[inference.COMPOSITIONAL_BANDS].sum(1).max()),
            "reference_dropped_for_unpenalized_fits":
                inference.REFERENCE_BAND,
        },
    }

    # ---------------------------------------------------------- outputs
    _log("writing outputs")
    json_path = report.write_json(results, C.RESULTS_DIR / "results.json")
    csv_path = report.write_csv(results, C.RESULTS_DIR / "metrics.csv")

    fig_paths = []
    nulls = [n for n in (null_m1, null_m0) if n]
    if nulls:
        fig_paths.append(plots.permutation_null(
            nulls, C.RESULTS_DIR / "fig_null_distribution.png"))
    fig_paths.append(plots.calibration(
        results["q8_calibration"], cv_m1,
        C.RESULTS_DIR / "fig_calibration.png"))

    # ------------------------------------------------------- to console
    print("\n" + "=" * 78)
    print("ANSWERS")
    print("=" * 78)
    titles = {
        "q1_lfp_adds_over_pressure":
            "1. Does the brain's own signal add anything to the hit?",
        "q2_direction_check": "2. Do the retained features point the right way?",
        "q3_critical_slowing_agreement":
            "3. Do the two critical slowing features agree?",
        "q4_within_stratum":
            "4. Do two larvae that took the same hit differ predictably?",
        "q5_injury_gradient":
            "5. Does predicted susceptibility follow the injury gradient?",
        "q6_behaviour_vs_dose":
            "6. Is the behavioural signal just re-reporting injury dose?",
        "q7_leave_one_stratum_out":
            "7. Does it hold for an injury level it has never seen?",
        "q8_calibration": "8. Are the predicted latencies right in seconds?",
    }
    for key, title in titles.items():
        print(f"\n{title}")
        print("   " + results[key]["answer"].replace("\n", " "))

    print("\n" + "=" * 78)
    print(f"events per predictor (M1): {cohort['events_per_predictor_M1']}"
          f"  -> {'FLAGGED' if cohort['epv_M1_below_threshold'] else 'ok'}")
    print(f"leakage audit: {len(pipeline.AUDIT_LOG)} fit/predict pairs, "
          f"{len(violations)} violations")
    print(f"proportional hazards: global p = {ph['global_p']:.4f} -> "
          f"{switch['decision']}")
    print("=" * 78)
    for p in [json_path, csv_path] + fig_paths:
        print(f"  wrote {p}")
    _log(f"done in {time.time() - t_start:.1f}s")

    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
