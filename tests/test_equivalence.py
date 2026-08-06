"""The fast solver must agree with lifelines.

`pte.coxnet` exists only because lifelines is too slow to permit 1000
end-to-end permutation refits. That trade is only legitimate if the two
give the same answer, so this asserts it across the whole tuning grid,
with and without clutch strata.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")

from pte import config as C
from pte import coxnet, data

TOL = 0.01          # absolute, on the log-hazard scale


@pytest.fixture(scope="module")
def setup():
    df = data.load()
    feats = C.M1_FEATURES
    X = df[feats].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    return {
        "df": df, "feats": feats, "X": X,
        "t": df[C.DURATION_COL].to_numpy(float),
        "e": df[C.EVENT_COL].to_numpy(float),
        "clutch": df[C.CLUTCH_COL].to_numpy(),
        "w": np.array([0.0 if f in C.UNPENALIZED else 1.0 for f in feats]),
    }


def _lifelines(setup, pen, l1, strata):
    d = pd.DataFrame(setup["X"], columns=setup["feats"])
    d["T"], d["E"] = setup["t"], setup["e"]
    kw = {}
    if strata:
        d["clutch"] = setup["clutch"]
        kw["strata"] = "clutch"
    m = CoxPHFitter(penalizer=pen * setup["w"] if pen > 0 else 0.0,
                    l1_ratio=l1)
    m.fit(d, "T", "E", **kw)
    return m.params_.reindex(setup["feats"]).to_numpy()


@pytest.mark.parametrize("pen", C.PENALIZER_GRID)
@pytest.mark.parametrize("l1", C.L1_RATIO_GRID)
def test_matches_lifelines_unstratified(setup, pen, l1):
    mine = coxnet.fit(setup["X"], setup["t"], setup["e"], setup["feats"],
                      pen, l1, penalty_weights=setup["w"]).beta
    theirs = _lifelines(setup, pen, l1, strata=False)
    assert np.max(np.abs(mine - theirs)) < TOL, (
        f"pen={pen} l1={l1}: max coefficient difference "
        f"{np.max(np.abs(mine - theirs)):.5f}"
    )


@pytest.mark.parametrize("pen", [0.01, 0.03, 0.1, 0.3, 1.0])
@pytest.mark.parametrize("l1", [0.1, 0.5, 1.0])
def test_matches_lifelines_stratified_by_clutch(setup, pen, l1):
    mine = coxnet.fit(setup["X"], setup["t"], setup["e"], setup["feats"],
                      pen, l1, penalty_weights=setup["w"],
                      strata=setup["clutch"]).beta
    theirs = _lifelines(setup, pen, l1, strata=True)
    assert np.max(np.abs(mine - theirs)) < TOL


def test_penalizer_convention_matches_lifelines(setup):
    """lifelines penalizes the MEAN log-likelihood; we rescale internally.

    If that factor of n were dropped, a mid-grid penalty would shrink
    far too little and this comparison would blow up.
    """
    pen, l1 = 0.1, 1.0
    mine = coxnet.fit(setup["X"], setup["t"], setup["e"], setup["feats"],
                      pen, l1, penalty_weights=setup["w"]).beta
    theirs = _lifelines(setup, pen, l1, strata=False)
    # lifelines leaves shrunk coefficients at ~1e-9 rather than exact zero,
    # so compare sparsity at a tolerance rather than by exact count.
    n_mine = int(np.sum(np.abs(mine) > 1e-6))
    n_theirs = int(np.sum(np.abs(theirs) > 1e-6))
    assert abs(n_mine - n_theirs) <= 1, (
        f"sparsity disagrees: {n_mine} vs {n_theirs} non-zero coefficients"
    )
    assert np.max(np.abs(mine - theirs)) < TOL


def test_unpenalized_covariate_is_never_shrunk_to_zero(setup):
    j = setup["feats"].index(C.PRESSURE)
    for pen in (0.3, 1.0, 3.0):
        f = coxnet.fit(setup["X"], setup["t"], setup["e"], setup["feats"],
                       pen, 1.0, penalty_weights=setup["w"])
        assert f.beta[j] != 0.0, (
            f"max_pressure_kPa was zeroed at penalizer={pen} despite carrying "
            f"zero penalty weight"
        )


def test_l1_actually_selects(setup):
    weak = coxnet.fit(setup["X"], setup["t"], setup["e"], setup["feats"],
                      0.001, 1.0, penalty_weights=setup["w"])
    strong = coxnet.fit(setup["X"], setup["t"], setup["e"], setup["feats"],
                        1.0, 1.0, penalty_weights=setup["w"])
    assert len(strong.selected) < len(weak.selected)


def test_baseline_hazard_gives_sane_median_predictions(setup):
    f = coxnet.fit(setup["X"], setup["t"], setup["e"], setup["feats"],
                   0.03, 0.5, penalty_weights=setup["w"],
                   strata=setup["clutch"], compute_baseline=True)
    med = f.predict_median(setup["X"], setup["clutch"])
    finite = med[np.isfinite(med)]
    assert len(finite) > 50
    assert finite.min() > 0
    assert finite.max() <= setup["t"].max() * 1.01


def test_solver_converges_across_the_grid(setup):
    for pen in C.PENALIZER_GRID:
        for l1 in C.L1_RATIO_GRID:
            f = coxnet.fit(setup["X"], setup["t"], setup["e"], setup["feats"],
                           pen, l1, penalty_weights=setup["w"],
                           strata=setup["clutch"])
            assert f.converged, f"no convergence at pen={pen}, l1={l1}"
