"""The fold-local pipeline and the leakage guard.

Everything that learns from data -- centring, scaling, hyperparameter
selection, and the L1 feature selection itself -- happens inside
`FoldPipeline.fit`, which is only ever handed training-fold rows.

Every fit/predict pair is recorded in a module-level audit log, and
`predict` refuses by default to score rows that were part of its own
training fold. `tests/test_leakage.py` reads the audit log and also runs
a sentinel-corruption canary against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import coxnet


class LeakageError(AssertionError):
    """Raised when a transform would see data outside its training fold."""


# --------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------
@dataclass
class AuditRecord:
    tag: str
    fit_index: tuple
    predict_index: tuple
    allow_overlap: bool


AUDIT_LOG: list[AuditRecord] = []


def reset_audit() -> None:
    AUDIT_LOG.clear()


def audit_violations() -> list[AuditRecord]:
    """Records where a prediction overlapped its own training fold.

    Entries flagged `allow_overlap` are the deliberate in-sample calls
    (the final full-data model reported in the hazard-ratio table), and
    are excluded.
    """
    bad = []
    for r in AUDIT_LOG:
        if r.allow_overlap:
            continue
        if set(r.fit_index) & set(r.predict_index):
            bad.append(r)
    return bad


# --------------------------------------------------------------------
# Fold-local pipeline
# --------------------------------------------------------------------
@dataclass
class FoldPipeline:
    """Scale -> tune -> select -> fit, all confined to one training fold."""

    features: list
    penalizer_grid: list = field(default_factory=lambda: list(C.PENALIZER_GRID))
    l1_grid: list = field(default_factory=lambda: list(C.L1_RATIO_GRID))
    inner_folds: int = C.INNER_FOLDS
    unpenalized: set = field(default_factory=lambda: set(C.UNPENALIZED))
    use_clutch_strata: bool = True
    seed: int = C.SEED
    tag: str = "cv"

    # learned state
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    fit_index_: tuple = ()
    model_: coxnet.CoxnetFit | None = None
    best_penalizer_: float | None = None
    best_l1_ratio_: float | None = None
    inner_score_: float | None = None

    # ---------------------------------------------------------- helpers
    def _penalty_weights(self) -> np.ndarray:
        return np.array(
            [0.0 if f in self.unpenalized else 1.0 for f in self.features]
        )

    def _matrix(self, df, index) -> np.ndarray:
        return df.loc[index, self.features].to_numpy(dtype=np.float64)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    @staticmethod
    def _strata_codes(df, index, use: bool):
        if not use:
            return None
        return df.loc[index, C.CLUTCH_COL].to_numpy()

    # -------------------------------------------------------------- fit
    def fit(self, df, index) -> "FoldPipeline":
        index = list(index)
        self.fit_index_ = tuple(index)

        X = self._matrix(df, index)

        # Scaling constants come from the training fold and nowhere else.
        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0)
        sd[sd < 1e-12] = 1.0            # constant-in-fold column stays at 0
        self.scale_ = sd
        Z = self._scale(X)

        t = df.loc[index, C.DURATION_COL].to_numpy(float)
        e = df.loc[index, C.EVENT_COL].to_numpy(float)
        strata = self._strata_codes(df, index, self.use_clutch_strata)
        w = self._penalty_weights()

        self.best_penalizer_, self.best_l1_ratio_, self.inner_score_ = \
            self._tune(Z, t, e, strata, w)

        self.model_ = coxnet.fit(
            Z, t, e, self.features, self.best_penalizer_, self.best_l1_ratio_,
            penalty_weights=w, strata=strata, compute_baseline=True,
        )
        return self

    # ------------------------------------------------------------- tune
    def _tune(self, Z, t, e, strata, w):
        """Inner cross-validated partial-likelihood tuning.

        Scored by the cross-validated partial likelihood deviance
        (Verweij & van Houwelingen): the difference between the log
        partial likelihood of the full inner set and that of the inner
        training part, evaluated at the training coefficients. This is
        the standard criterion for penalized Cox and, unlike a held-out
        concordance, stays defined when an inner fold contains few
        events.
        """
        n = len(t)
        if len(self.penalizer_grid) == 1 and len(self.l1_grid) == 1:
            return self.penalizer_grid[0], self.l1_grid[0], float("nan")

        rng = np.random.default_rng(self.seed + n)
        folds = self._event_balanced_folds(e, self.inner_folds, rng)

        # Risk sets depend only on times, events and strata -- none of
        # which change across the tuning grid -- so build them once per
        # inner fold and reuse them for all 24 hyperparameter settings.
        def codes_of(st, idx):
            if st is None:
                return np.zeros(len(idx), dtype=np.int64)
            return np.unique(st[idx], return_inverse=True)[1]

        allidx = np.arange(n)
        rs_full = coxnet.RiskSets.build(t, e, codes_of(strata, allidx))

        prepared = []
        for test_idx in folds:
            tr = np.setdiff1d(allidx, test_idx)
            if e[tr].sum() < 2 or e[test_idx].sum() < 1:
                continue
            prepared.append((
                tr,
                None if strata is None else strata[tr],
                coxnet.RiskSets.build(t[tr], e[tr], codes_of(strata, tr)),
            ))

        if not prepared:
            return self.penalizer_grid[len(self.penalizer_grid) // 2], \
                self.l1_grid[0], float("nan")

        best = (None, None, -np.inf)
        for l1 in self.l1_grid:
            # Track a warm start down the penalty path within each fold.
            warm = {k: None for k in range(len(prepared))}
            scores = {pen: 0.0 for pen in self.penalizer_grid}
            usable = {pen: 0 for pen in self.penalizer_grid}

            for pen in sorted(self.penalizer_grid, reverse=True):
                for k, (tr, st_tr, rs_tr) in enumerate(prepared):
                    f = coxnet.fit(
                        Z[tr], t[tr], e[tr], self.features, pen, l1,
                        penalty_weights=w, strata=st_tr,
                        warm_start=warm[k], max_iter=200, risk_sets=rs_tr,
                    )
                    warm[k] = f.beta.copy()
                    scores[pen] += self._cv_partial_loglik(
                        f.beta, Z, t, rs_full, rs_tr, tr)
                    usable[pen] += 1

            for pen in self.penalizer_grid:
                if usable[pen] == 0:
                    continue
                s = scores[pen] / usable[pen]
                if s > best[2]:
                    best = (pen, l1, s)

        if best[0] is None:                       # degenerate fold
            return self.penalizer_grid[len(self.penalizer_grid) // 2], \
                self.l1_grid[0], float("nan")
        return best[0], best[1], float(best[2])

    @staticmethod
    def _cv_partial_loglik(beta, Z, t, rs_full, rs_train, train_idx):
        """ll(full inner set | beta) - ll(inner training part | beta).

        Verweij & van Houwelingen's cross-validated partial likelihood.
        Both risk-set structures are passed in prebuilt.
        """
        ll_full = -coxnet._nll_grad_hess(
            beta, Z[rs_full.order], rs_full, want_hess=False)[0]
        Ztr = Z[train_idx]
        ll_train = -coxnet._nll_grad_hess(
            beta, Ztr[rs_train.order], rs_train, want_hess=False)[0]
        return ll_full - ll_train

    @staticmethod
    def _event_balanced_folds(e, k, rng):
        """Split so every inner fold carries a share of the events."""
        folds = [[] for _ in range(k)]
        for group in (np.flatnonzero(e == 1), np.flatnonzero(e == 0)):
            g = rng.permutation(group)
            for i, idx in enumerate(g):
                folds[i % k].append(idx)
        return [np.array(sorted(f)) for f in folds if len(f)]

    # ---------------------------------------------------------- predict
    def predict(self, df, index, allow_overlap: bool = False) -> np.ndarray:
        """Linear predictor for held-out rows, with the leakage guard."""
        index = list(index)
        AUDIT_LOG.append(AuditRecord(
            tag=self.tag, fit_index=tuple(self.fit_index_),
            predict_index=tuple(index), allow_overlap=allow_overlap,
        ))
        if not allow_overlap:
            overlap = set(self.fit_index_) & set(index)
            if overlap:
                raise LeakageError(
                    f"[{self.tag}] prediction rows {sorted(overlap)} were part "
                    f"of this pipeline's own training fold"
                )
        Z = self._scale(self._matrix(df, index))
        return self.model_.predict_linear(Z)

    def predict_median_time(self, df, index, allow_overlap: bool = False):
        """Predicted median latency in seconds for held-out rows."""
        eta_guard = self.predict(df, index, allow_overlap=allow_overlap)  # audits
        del eta_guard
        Z = self._scale(self._matrix(df, index))
        strata = (df.loc[index, C.CLUTCH_COL].to_numpy()
                  if self.use_clutch_strata else None)
        return self.model_.predict_median(Z, strata)

    # --------------------------------------------------------- fingerprint
    def fingerprint(self) -> dict:
        """State learned from the training fold, for the canary test."""
        return {
            "mean": np.round(self.mean_, 12).tolist(),
            "scale": np.round(self.scale_, 12).tolist(),
            "penalizer": self.best_penalizer_,
            "l1_ratio": self.best_l1_ratio_,
            "selected": tuple(self.model_.selected),
            "beta": np.round(self.model_.beta, 12).tolist(),
        }
