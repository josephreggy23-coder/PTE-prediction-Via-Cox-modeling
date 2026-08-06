"""Elastic-net regularized Cox proportional hazards, in NumPy.

Why this exists instead of just calling lifelines everywhere: the
evaluation plan requires 1000 permutations, each one a full end-to-end
refit of a leave-one-fish-out cross-validation with nested tuning. A
lifelines penalized fit takes ~190 ms (~460 ms stratified), which puts a
single permutation at ~38 minutes and the full null at several weeks.

This solver does the same job in ~1-2 ms by using a proximal Newton
scheme (exact Breslow gradient and Hessian, then cyclic coordinate
descent on the penalized quadratic, then a backtracking line search).

lifelines remains the authority for everything *reported* -- hazard
ratios, confidence intervals, Schoenfeld residuals, the Weibull AFT
fallback. `tests/test_equivalence.py` asserts the two agree on
coefficients, so the fast path is validated rather than trusted.

Conventions match lifelines, which applies `penalizer` to the MEAN log
partial likelihood. In total-likelihood terms that is:

    objective(beta) = -loglik_breslow(beta)
                      + n * penalizer * ( l1_ratio * sum_j w_j |beta_j|
                                        + 0.5 (1-l1_ratio) sum_j w_j beta_j^2 )

with w_j = 0 for covariates named in the unpenalized set. The factor of
n is applied internally, so `penalizer` here means exactly what it means
in lifelines. This was established empirically, not assumed; see
`tests/test_equivalence.py`.

One data-specific caution baked in here: lfp_power_{low,mid,high}_norm
sum to 1.000 +/- 0.0007, so the M1 design is only full rank by virtue of
rounding noise. With no penalty at all the solver chases that noise and
the band coefficients diverge. Any unpenalized fit must therefore drop
one band as a compositional reference -- see inference.reference_free().
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C

# Inner coordinate-descent budget for the proximal-Newton subproblem.
CD_MAX_SWEEPS = 10
CD_TOL = 1e-7


# --------------------------------------------------------------------
# Risk-set bookkeeping, precomputed once per dataset
# --------------------------------------------------------------------
@dataclass
class RiskSets:
    """Sort order and tie structure for Breslow risk sets, per stratum.

    Times never change during optimization, so all of this is computed
    once and reused across every iteration and every point on the
    penalty grid.
    """

    order: np.ndarray          # indices sorting rows by time, ascending
    event: np.ndarray          # event flags in sorted order
    tie_start: np.ndarray      # for each position, first position of its tie group
    stratum_bounds: list       # (lo, hi) slices into the sorted arrays
    n: int
    # Vectorization aids: strata occupy contiguous blocks of the sorted
    # arrays, so a single global suffix-cumsum minus the value at the
    # block's end recovers every within-stratum risk-set total at once.
    block_end: np.ndarray = None    # per position, the end of its stratum block
    ev_pos: np.ndarray = None       # positions of events
    ev_head: np.ndarray = None      # tie-group head for each event position
    ev_tail: np.ndarray = None      # stratum end for each event position

    @staticmethod
    def build(time: np.ndarray, event: np.ndarray,
              strata: np.ndarray | None = None) -> "RiskSets":
        n = len(time)
        if strata is None:
            strata = np.zeros(n, dtype=np.int64)

        order_parts, bounds, pos = [], [], 0
        for s in np.unique(strata):
            idx = np.flatnonzero(strata == s)
            idx = idx[np.argsort(time[idx], kind="stable")]
            order_parts.append(idx)
            bounds.append((pos, pos + len(idx)))
            pos += len(idx)

        order = np.concatenate(order_parts) if order_parts else np.array([], int)
        t_sorted = time[order]
        e_sorted = event[order].astype(np.float64)

        # First position of each tie group, computed within stratum.
        # Vectorized: this is built once per fold-set and reused across the
        # whole tuning grid, and a Python loop here was measurably the
        # second-largest cost in the permutation run.
        tie_start = np.arange(n, dtype=np.int64)
        for lo, hi in bounds:
            seg = t_sorted[lo:hi]
            if len(seg) < 2:
                continue
            is_new = np.ones(len(seg), dtype=bool)
            is_new[1:] = seg[1:] != seg[:-1]
            pos = np.arange(lo, hi)
            tie_start[lo:hi] = np.maximum.accumulate(np.where(is_new, pos, -1))

        block_end = np.empty(n, dtype=np.int64)
        for lo, hi in bounds:
            block_end[lo:hi] = hi

        # A stratum with no events contributes nothing to the likelihood.
        live = np.zeros(n, dtype=bool)
        for lo, hi in bounds:
            if e_sorted[lo:hi].sum() > 0:
                live[lo:hi] = True

        ev_pos = np.flatnonzero((e_sorted > 0) & live)
        return RiskSets(
            order, e_sorted, tie_start, bounds, n,
            block_end=block_end,
            ev_pos=ev_pos,
            ev_head=tie_start[ev_pos],
            ev_tail=block_end[ev_pos],
        )

    def stability_shift(self, eta: np.ndarray) -> np.ndarray:
        """Per-stratum max of eta, broadcast to every position.

        Subtracting this keeps every exp() in (0, 1] and, crucially,
        puts all strata on the same numerical scale so the global
        suffix-cumsum trick does not lose precision.
        """
        shift = np.empty_like(eta)
        for lo, hi in self.stratum_bounds:
            shift[lo:hi] = eta[lo:hi].max() if hi > lo else 0.0
        return shift


# --------------------------------------------------------------------
# Breslow partial likelihood: value, gradient, Hessian
# --------------------------------------------------------------------
def _nll_grad_hess(beta: np.ndarray, Xs: np.ndarray, rs: RiskSets,
                   want_hess: bool = True):
    """Negative Breslow log partial likelihood and its derivatives.

    Xs is the design matrix already permuted into risk-set sort order.
    """
    n, p = Xs.shape
    ev = rs.ev_pos
    if len(ev) == 0:
        return 0.0, np.zeros(p), (np.zeros((p, p)) if want_hess else None)

    eta = Xs @ beta
    shift = rs.stability_shift(eta)
    w = np.exp(eta - shift)

    def suffix(a):
        """Within-stratum suffix sums for every position, in one pass."""
        g = np.cumsum(a[::-1], axis=0)[::-1]
        return np.concatenate([g, np.zeros((1,) + a.shape[1:])], axis=0)

    g0 = suffix(w)
    wx = w[:, None] * Xs
    g1 = suffix(wx)

    head, tail = rs.ev_head, rs.ev_tail
    S0 = g0[head] - g0[tail]
    S1 = g1[head] - g1[tail]

    nll = -float(np.sum(eta[ev] - shift[ev] - np.log(S0)))

    mean_x = S1 / S0[:, None]
    grad = -np.sum(Xs[ev] - mean_x, axis=0)

    hess = None
    if want_hess:
        # Risk-set second moment. n*p*p is ~12k doubles here, negligible.
        g2 = suffix(wx[:, :, None] * Xs[:, None, :])
        S2 = g2[head] - g2[tail]
        hess = np.einsum(
            "ijk->jk", S2 / S0[:, None, None]
        ) - np.einsum("ij,ik->jk", mean_x, mean_x)

    return nll, grad, hess


def _penalty(beta, penalizer, l1_ratio, weights):
    """Penalty on the total-likelihood scale; caller pre-multiplies by n."""
    return penalizer * (
        l1_ratio * float(np.sum(weights * np.abs(beta)))
        + 0.5 * (1.0 - l1_ratio) * float(np.sum(weights * beta**2))
    )


def _soft(z: float, t: float) -> float:
    """Soft threshold, in plain Python floats.

    This is called on the order of a thousand times per fit, so it must
    not touch NumPy scalars -- doing so was, by profiling, the single
    largest cost in the whole solver.
    """
    if z > t:
        return z - t
    if z < -t:
        return z + t
    return 0.0


# --------------------------------------------------------------------
# Fitted model
# --------------------------------------------------------------------
@dataclass
class CoxnetFit:
    beta: np.ndarray
    feature_names: list
    penalizer: float
    l1_ratio: float
    loglik: float
    converged: bool
    n_iter: int
    # Breslow baseline cumulative hazard per stratum, filled by _baseline
    baseline: dict = field(default_factory=dict)

    def predict_linear(self, X: np.ndarray) -> np.ndarray:
        """Linear predictor. Higher value = higher hazard = shorter latency."""
        return X @ self.beta

    @property
    def selected(self) -> list:
        return [n for n, b in zip(self.feature_names, self.beta) if b != 0.0]

    def predict_median(self, X: np.ndarray,
                       strata: np.ndarray | None = None) -> np.ndarray:
        """Predicted median latency in seconds, from the Breslow baseline.

        Returns np.inf where the predicted survival curve never crosses
        0.5 within observed follow-up -- that larva is predicted not to
        reach stage III before the censoring horizon, which is a real
        prediction and must not be silently clipped.
        """
        eta = self.predict_linear(X)
        out = np.empty(len(X))
        if strata is None:
            strata = np.zeros(len(X), dtype=np.int64)

        for i in range(len(X)):
            key = strata[i]
            if key not in self.baseline:
                key = next(iter(self.baseline))
            times, cumhaz = self.baseline[key]
            surv = np.exp(-cumhaz * np.exp(eta[i]))
            below = np.flatnonzero(surv <= 0.5)
            out[i] = times[below[0]] if len(below) else np.inf
        return out


def _baseline(beta, X, time, event, strata):
    """Breslow baseline cumulative hazard, computed per stratum."""
    out = {}
    eta = X @ beta
    for s in np.unique(strata):
        m = strata == s
        t, e, w = time[m], event[m], np.exp(eta[m])
        order = np.argsort(t, kind="stable")
        t, e, w = t[order], e[order], w[order]
        risk = np.cumsum(w[::-1])[::-1]
        # One increment per unique event time.
        ut = np.unique(t[e == 1])
        inc = np.array([
            (e[t == u].sum()) / risk[np.flatnonzero(t >= u)[0]] for u in ut
        ])
        out[s] = (ut, np.cumsum(inc))
    return out


# --------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------
def fit(X: np.ndarray, time: np.ndarray, event: np.ndarray,
        feature_names: list, penalizer: float, l1_ratio: float,
        penalty_weights: np.ndarray | None = None,
        strata: np.ndarray | None = None,
        warm_start: np.ndarray | None = None,
        max_iter: int = C.MAX_ITER, tol: float = C.TOL,
        compute_baseline: bool = False,
        risk_sets: "RiskSets | None" = None) -> CoxnetFit:
    """Proximal-Newton elastic-net Cox fit.

    `risk_sets` may be supplied pre-built. Times and strata do not change
    across a tuning grid, so building them once per fold and reusing them
    across all 24 hyperparameter combinations is a large saving in the
    permutation loop.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    n, p = X.shape

    if penalty_weights is None:
        penalty_weights = np.ones(p)
    w = np.asarray(penalty_weights, dtype=np.float64)

    strata_arr = (np.zeros(n, dtype=np.int64) if strata is None
                  else np.asarray(strata))
    codes = np.unique(strata_arr, return_inverse=True)[1]

    rs = risk_sets if risk_sets is not None else RiskSets.build(
        time, event, codes)
    Xs = X[rs.order]

    # lifelines penalizes the mean log-likelihood; rescale to the total.
    pen_n = penalizer * n

    beta = (np.zeros(p) if warm_start is None
            else np.array(warm_start, dtype=np.float64))

    obj = _nll_grad_hess(beta, Xs, rs, want_hess=False)[0] + _penalty(
        beta, pen_n, l1_ratio, w)
    converged, it = False, 0

    for it in range(1, max_iter + 1):
        nll, grad, hess = _nll_grad_hess(beta, Xs, rs, want_hess=True)
        hess = hess + np.eye(p) * 1e-10        # keep diagonal strictly positive

        # --- cyclic coordinate descent on the penalized quadratic ----
        # The subproblem is solved inexactly on purpose: proximal Newton
        # tolerates a loose inner solve, and the outer line search plus
        # the next Newton step clean up the remainder. Exact solves here
        # cost far more than the extra outer iteration they save.
        b = beta.copy()
        hb = np.zeros(p)                        # H @ (b - beta), kept current
        hdiag = np.diag(hess).copy()
        denom_l2 = hdiag + pen_n * (1.0 - l1_ratio) * w
        thresh = pen_n * l1_ratio * w

        # Pure-Python views: p is ~11, so NumPy per-element overhead
        # dominates real arithmetic in this loop.
        hcols = [hess[:, j].tolist() for j in range(p)]
        gl, hdl, dl2, thl = (grad.tolist(), hdiag.tolist(),
                             denom_l2.tolist(), thresh.tolist())
        bl = b.tolist()
        hbl = [0.0] * p
        rng_p = range(p)

        for _ in range(CD_MAX_SWEEPS):
            max_delta = 0.0
            for j in rng_p:
                r = hbl[j] - hdl[j] * bl[j] + gl[j]
                new = (_soft(-r, thl[j]) / dl2[j] if thl[j] > 0.0
                       else -r / dl2[j])
                d = new - bl[j]
                if d != 0.0:
                    col = hcols[j]
                    for k in rng_p:
                        hbl[k] += col[k] * d
                    bl[j] = new
                    if abs(d) > max_delta:
                        max_delta = abs(d)
            if max_delta < CD_TOL:
                break
        b = np.array(bl)

        # --- backtracking line search on the true objective -----------
        step = beta.copy()
        direction = b - beta
        t_step, improved = 1.0, False
        for _ in range(30):
            cand = beta + t_step * direction
            cand_obj = _nll_grad_hess(cand, Xs, rs, want_hess=False)[0] + \
                _penalty(cand, pen_n, l1_ratio, w)
            if np.isfinite(cand_obj) and cand_obj <= obj + 1e-12:
                step, improved = cand, True
                break
            t_step *= 0.5

        if not improved:
            converged = True
            break

        delta_obj = obj - cand_obj
        beta, obj = step, cand_obj
        if delta_obj < tol * max(1.0, abs(obj)):
            converged = True
            break

    # Exact-zero coefficients where L1 shrank them below numerical noise.
    beta[np.abs(beta) < 1e-9] = 0.0
    loglik = -_nll_grad_hess(beta, Xs, rs, want_hess=False)[0]

    out = CoxnetFit(beta=beta, feature_names=list(feature_names),
                    penalizer=penalizer, l1_ratio=l1_ratio,
                    loglik=float(loglik), converged=converged, n_iter=it)
    if compute_baseline:
        out.baseline = _baseline(beta, X, time, event, codes)
    return out


def fit_path(X, time, event, feature_names, penalizers, l1_ratio,
             penalty_weights=None, strata=None, **kw):
    """Fit a descending penalty path with warm starts (much faster)."""
    fits, warm = [], None
    for pen in sorted(penalizers, reverse=True):
        f = fit(X, time, event, feature_names, pen, l1_ratio,
                penalty_weights=penalty_weights, strata=strata,
                warm_start=warm, **kw)
        warm = f.beta.copy()
        fits.append(f)
    return fits
