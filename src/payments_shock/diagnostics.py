"""Residual and comparison diagnostics.

These decide which assumption badges the dashboard lights up. A model that
fails Ljung-Box has dynamics left in its residuals, which means its
counterfactual is missing structure and its intervals are too narrow. A model
that fails Breusch-Pagan has a variance that moves, which breaks interval
coverage but not the point estimate.

Only in-sample comparison is possible. Out-of-sample here *is* the
counterfactual, and there is no second reality to score it against.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .data import Design
from .models.base import Paths, add_const, ols


def ljung_box(resid: np.ndarray, lags: int) -> float:
    r = np.asarray(resid, dtype=float)
    r = r - r.mean()
    n = len(r)
    denom = float((r**2).sum())
    if denom <= 0 or n <= lags + 1:
        return float("nan")
    q = 0.0
    for k in range(1, lags + 1):
        ac = float((r[k:] * r[:-k]).sum()) / denom
        q += ac**2 / (n - k)
    q *= n * (n + 2)
    return float(1 - stats.chi2.cdf(q, lags))


def breusch_pagan(resid: np.ndarray, X: np.ndarray) -> float:
    r = np.asarray(resid, dtype=float)
    g = r**2 / (r**2).mean()
    Z = add_const(X)
    b = ols(Z, g)
    fit = Z @ b
    ss = float(((fit - g.mean()) ** 2).sum()) / 2.0
    return float(1 - stats.chi2.cdf(ss, max(X.shape[1], 1)))


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> float:
    n = min(len(e1), len(e2))
    d = e1[:n] ** 2 - e2[:n] ** 2
    dbar = d.mean()
    gamma0 = d.var()
    if gamma0 <= 0:
        return float("nan")
    var = gamma0
    for k in range(1, h):
        var += 2 * float((d[k:] - dbar) @ (d[:-k] - dbar)) / n
    stat = dbar / np.sqrt(max(var, 1e-18) / n)
    return float(2 * (1 - stats.norm.cdf(abs(stat))))


def evaluate(d: Design, p: Paths) -> dict:
    r = np.asarray(p.residuals, dtype=float)
    r = r[np.isfinite(r)]
    mask = p.train_mask
    X = d.X_obs[mask][-len(r):] if len(r) else d.X_obs[mask]
    return {
        "rmse": float(np.sqrt((r**2).mean())) if len(r) else float("nan"),
        "ljung_box_1": ljung_box(r, 1),
        "ljung_box_5": ljung_box(r, 5),
        "ljung_box_10": ljung_box(r, 10),
        "breusch_pagan": breusch_pagan(r, X) if len(r) == len(X) else float("nan"),
        "n_residuals": int(len(r)),
    }


def placebo(model_cls, assumptions, design_builder, shift_days: int = 120) -> float:
    """Run the whole pipeline on a fake intervention date well before the real
    one. A well-behaved estimator should report roughly zero damage there.
    Returns the placebo payment effect in percent.
    """
    import copy

    from .decomposition import attribute

    a = copy.deepcopy(assumptions)
    a.intervention_date = assumptions.intervention_date - np.timedelta64(shift_days, "D")
    lo, hi = assumptions.key_window
    a.key_window = (lo - np.timedelta64(shift_days, "D"), hi - np.timedelta64(shift_days, "D"))
    d = design_builder(a)
    p = model_cls(a).paths(d)
    return attribute(d, p).payments_pct
