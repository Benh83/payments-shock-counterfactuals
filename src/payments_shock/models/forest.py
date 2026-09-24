"""Random forest counterfactual.

A nonparametric benchmark. It buys flexibility in how controls map into
e-commerce and pays for it in two ways that matter here.

First, a forest cannot extrapolate: every prediction is an average of training
leaves, so the counterfactual is structurally incapable of leaving the range of
pre-intervention outcomes. If the true no-suspension path would have set a new
high or low, the forest will not find it.

Second, the plain specification has no dynamics, so its residuals are serially
correlated by construction and bootstrap standard errors understate real
forecast uncertainty. The ``rf_lags`` knob adds autoregressive features and
runs the forecast recursively, which repairs the first problem partially and
the second substantially; intervals come from a moving-block bootstrap that
respects the remaining dependence.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import CounterfactualModel


class Forest(CounterfactualModel):
    key = "rf"
    name = "Random forest"
    family = "Nonparametric ensemble"
    tagline = "Flexible mapping from controls. Cannot extrapolate past observed levels."

    def __init__(self, assumptions):
        super().__init__(assumptions)
        self.lags = int(assumptions.rf_lags)

    def _design(self, X, y, t, yhat):
        row = [X[t]]
        for l in range(1, self.lags + 1):
            row.append([yhat[t - l]])
        return np.concatenate(row)

    def _matrix(self, X, y, lo, hi):
        rows = [self._design(X, y, t, y) for t in range(lo, hi)]
        return np.array(rows)

    def _fit(self, y, X, mask):
        n = int(mask.sum())
        lo = self.lags
        Xt = self._matrix(X, y, lo, n)
        self.model = RandomForestRegressor(
            n_estimators=self.a.rf_trees,
            max_depth=self.a.rf_max_depth,
            min_samples_leaf=3,
            random_state=self.a.seed,
            n_jobs=-1,
        ).fit(Xt, y[lo:n])
        # Moving-block bootstrap ensemble for honest interval width.
        rng = np.random.default_rng(self.a.seed)
        self.boot = []
        block = 20
        idx = np.arange(len(Xt))
        for _ in range(24):
            starts = rng.integers(0, max(len(idx) - block, 1), size=len(idx) // block + 1)
            take = np.concatenate([idx[s : s + block] for s in starts])[: len(idx)]
            self.boot.append(
                RandomForestRegressor(
                    n_estimators=max(self.a.rf_trees // 4, 50),
                    max_depth=self.a.rf_max_depth,
                    min_samples_leaf=3,
                    random_state=int(rng.integers(1e6)),
                    n_jobs=-1,
                ).fit(Xt[take], y[lo:n][take])
            )
        self.n_train = n
        self._fitted = True

    def _roll(self, model, X, y, n):
        lo = self.lags
        out = np.array(y, dtype=float).copy()
        yhat = np.array(y, dtype=float).copy()
        # In-sample rows use observed lags, so they are one batched predict.
        if n > lo:
            out[lo:n] = model.predict(self._matrix(X, y, lo, n))
        if self.lags == 0:
            if len(X) > n:
                out[n:] = model.predict(X[n:])
            return out
        # Out-of-sample has to be recursive: each step feeds its own forecast.
        for t in range(max(lo, n), len(X)):
            row = self._design(X, yhat, t, yhat).reshape(1, -1)
            p = float(model.predict(row)[0])
            out[t] = p
            yhat[t] = p
        return out

    def _predict(self, X, y, mask):
        n = int(mask.sum())
        mean = self._roll(self.model, X, y, n)
        draws = np.array([self._roll(m, X, y, n) for m in self.boot])
        sd = draws.std(axis=0)
        resid_sd = float(np.std(y[self.lags : n] - mean[self.lags : n]))
        # Bootstrap spread alone is the standard error of a mean, not a
        # forecast interval. Add the irreducible residual scale.
        sd = np.sqrt(sd**2 + resid_sd**2)
        return mean, sd
