"""Bayesian structural time series, as a Gaussian state-space model.

Following Brodersen et al. (2015), the observation decomposes into a local
level, a day-of-week seasonal, and a regression on contemporaneous controls
whose coefficients are allowed to drift:

    y_t     = mu_t + gamma_t + X_t'beta_t + eps_t
    mu_t+1  = mu_t + eta_mu
    gamma_t+1 = -sum_{s=0..3} gamma_{t-s} + eta_gamma
    beta_t+1 = beta_t + eta_beta

Implemented as a Kalman filter in PyTorch. Variance parameters are learned by
maximising the exact Gaussian likelihood; the counterfactual is the filter run
forward past the intervention on the counterfactual covariate path, so the
prediction interval is the model's own state uncertainty rather than a
bootstrap approximation.

Freeing the coefficients to drift is what lets this model absorb a structural
break in how macro conditions map into e-commerce, which is the single most
likely failure mode for a fixed-coefficient regression across a war.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import CounterfactualModel


class BSTS(CounterfactualModel):
    key = "bsts"
    name = "Bayesian structural time series"
    family = "State space"
    tagline = "Drifting level, weekly cycle, drifting control loadings. Best in-sample fit, widest bands."

    SEAS = 5  # trading-week cycle

    def __init__(self, assumptions):
        super().__init__(assumptions)
        self.use_seas = assumptions.bsts_seasonality
        self.tvp = assumptions.bsts_time_varying

    # ------------------------------------------------------------------
    def _dims(self, k):
        n_seas = self.SEAS - 1 if self.use_seas else 0
        return 1 + n_seas + k, n_seas

    def _transition(self, m, n_seas, k):
        T = torch.zeros(m, m, dtype=torch.float64)
        T[0, 0] = 1.0
        if n_seas:
            T[1, 1 : 1 + n_seas] = -1.0
            for i in range(n_seas - 1):
                T[2 + i, 1 + i] = 1.0
        off = 1 + n_seas
        for i in range(k):
            T[off + i, off + i] = 1.0
        return T

    def _filter(self, y, X, n_train, log_pars, forecast_to=None):
        k = X.shape[1]
        m, n_seas = self._dims(k)
        T = self._transition(m, n_seas, k)
        s_obs, s_lev, s_seas, s_beta = [torch.exp(p) for p in log_pars]
        q = torch.zeros(m, dtype=torch.float64)
        q[0] = s_lev
        if n_seas:
            q[1] = s_seas
        if k:
            q[1 + n_seas :] = s_beta if self.tvp else torch.tensor(1e-10, dtype=torch.float64)

        a = torch.zeros(m, dtype=torch.float64)
        a[0] = y[:20].mean()
        P = torch.eye(m, dtype=torch.float64) * 10.0
        P[0, 0] = 1e3
        if k:
            P[1 + n_seas :, 1 + n_seas :] = torch.eye(k, dtype=torch.float64) * 1e2

        Tend = len(y) if forecast_to is None else forecast_to
        ll = torch.tensor(0.0, dtype=torch.float64)
        mean = torch.zeros(Tend, dtype=torch.float64)
        var = torch.zeros(Tend, dtype=torch.float64)

        for t in range(Tend):
            Z = torch.zeros(m, dtype=torch.float64)
            Z[0] = 1.0
            if n_seas:
                Z[1] = 1.0
            if k:
                Z[1 + n_seas :] = X[t]
            pred = Z @ a
            F = Z @ P @ Z + s_obs
            mean[t] = pred
            var[t] = F
            if t < n_train:
                v = y[t] - pred
                ll = ll - 0.5 * (torch.log(2 * torch.pi * F) + v * v / F)
                K = (P @ Z) / F
                a = a + K * v
                P = P - torch.outer(K, Z @ P)
            a = T @ a
            P = T @ P @ T.T + torch.diag(q)
        return ll, mean, var

    # ------------------------------------------------------------------
    def _fit(self, y, X, mask):
        torch.manual_seed(self.a.seed)
        n = int(mask.sum())
        yt = torch.tensor(y, dtype=torch.float64)
        Xt = torch.tensor(X, dtype=torch.float64)
        scale = float(np.log(max(self.a.bsts_level_scale, 1e-4)))
        pars = [
            torch.tensor(-4.0, dtype=torch.float64, requires_grad=True),   # obs
            torch.tensor(scale, dtype=torch.float64, requires_grad=True),  # level
            torch.tensor(-6.0, dtype=torch.float64, requires_grad=True),   # seasonal
            torch.tensor(-8.0, dtype=torch.float64, requires_grad=True),   # beta drift
        ]
        opt = torch.optim.Adam(pars, lr=0.08)
        for _ in range(140):
            opt.zero_grad()
            ll, _, _ = self._filter(yt, Xt, n, pars, forecast_to=n)
            (-ll).backward()
            opt.step()
        self.pars = [p.detach() for p in pars]
        with torch.no_grad():
            ll, _, _ = self._filter(yt, Xt, n, self.pars, forecast_to=n)
        self.loglik = float(ll.item())
        self.n_train = n
        self._fitted = True

    def _predict(self, X, y, mask):
        n = int(mask.sum())
        with torch.no_grad():
            _, mean, var = self._filter(
                torch.tensor(y, dtype=torch.float64),
                torch.tensor(X, dtype=torch.float64),
                n,
                self.pars,
            )
        return mean.numpy(), np.sqrt(np.maximum(var.numpy(), 1e-12))
