"""C-ARIMA: ARIMA with exogenous controls, estimated by maximum likelihood.

The baseline specification, following Menchetti, Cipollini & Mealli (2023).
After the intervention the recursion is run forward on its own predictions,
which is exactly the counterfactual: the series evolves according to the
dynamics and covariates it had before the rails went down.

    y_t = c + X_t'b + sum_i a_i y_{t-i} + sum_j r_j u_{t-j} + u_t

Estimated in PyTorch by profiling out the innovation variance and minimising
the conditional negative log-likelihood. Forecast uncertainty comes from a
residual bootstrap that propagates through the recursion, so the intervals
widen with horizon the way the dynamics actually imply.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import CounterfactualModel


class CARIMA(CounterfactualModel):
    key = "carima"
    name = "C-ARIMA"
    family = "Parametric time series"
    tagline = "Linear dynamics plus controls. Transparent, and the weakest link is stationarity."

    def __init__(self, assumptions):
        super().__init__(assumptions)
        self.p, self.d, self.q = assumptions.arima_order

    # ------------------------------------------------------------------
    def _prep(self, y, X, mask):
        n_train = int(mask.sum())
        assert mask[:n_train].all(), "estimation sample must be a contiguous prefix"
        return n_train

    def _diff(self, y: np.ndarray) -> np.ndarray:
        """Work in the differenced series when d = 1.

        The recursion is estimated and rolled forward on the differences and
        integrated back afterwards, which is what makes ARIMA(1,1,1) a
        different model from ARMA(1,1) rather than a relabelling of it.
        """
        if self.d == 0:
            return np.asarray(y, dtype=float)
        z = np.zeros(len(y))
        z[1:] = np.diff(np.asarray(y, dtype=float))
        return z

    def _integrate(self, zhat: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
        if self.d == 0:
            return zhat
        out = np.asarray(y, dtype=float).copy()
        for t in range(1, len(zhat)):
            anchor = y[t - 1] if t <= n - 1 else out[t - 1]
            out[t] = anchor + zhat[t]
        return out

    def _fit(self, y, X, mask):
        torch.manual_seed(self.a.seed)
        n = self._prep(y, X, mask)
        z = self._diff(y)
        yt = torch.tensor(z[:n], dtype=torch.float64)
        Xt = torch.tensor(X[:n], dtype=torch.float64)
        k = Xt.shape[1]
        p, q = self.p, self.q
        warm = max(p, q)

        beta = torch.zeros(k, dtype=torch.float64, requires_grad=True)
        # A log-level series is close to a random walk, so start the leading AR
        # term near one. Starting at zero leaves Adam to climb a very flat
        # likelihood and it does not get there in a reasonable budget.
        init = torch.zeros(p, dtype=torch.float64)
        if p:
            target = 0.90 * max(p, 1) / 0.99  # so tanh(a)*(0.99/p) ~= 0.90
            init[0] = torch.atanh(torch.tensor(min(target, 0.999), dtype=torch.float64))
        const_init = float(yt.mean().item()) * (1.0 - 0.90) if p else float(yt.mean().item())
        const = torch.tensor([const_init], dtype=torch.float64, requires_grad=True)
        alpha = init.clone().requires_grad_(True)
        rho = torch.zeros(q, dtype=torch.float64, requires_grad=True)
        params = [beta, const, alpha, rho]

        def nll():
            # Constrain the AR block into the stationary region with a tanh
            # squash; an explosive root would make the counterfactual diverge.
            a = torch.tanh(alpha) * (0.99 / max(p, 1))
            r = torch.tanh(rho)
            # Everything that does not depend on past innovations is computed
            # in one shot; only the MA recursion needs a Python loop, and it
            # keeps innovations in a list so no full-tensor clone is made per
            # step (that turns an O(n) pass into an O(n^2) autograd graph).
            base = const[0] + Xt @ beta
            for i in range(p):
                base = base + a[i] * torch.roll(yt, i + 1)
            u_hist = [torch.zeros((), dtype=torch.float64) for _ in range(warm)]
            errs = []
            for t in range(warm, len(yt)):
                mu = base[t]
                for j in range(q):
                    mu = mu + r[j] * u_hist[t - 1 - j]
                e = yt[t] - mu
                u_hist.append(e)
                errs.append(e)
            e = torch.stack(errs)
            s2 = (e**2).mean()
            return 0.5 * len(e) * torch.log(s2 + 1e-12)

        opt = torch.optim.Adam(params, lr=0.05)
        for _ in range(260):
            opt.zero_grad()
            loss = nll()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

        with torch.no_grad():
            self.beta = beta.detach().numpy()
            self.const = float(const.item())
            self.alpha = (torch.tanh(alpha) * (0.99 / max(self.p, 1))).numpy()
            self.rho = torch.tanh(rho).numpy()
        self.loglik = float(-nll().item())
        self.n_train = n
        self._fitted = True

    # ------------------------------------------------------------------
    def _recurse(self, X, y, n_train, innov=None):
        p, q, warm = self.p, self.q, max(self.p, self.q)
        T = len(X)
        yhat = np.array(y, dtype=float).copy()
        u = np.zeros(T)
        out = np.array(y, dtype=float).copy()
        for t in range(warm, T):
            mu = self.const + X[t] @ self.beta
            for i in range(p):
                src = y[t - 1 - i] if t - 1 - i < n_train else yhat[t - 1 - i]
                mu += self.alpha[i] * src
            for j in range(q):
                mu += self.rho[j] * u[t - 1 - j]
            out[t] = mu
            if t < n_train:
                u[t] = y[t] - mu
                yhat[t] = y[t]
            else:
                shock = 0.0 if innov is None else innov[t]
                u[t] = shock
                yhat[t] = mu + shock
                out[t] = yhat[t]
        return out

    def _predict(self, X, y, mask):
        n = int(mask.sum())
        z = self._diff(y)
        warm = max(self.p, self.q)
        zhat = self._recurse(X, z, n)
        resid = z[warm:n] - zhat[warm:n]
        rng = np.random.default_rng(self.a.seed)
        draws = []
        for _ in range(240):
            innov = np.zeros(len(X))
            innov[n:] = rng.choice(resid, size=len(X) - n, replace=True)
            draws.append(self._integrate(self._recurse(X, z, n, innov=innov), y, n))
        sd = np.std(np.array(draws), axis=0)
        sd[:n] = float(np.std(resid))
        return self._integrate(zhat, y, n), sd
