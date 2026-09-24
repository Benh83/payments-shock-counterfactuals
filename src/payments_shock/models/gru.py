"""Recurrent counterfactual forecaster (PyTorch GRU ensemble).

A learned nonlinear alternative to the ARIMA recursion. The network reads a
rolling window of controls and past outcomes and predicts the next day; past
the intervention it is rolled forward on its own output along the
counterfactual covariate path.

It earns its place for one reason: it is the only estimator here that can
represent a nonlinear, state-dependent response of e-commerce to macro
conditions, which is plausible when the ruble moves 60% in a fortnight. It
costs credibility for a matching reason: with roughly 420 usable
pre-intervention trading days the capacity has to be held down hard, and the
model has no interpretable identifying structure. Uncertainty is the spread of
a deep ensemble combined with Monte-Carlo dropout, which is a proper
predictive band only if the ensemble spans genuine model uncertainty.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import CounterfactualModel

SEQ = 20


class _Net(nn.Module):
    def __init__(self, n_in, hidden, dropout):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(self.drop(h[:, -1])).squeeze(-1)


class GRUForecaster(CounterfactualModel):
    key = "gru"
    name = "GRU ensemble"
    family = "Neural sequence model"
    tagline = "Nonlinear dynamics, no interpretable identification. Ensemble bands."

    def _windows(self, feats, y, lo, hi):
        xs, ys = [], []
        for t in range(max(lo, SEQ), hi):
            xs.append(feats[t - SEQ : t])
            ys.append(y[t])
        return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(
            np.array(ys), dtype=torch.float32
        )

    def _feats(self, X, yseq):
        return np.column_stack([X, yseq])

    def _fit(self, y, X, mask):
        n = int(mask.sum())
        self.y_mu = float(np.mean(y[:n]))
        self.y_sd = float(np.std(y[:n]) or 1.0)
        yz = (y - self.y_mu) / self.y_sd
        feats = self._feats(X, yz)
        xb, yb = self._windows(feats, yz, 0, n)

        self.nets = []
        for m in range(self.a.gru_members):
            torch.manual_seed(self.a.seed + m)
            net = _Net(feats.shape[1], self.a.gru_hidden, self.a.gru_dropout)
            opt = torch.optim.Adam(net.parameters(), lr=8e-3, weight_decay=1e-3)
            loss_fn = nn.SmoothL1Loss()
            net.train()
            for _ in range(220):
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            self.nets.append(net)
        self.n_train = n
        self._fitted = True

    def _roll(self, net, X, y, n, stochastic=False):
        net.train() if stochastic else net.eval()
        yz = (np.array(y, dtype=float) - self.y_mu) / self.y_sd
        path = yz.copy()
        out = yz.copy()
        with torch.no_grad():
            # In-sample the inputs are known, so the whole block is one batch.
            if n > SEQ:
                feats = self._feats(X, path)
                xb = torch.tensor(
                    np.stack([feats[t - SEQ : t] for t in range(SEQ, n)]), dtype=torch.float32
                )
                out[SEQ:n] = net(xb).numpy()
            # Past the intervention the model eats its own forecasts.
            for t in range(max(SEQ, n), len(X)):
                feats = np.column_stack([X[t - SEQ : t], path[t - SEQ : t]])
                xb = torch.tensor(feats[None, ...], dtype=torch.float32)
                p = float(net(xb).item())
                out[t] = p
                path[t] = p
        return out * self.y_sd + self.y_mu

    def _predict(self, X, y, mask):
        n = int(mask.sum())
        draws = np.array([self._roll(net, X, y, n) for net in self.nets])
        mc = np.array(
            [self._roll(self.nets[i % len(self.nets)], X, y, n, stochastic=True) for i in range(6)]
        )
        allp = np.vstack([draws, mc])
        mean = draws.mean(axis=0)
        sd = allp.std(axis=0)
        resid_sd = float(np.std(y[SEQ:n] - mean[SEQ:n]))
        sd = np.sqrt(sd**2 + resid_sd**2)
        mean[:SEQ] = y[:SEQ]
        return mean, sd
