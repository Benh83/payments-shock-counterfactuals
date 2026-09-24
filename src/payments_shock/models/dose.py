"""Continuous-treatment distributed-lag model on card-network availability.

The suspension was not a switch. Cross-border card acceptance degraded over
four trading days: the availability series fall 39.60 -> 23.76 -> 15.84 ->
7.92 -> 0 between 4 and 10 March. That staged rollout is variation in
treatment intensity inside the event window, and it is the only source of
identifying variation in this dataset that the war does not also produce. The
war did not step down in four equal increments on those particular days.

    d log y_t = a + sum_j theta_j * d dose_{t-j} + G' d X_t + e_t

``dose`` is normalised network availability: 1 while rails are intact, 0 once
both series hit zero. The payment effect at t is the accumulated response to
the dose path; the counterfactual is the observed series with that accumulation
added back.

Identification rests on the shape of the rollout being uncorrelated with the
shape of contemporaneous war news, which is testable in part by placebo dose
paths on pre-period dates.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from ..data import Design
from .base import CounterfactualModel, Paths, add_const, ols


class DoseResponse(CounterfactualModel):
    key = "dose"
    name = "Dose-response (rollout intensity)"
    family = "Continuous treatment"
    tagline = "Reads the effect off the four-day staged collapse in card acceptance."
    two_stage = False

    def paths(self, d: Design) -> Paths:
        y = d.y.copy()
        dy = np.diff(y, prepend=y[0])
        dose = d.dose
        ddose = np.diff(dose, prepend=dose[0])
        L = int(self.a.dose_lags)

        cols = [ddose]
        for j in range(1, L + 1):
            cols.append(np.concatenate([np.zeros(j), ddose[:-j]]))
        dX = np.diff(d.X_obs, axis=0, prepend=d.X_obs[:1])
        Z = add_const(np.column_stack(cols + [dX]))

        # Estimate on the war-and-after sample so the macro loadings are the
        # wartime ones, not the peacetime ones.
        start = int(np.argmax(d.index >= C.WAR_START))
        fit_mask = np.zeros(len(y), dtype=bool)
        fit_mask[start:] = True

        coef = ols(Z[fit_mask], dy[fit_mask])
        theta = coef[1 : L + 2]
        resid = dy[fit_mask] - Z[fit_mask] @ coef

        # Accumulated payment effect: response to the dose path only.
        contrib = np.zeros(len(y))
        for j, th in enumerate(theta):
            lagged = np.concatenate([np.zeros(j), ddose[: len(ddose) - j]]) if j else ddose
            contrib += th * lagged
        cum = np.cumsum(contrib)
        cum[d.index < self.a.intervention_date] = 0.0

        cf = y - cum  # war present, payments removed

        # Baseline: pre-war OLS on frozen controls, extended forward.
        pre = d.prewar_mask
        b = ols(add_const(d.X_obs[pre]), y[pre])
        baseline = add_const(d.X_base) @ b

        s = float(np.std(resid))
        horizon = np.clip(np.arange(len(y)) - start + 1, 1, None)
        cf_sd = s * np.sqrt(horizon)
        base_resid = y[pre] - (add_const(d.X_obs[pre]) @ b)
        base_sd = float(np.std(base_resid)) * np.sqrt(horizon / 3.0)

        fitted = np.zeros(len(y))
        fitted[fit_mask] = y[fit_mask] - resid
        self.theta = theta
        return Paths(
            observed=y,
            cf=cf,
            cf_sd=cf_sd,
            baseline=baseline,
            baseline_sd=base_sd,
            fitted=fitted,
            residuals=resid,
            train_mask=fit_mask,
        )
