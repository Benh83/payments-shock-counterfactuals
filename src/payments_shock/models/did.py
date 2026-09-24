"""Difference-in-differences against Ukrainian e-commerce.

The other five estimators infer the war channel from covariates. This one
observes it. Ukraine absorbed the same war and none of the Visa/MasterCard-
in-Russia suspension, so Ukrainian e-commerce is a live readout of what war
volatility alone does to online retail demand in the theatre.

    log RU_t  =  trend_t  +  phi * war_shock_t  +  payments_t
    war_shock_t = log UA_t - trend^UA_t

``phi`` is the exposure elasticity: how much of Ukraine's war-driven
contraction transmits to Russia. Setting it to 1 assumes the two economies
take the war proportionally in logs. It is instead calibrated by default on
the seven trading days between the invasion and the suspension announcement,
the one stretch of the sample where war is live and rails are intact.

The whole estimator stands or falls on Ukraine being a valid control, which is
a strong claim and the reason it is presented alongside the others rather than
as the answer.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from ..data import Design
from .base import CounterfactualModel, Paths


class UkraineDiD(CounterfactualModel):
    key = "did"
    name = "Ukraine difference-in-differences"
    family = "Design-based"
    tagline = "Uses a war-exposed, payments-unexposed control instead of assuming the war away."
    two_stage = False

    def paths(self, d: Design) -> Paths:
        idx = d.index
        raw_ru = np.log(d.y_level)
        ua = np.log(d.X_obs[:, d.columns.index(C.WAR_CONTROL)]) if False else None

        # Ukrainian series is in the standardised design matrix; recover it
        # from the source frame instead so the trend fit is in log levels.
        from ..data import load_raw

        rawf = load_raw()
        ru = np.log(rawf[C.OUTCOME].to_numpy())
        uk = np.log(rawf[C.WAR_CONTROL].to_numpy())

        pre_end = int(np.sum(idx <= C.PRE_WAR_END))
        lo = max(pre_end - self.a.did_pre_days, 10)
        t = np.arange(len(idx), dtype=float)

        ru_fit = np.polyfit(t[lo:pre_end], ru[lo:pre_end], 1)
        uk_fit = np.polyfit(t[lo:pre_end], uk[lo:pre_end], 1)
        ru_trend = np.polyval(ru_fit, t)
        uk_trend = np.polyval(uk_fit, t)

        phi = float(self.a.war_exposure)
        war_shock = uk - uk_trend
        war_shock[:pre_end] = 0.0

        baseline = ru_trend.copy()
        cf = baseline + phi * war_shock

        resid = ru[lo:pre_end] - ru_trend[lo:pre_end]
        s = float(np.std(resid))
        horizon = np.clip(t - pre_end + 1, 1, None)
        base_sd = s * np.sqrt(horizon / 5.0)
        base_sd[:pre_end] = s
        # phi is estimated from seven observations; treat it as uncertain.
        phi_sd = 0.35 * max(abs(phi), 0.2)
        cf_sd = np.sqrt(base_sd**2 + (phi_sd * war_shock) ** 2)

        fitted = ru_trend.copy()
        return Paths(
            observed=ru,
            cf=cf,
            cf_sd=cf_sd,
            baseline=baseline,
            baseline_sd=base_sd,
            fitted=fitted,
            residuals=ru[lo:pre_end] - ru_trend[lo:pre_end],
            train_mask=(t < pre_end) & (t >= lo),
        )
