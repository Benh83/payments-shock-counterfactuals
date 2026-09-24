"""Split the observed collapse into war and payments channels.

The question is not "what level would e-commerce have held" but "how much of
the decline over the two key weeks came from which shock". So the accounting
is done on *declines* measured from a common reference level, not on levels.

Reference R is the observed average over the five trading days immediately
before the intervention. Endpoint E is the average over the last three trading
days of the assessment window. Three worlds give three declines:

    D_obs   = (E_obs  - R) / R      war and suspension both live
    D_cf    = (E_cf   - R) / R      war live, card rails intact
    D_base  = (E_base - R) / R      neither shock

The counterfactual and observed worlds are identical before the intervention,
so they share R by construction. The no-shock baseline is rescaled to the same
reference before comparison: its *level* is a long extrapolation and the least
credible thing the models produce, but its *shape* is what the war channel
needs, and rescaling uses only the shape.

    payments = D_obs - D_cf     what the suspension added to the decline
    war      = D_cf  - D_base   what the war did on its own
    secular  = D_base           what would have happened anyway

and the three sum to the observed decline by construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data import Design
from .models.base import Paths

REF_DAYS = 5
END_DAYS = 3


@dataclass
class Attribution:
    window_start: str
    window_end: str
    n_days: int

    reference_mn: float
    observed_end_mn: float
    counterfactual_end_mn: float
    baseline_end_mn: float

    # Decline decomposition, percentage points of the reference level.
    decline_pct: float
    payments_pp: float
    war_pp: float
    secular_pp: float

    # Shares of the observed decline.
    payments_share: float
    war_share: float
    secular_share: float
    shares_well_posed: bool

    # Cumulative volume over the window, millions.
    lost_total_mn: float
    lost_war_mn: float
    lost_payments_mn: float

    payments_ci_low: float
    payments_ci_high: float
    payments_significant: bool

    def to_dict(self):
        return asdict(self)


def _ref_mask(d: Design) -> np.ndarray:
    pre = d.index < d.assumptions.intervention_date
    idx = np.where(pre)[0][-REF_DAYS:]
    m = np.zeros(len(d.index), dtype=bool)
    m[idx] = True
    return m


def _end_mask(d: Design) -> np.ndarray:
    idx = np.where(d.window_mask)[0][-END_DAYS:]
    m = np.zeros(len(d.index), dtype=bool)
    m[idx] = True
    return m


def rescaled_paths(d: Design, p: Paths) -> dict[str, np.ndarray]:
    """All three worlds in millions, each normalised onto the observed
    reference level.

    Every counterfactual here is an extrapolation, and the level of an
    extrapolation is its least trustworthy feature: a model that is 8% high on
    average before the intervention will be roughly 8% high after it, and that
    bias would land entirely in the attribution. Matching each path to the
    observed average over the five trading days before the intervention throws
    the level away and keeps the shape, which is the part carrying the causal
    signal. It also makes the estimators comparable to each other rather than
    to their own calibration error.
    """
    O = d.inverse(p.observed)
    ref = _ref_mask(d)
    o_ref = float(O[ref].mean())

    def norm(x):
        arr = d.inverse(x)
        s = o_ref / max(float(arr[ref].mean()), 1e-9)
        return arr * s, s

    C, s_cf = norm(p.cf)
    B, s_b = norm(p.baseline)
    return {
        "observed": O,
        "cf": C,
        "baseline": B,
        "cf_lo": d.inverse(p.cf - 1.96 * p.cf_sd) * s_cf,
        "cf_hi": d.inverse(p.cf + 1.96 * p.cf_sd) * s_cf,
        "scale_cf": s_cf,
        "scale_baseline": s_b,
    }


def attribute(d: Design, p: Paths, z: float = 1.96) -> Attribution:
    paths = rescaled_paths(d, p)
    O, C, B = paths["observed"], paths["cf"], paths["baseline"]
    ref, end = _ref_mask(d), _end_mask(d)

    R = float(O[ref].mean())
    e_obs = float(O[end].mean())
    e_cf = float(C[end].mean())
    e_base = float(B[end].mean())

    d_obs = (e_obs - R) / R * 100.0
    d_cf = (e_cf - R) / R * 100.0
    d_base = (e_base - R) / R * 100.0

    payments = d_obs - d_cf
    war = d_cf - d_base
    secular = d_base

    # Shares are taken over the shock-attributable part of the decline, not
    # over the raw decline. The secular term is whatever the no-shock world was
    # going to do anyway and can carry either sign; folding it into the
    # denominator would let a mildly rising baseline push the payment share
    # above one for arithmetic reasons rather than economic ones.
    shock = payments + war
    if abs(shock) > 1e-9:
        shares = (payments / shock, war / shock)
        well_posed = (payments <= 0) == (war <= 0)
    else:
        shares = (float("nan"), float("nan"))
        well_posed = False
    secular_share = secular / d_obs if abs(d_obs) > 1e-9 else float("nan")

    w = d.window_mask
    lost_pay = float((C[w] - O[w]).sum())
    lost_war = float((B[w] - C[w]).sum())
    lost_total = float((B[w] - O[w]).sum())

    sd_end = (d.inverse(p.cf) * paths["scale_cf"] * p.cf_sd)[end]
    se = float(np.sqrt((sd_end**2).sum())) / max(end.sum(), 1)
    lo = (e_obs - e_cf - z * se) / R * 100.0
    hi = (e_obs - e_cf + z * se) / R * 100.0

    return Attribution(
        window_start=str(pd.Timestamp(d.index[w][0]).date()),
        window_end=str(pd.Timestamp(d.index[w][-1]).date()),
        n_days=int(w.sum()),
        reference_mn=R,
        observed_end_mn=e_obs,
        counterfactual_end_mn=e_cf,
        baseline_end_mn=e_base,
        decline_pct=d_obs,
        payments_pp=payments,
        war_pp=war,
        secular_pp=secular,
        payments_share=float(shares[0]),
        war_share=float(shares[1]),
        secular_share=float(secular_share),
        shares_well_posed=bool(well_posed),
        lost_total_mn=lost_total,
        lost_war_mn=lost_war,
        lost_payments_mn=lost_pay,
        payments_ci_low=lo,
        payments_ci_high=hi,
        payments_significant=bool(lo * hi > 0),
    )


def daily_series(d: Design, p: Paths, start: str = "2022-01-03") -> dict:
    """Paths in millions over the plotting range."""
    paths = rescaled_paths(d, p)
    keep = d.index >= pd.Timestamp(start)
    return {
        "dates": [str(pd.Timestamp(x).date()) for x in d.index[keep]],
        "observed": np.round(paths["observed"][keep], 1).tolist(),
        "cf": np.round(paths["cf"][keep], 1).tolist(),
        "cf_lo": np.round(paths["cf_lo"][keep], 1).tolist(),
        "cf_hi": np.round(paths["cf_hi"][keep], 1).tolist(),
        "baseline": np.round(paths["baseline"][keep], 1).tolist(),
        "dose": np.round(d.dose[keep], 3).tolist(),
    }
