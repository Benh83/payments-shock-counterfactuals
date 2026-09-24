"""Load, clean and stage the daily panel.

One function matters downstream: :func:`build_design`. It returns the outcome,
three covariate matrices, and the masks that define the estimation sample.

The three covariate matrices correspond to three worlds:

``X_obs``   what actually happened. War and payment shock both in the data.
``X_cf``    war, no payment shock. Contaminated controls partially walked back
            toward their pre-announcement path by ``lambda``.
``X_base``  neither shock. Controls frozen (or drifted) at their pre-war path.

Feeding the same fitted model the three matrices yields the three paths whose
differences are the attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "russia_ecom_daily.csv"


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Read the source export and return a clean, date-indexed frame."""
    path = Path(path) if path is not None else DATA_PATH
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in C.DROP_COLUMNS if c in df.columns], errors="ignore")
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    df = df.sort_values("date").set_index("date")
    df = df[~df.index.duplicated(keep="first")]
    if df.isna().any().any():
        df = df.interpolate(limit_direction="both")
    return df.astype(float)


def _standardise(frame: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    mu, sd = ref.mean(), ref.std().replace(0.0, 1.0)
    return (frame - mu) / sd


@dataclass
class Design:
    """Everything a model needs, already aligned."""

    index: pd.DatetimeIndex
    y: np.ndarray              # transformed outcome, full sample
    y_level: np.ndarray        # outcome in millions
    X_obs: np.ndarray
    X_cf: np.ndarray
    X_base: np.ndarray
    columns: list[str]
    train_mask: np.ndarray     # estimation sample (pre-intervention, anticipation-trimmed)
    prewar_mask: np.ndarray    # strictly pre-war estimation sample
    post_mask: np.ndarray      # from intervention onward
    window_mask: np.ndarray    # the two key weeks
    dose: np.ndarray           # treatment intensity, 0 = rails fully down, 1 = normal
    assumptions: C.Assumptions

    @property
    def n(self) -> int:
        return len(self.index)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Back to millions of dollars of e-commerce volume."""
        return np.exp(y) if self.assumptions.log_outcome else y


def _covariate_paths(
    raw: pd.DataFrame, a: C.Assumptions, controls: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = raw[controls].copy()

    # ---- counterfactual world: war yes, payment shock no -----------------
    # Walk contaminated controls back toward the level they held just before
    # the announcement. lambda = 0 leaves them alone, lambda = 1 pins them.
    cf = obs.copy()
    lam = float(np.clip(a.contamination, 0.0, 1.0))
    if lam > 0:
        anchor_dates = raw.index[raw.index < a.intervention_date][-5:]
        post = raw.index >= a.intervention_date
        for col in controls:
            if col not in C.CONTAMINATED_CONTROLS:
                continue
            anchor = float(raw.loc[anchor_dates, col].mean())
            cf.loc[post, col] = (1 - lam) * obs.loc[post, col] + lam * anchor

    # ---- baseline world: neither shock -----------------------------------
    base = obs.copy()
    prewar = raw.index <= C.PRE_WAR_END
    post_war = raw.index > C.PRE_WAR_END
    ref = raw.loc[prewar, controls]
    for col in controls:
        anchor = float(ref[col].iloc[-10:].mean())
        if a.baseline_covariate_rule == "drift":
            slope = float(np.polyfit(np.arange(60), ref[col].iloc[-60:].to_numpy(), 1)[0])
            steps = np.arange(1, int(post_war.sum()) + 1)
            base.loc[post_war, col] = anchor + slope * steps
        else:
            base.loc[post_war, col] = anchor
    return obs, cf, base


def build_design(
    a: C.Assumptions | None = None, raw: pd.DataFrame | None = None
) -> Design:
    a = a or C.DEFAULT
    raw = load_raw() if raw is None else raw
    controls = [c for c in a.controls if c in raw.columns and c not in C.TREATMENT_INTENSITY]

    y_level = raw[C.OUTCOME].to_numpy(dtype=float)
    y = np.log(y_level) if a.log_outcome else y_level.copy()

    obs, cf, base = _covariate_paths(raw, a, controls)

    # Log-transform strictly positive, trending covariates so the regression
    # lives in the same units as the outcome. Rate series stay in levels.
    rate_like = {"CPI", "vix_close", "Dem_mthly"}
    for frame in (obs, cf, base):
        for col in controls:
            if col in rate_like:
                continue
            frame[col] = np.log(frame[col].clip(lower=1e-6))

    # Standardisation uses full-sample moments, not pre-war moments. Pre-war
    # variation in these series is tiny next to what 2022 did to them: the
    # ruble's log level sits roughly seventeen pre-war standard deviations out
    # in March. Scaling by that yardstick makes every control a vast outlier,
    # a linear projection on it explodes, and the contamination slider loses
    # all bite because every setting is clipped at the same bound. Full-sample
    # scaling is a monotone rescaling of regressors only; it uses no outcome
    # information and changes no effect estimate, but it keeps the design
    # matrix in a range where the models behave.
    ref = obs
    clip = 6.0
    X_obs = _standardise(obs, ref).clip(-clip, clip).to_numpy(dtype=float)
    X_cf = _standardise(cf, ref).clip(-clip, clip).to_numpy(dtype=float)
    X_base = _standardise(base, ref).clip(-clip, clip).to_numpy(dtype=float)

    idx = raw.index
    cutoff = a.intervention_date - pd.Timedelta(days=0)
    train = idx < cutoff
    if a.anticipation_days > 0:
        keep = idx[train][: -a.anticipation_days]
        train = idx.isin(keep)
    prewar = idx <= C.PRE_WAR_END
    post = idx >= a.intervention_date
    window = a.key_window_mask(idx)

    # Dose: 1 while cross-border rails are intact, 0 once both network series
    # hit zero. Averaged across the two availability series, normalised on the
    # pre-announcement level.
    inten = raw[C.TREATMENT_INTENSITY]
    pre_level = inten.loc[idx < a.intervention_date].iloc[-5:].mean()
    dose = (inten / pre_level.replace(0.0, np.nan)).mean(axis=1).fillna(0.0)
    dose = dose.clip(0.0, 1.0).to_numpy(dtype=float)

    return Design(
        index=idx,
        y=y,
        y_level=y_level,
        X_obs=X_obs,
        X_cf=X_cf,
        X_base=X_base,
        columns=controls,
        train_mask=train,
        prewar_mask=prewar,
        post_mask=post,
        window_mask=window,
        dose=dose,
        assumptions=a,
    )


def calibrate_war_exposure(raw: pd.DataFrame | None = None) -> dict:
    """Read the war exposure elasticity off the war-only window.

    For every 1% the war took off Ukrainian e-commerce, how much did it take
    off Russian? Estimated on 24 February to 4 March 2022, the only stretch of
    the sample where the war is live and the card rails are still intact.

    The answer this dataset gives is blunt: Ukrainian volume fell roughly 80%
    against its pre-war level while Russian volume was flat to higher, so the
    raw ratio is slightly negative and the usable elasticity sits at its floor
    of zero. That is a result, not a failure — it says the war on its own had
    not yet dented Russian e-commerce when the networks pulled out. The raw
    value is returned alongside so the dashboard can show where the floor
    binds instead of quietly hiding it.
    """
    raw = load_raw() if raw is None else raw
    lo, hi = C.WAR_ONLY_WINDOW
    pre = raw.loc[raw.index <= C.PRE_WAR_END].iloc[-20:]
    win = raw.loc[(raw.index >= lo) & (raw.index <= hi)]
    ru = float(np.log(win[C.OUTCOME].mean() / pre[C.OUTCOME].mean()))
    ua = float(np.log(win[C.WAR_CONTROL].mean() / pre[C.WAR_CONTROL].mean()))
    raw_phi = ru / ua if abs(ua) > 1e-8 else 1.0
    return {
        "phi": float(np.clip(raw_phi, 0.0, 3.0)),
        "phi_raw": float(raw_phi),
        "russia_log_change": ru,
        "ukraine_log_change": ua,
        "at_floor": bool(raw_phi < 0.0),
        "n_days": int(len(win)),
    }
