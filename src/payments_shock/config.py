"""Campaign timeline, series definitions, and default analysis settings.

The exercise: on 5 March 2022 Visa, MasterCard and a set of other
international payment networks announced suspension of service inside Russia.
Card rails degraded over the following days and were fully down by 10 March.
Russia had already been at war since 24 February. Both shocks compress
Russian e-commerce volume. The task is to separate them.

Every date below is a trading day present in the source series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------

#: Full-scale invasion begins. First shock enters the series here.
WAR_START = pd.Timestamp("2022-02-24")

#: Visa / MasterCard suspension announced. Second shock enters here.
ANNOUNCEMENT = pd.Timestamp("2022-03-05")

#: First trading day on which degraded card rails are observable in the
#: network-availability series (NRBankinR falls 39.60 -> 23.76).
EFFECTIVE_START = pd.Timestamp("2022-03-07")

#: Rollout complete: both network-availability series hit zero.
ROLLOUT_COMPLETE = pd.Timestamp("2022-03-10")

#: The two weeks under the microscope: ten consecutive trading days spanning
#: the rollout and its immediate aftermath.
KEY_WINDOW = (pd.Timestamp("2022-03-07"), pd.Timestamp("2022-03-18"))

#: War-only identification window. The war is live, the payment networks are
#: not yet touched. Anything that happens to e-commerce here is attributable
#: to war and macro volatility alone.
WAR_ONLY_WINDOW = (pd.Timestamp("2022-02-24"), pd.Timestamp("2022-03-04"))

#: Pre-shock estimation sample ends here (last clean trading day before war).
PRE_WAR_END = pd.Timestamp("2022-02-23")

SAMPLE_START = pd.Timestamp("2021-01-04")
SAMPLE_END = pd.Timestamp("2022-12-30")


# --------------------------------------------------------------------------
# Series
# --------------------------------------------------------------------------

OUTCOME = "ecom"

#: Raw column -> human label. Where the source column name is ambiguous the
#: label says so rather than inventing precision.
SERIES_LABELS: dict[str, str] = {
    "ecom": "Russian e-commerce volume (mn)",
    "UKR_ecom": "Ukrainian e-commerce volume (mn)",
    "consum_index": "Russian consumer & retail equity index",
    "xr": "Ruble strength index (inverse FX proxy)",
    "CPI": "Russian CPI, % change",
    "RTS_OGI": "RTS oil & gas index",
    "USD_XR": "USD/RUB exchange rate",
    "vix_close": "CBOE VIX close (global risk)",
    "RCardinR": "Russian-issued cards used in Russia",
    "Dem_mthly": "Domestic demand indicator (monthly, interpolated)",
    "NRBankinR": "Non-Russian-issued cards usable in Russia",
    "Rbankoutside": "Russian-issued cards usable outside Russia",
}

#: These two series ARE the treatment. They trace the collapse of cross-border
#: card acceptance from 39.60 / 274.0 on 4 March to 0 / 0 on 10 March. Putting
#: them in a control set would regress the outcome on its own cause, so every
#: counterfactual model excludes them. Only the dose-response model touches
#: them, and it treats them as the intervention, not as a covariate.
TREATMENT_INTENSITY = ["NRBankinR", "Rbankoutside"]

#: Default control set for counterfactual models.
DEFAULT_CONTROLS = [
    "UKR_ecom",
    "consum_index",
    "CPI",
    "RTS_OGI",
    "USD_XR",
    "vix_close",
    "RCardinR",
    "Dem_mthly",
]

#: Controls that are contaminated by the payment shock itself, not only by the
#: war. Ruble collapse and equity repricing after 5 March partly reflect the
#: sanctions package the card suspension belongs to. The contamination slider
#: (lambda) walks these series back toward their pre-announcement path.
CONTAMINATED_CONTROLS = ["USD_XR", "RTS_OGI", "consum_index", "CPI", "xr"]

#: War-exposed, payments-unexposed. Ukraine took the war shock without taking
#: the Visa/MasterCard-in-Russia shock, which is what makes the war channel
#: separately identified rather than assumed.
WAR_CONTROL = "UKR_ecom"

#: Columns dropped on load (empty trailing columns in the source export).
DROP_COLUMNS = ["Unnamed: 13", "Unnamed: 14"]


# --------------------------------------------------------------------------
# Analysis settings
# --------------------------------------------------------------------------


@dataclass
class Assumptions:
    """Every knob the dashboard exposes.

    Defaults reproduce the headline specification. Each field is described in
    ``assumptions.py`` together with the identifying assumption it stresses.
    """

    # --- timing -----------------------------------------------------------
    intervention_date: pd.Timestamp = ANNOUNCEMENT
    #: Trading days before the intervention dropped from the estimation sample.
    #: Non-zero values buy insurance against anticipation: Russian consumers
    #: and banks had days of warning, and pre-positioning would otherwise be
    #: absorbed into the counterfactual as if it were normal behaviour.
    #: Defaults to 3: volume on 2-4 March runs 35-40bn against a February norm
    #: nearer 25bn, which is pull-forward ahead of widely-trailed sanctions,
    #: not ordinary demand. Left in the estimation sample it teaches every
    #: model that early March was a boom and inflates the counterfactual.
    anticipation_days: int = 3
    key_window: tuple[pd.Timestamp, pd.Timestamp] = KEY_WINDOW

    # --- identification ---------------------------------------------------
    #: Share of post-announcement covariate movement attributed to the payment
    #: shock rather than the war. 0 keeps observed covariates (they are assumed
    #: to be pure war signal); 1 restores them to their pre-announcement path
    #: (they are assumed to be fully sanctions-driven, so using them raw would
    #: launder payment effects into the counterfactual).
    contamination: float = 0.0
    #: Elasticity of Russian war exposure to Ukrainian war exposure. 1.0 means
    #: the two economies absorb the war shock proportionally in logs. Estimated
    #: from the war-only window when ``calibrate_exposure`` is set.
    war_exposure: float = 1.0
    calibrate_exposure: bool = True

    # --- specification ----------------------------------------------------
    controls: list[str] = field(default_factory=lambda: list(DEFAULT_CONTROLS))
    log_outcome: bool = True
    #: Extrapolation rule for the no-war baseline covariate path.
    baseline_covariate_rule: str = "freeze"  # freeze | drift

    # --- model-specific ---------------------------------------------------
    arima_order: tuple[int, int, int] = (1, 0, 3)
    bsts_seasonality: bool = True
    bsts_time_varying: bool = True
    bsts_level_scale: float = 0.05
    rf_trees: int = 400
    rf_max_depth: int = 8
    rf_lags: int = 0
    gru_hidden: int = 32
    gru_members: int = 8
    gru_dropout: float = 0.15
    dose_lags: int = 5
    did_pre_days: int = 60

    seed: int = 20220305

    def key_window_mask(self, index: pd.DatetimeIndex) -> pd.Series:
        lo, hi = self.key_window
        return (index >= lo) & (index <= hi)


DEFAULT = Assumptions()
