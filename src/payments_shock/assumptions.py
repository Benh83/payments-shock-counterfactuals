"""What each estimator needs to be true before its number means anything.

This module is the spine of the dashboard. A counterfactual is not a forecast;
it is a claim about a world that did not happen, and the claim is only as good
as the assumption that licenses it. For every model the registry below states
the identifying assumptions, why the model collapses without them, how well
this particular dataset supports them, and which control on the dashboard
stresses them.

Status vocabulary
    holds      supported by the data or by the design of the intervention
    strained   partially supported; the number survives but moves
    fails      not supported here; treat the model's output as an upper or
               lower bound rather than an estimate
    testable   can be checked with a diagnostic reported in the dashboard
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Knob:
    id: str
    label: str
    kind: str          # slider | int | choice | multi | toggle
    help: str
    low: float | None = None
    high: float | None = None
    step: float | None = None
    default: float | str | bool | None = None
    options: tuple = ()

    def to_dict(self):
        return asdict(self)


KNOBS: dict[str, Knob] = {
    "contamination": Knob(
        "contamination",
        "Covariate contamination (lambda)",
        "slider",
        "How much of the post-5-March move in the ruble, equities and CPI you "
        "attribute to the sanctions package rather than to the war. At 0 the "
        "controls are treated as pure war signal and the counterfactual is "
        "allowed to use them raw. At 1 they are restored to their pre-"
        "announcement level, on the view that letting a sanctions-driven "
        "ruble into the control set launders payment effects into the "
        "counterfactual and shrinks the estimated payment damage.",
        0.0, 1.0, 0.05, 0.0,
    ),
    "war_exposure": Knob(
        "war_exposure",
        "Russia/Ukraine war exposure (phi)",
        "slider",
        "For every 1% the war took off Ukrainian e-commerce, how much it took "
        "off Russian. 1.0 means proportional in logs. Calibrated by default on "
        "24 February to 4 March, the only window with war and intact rails.",
        0.0, 2.0, 0.05, 1.0,
    ),
    "anticipation_days": Knob(
        "anticipation_days",
        "Anticipation buffer (trading days)",
        "int",
        "Trading days before the announcement dropped from the estimation "
        "sample. Russian consumers and banks had warning; pre-positioning left "
        "in the training data is learned as normal behaviour and biases the "
        "counterfactual toward the already-depressed path.",
        0, 10, 1, 3,
    ),
    "intervention_date": Knob(
        "intervention_date",
        "Intervention date",
        "choice",
        "Announcement (5 March) versus first observable degradation in card "
        "acceptance (7 March). Announcement effects and operational effects "
        "are different treatments.",
        default="2022-03-05",
        options=("2022-03-05", "2022-03-07"),
    ),
    "key_window": Knob(
        "key_window",
        "Two-week assessment window",
        "choice",
        "Which ten trading days the headline number covers.",
        default="2022-03-07:2022-03-18",
        options=(
            "2022-03-07:2022-03-18",
            "2022-03-05:2022-03-16",
            "2022-03-10:2022-03-23",
        ),
    ),
    "baseline_covariate_rule": Knob(
        "baseline_covariate_rule",
        "No-shock covariate path",
        "choice",
        "How macro controls are assumed to have evolved in a world with no war "
        "at all: frozen at their pre-war level, or continuing their pre-war "
        "trend.",
        default="freeze",
        options=("freeze", "drift"),
    ),
    "controls": Knob(
        "controls", "Control set", "multi",
        "Which series the counterfactual is allowed to condition on. Card-"
        "availability series are never selectable: they are the treatment.",
    ),
    "arima_order": Knob(
        "arima_order", "ARIMA order (p,d,q)", "choice",
        "Dynamic specification. (1,0,3) was selected on AIC/BIC over the "
        "pre-intervention sample.",
        default="1,0,3", options=("1,0,3", "1,0,0", "2,0,1", "1,1,1"),
    ),
    "bsts_time_varying": Knob(
        "bsts_time_varying", "Allow control loadings to drift", "toggle",
        "Lets the mapping from macro conditions to e-commerce change through "
        "the war. Switching it off imposes that the pre-war relationship held "
        "unchanged across the invasion.",
        default=True,
    ),
    "bsts_seasonality": Knob(
        "bsts_seasonality", "Day-of-week component", "toggle",
        "Trading-week cycle in the state vector.", default=True,
    ),
    "bsts_level_scale": Knob(
        "bsts_level_scale", "Local level prior scale", "slider",
        "How freely the underlying level is allowed to wander. Larger values "
        "let the counterfactual track recent drift and widen the bands.",
        0.005, 0.30, 0.005, 0.05,
    ),
    "rf_lags": Knob(
        "rf_lags", "Autoregressive lags in the forest", "int",
        "Adding lagged outcomes gives the forest dynamics it otherwise lacks "
        "and removes most of the serial correlation in its residuals.",
        0, 5, 1, 0,
    ),
    "rf_max_depth": Knob(
        "rf_max_depth", "Forest depth", "int",
        "Capacity. Deeper trees fit the pre-period harder and extrapolate no "
        "better.", 3, 16, 1, 8,
    ),
    "gru_members": Knob(
        "gru_members", "Ensemble members", "int",
        "Independent networks averaged. Ensemble spread is most of the "
        "reported uncertainty.", 2, 12, 1, 8,
    ),
    "gru_hidden": Knob(
        "gru_hidden", "Hidden units", "int",
        "Capacity against roughly 420 pre-intervention observations.",
        8, 64, 8, 32,
    ),
    "dose_lags": Knob(
        "dose_lags", "Dose response lags", "int",
        "Trading days over which a change in card availability is allowed to "
        "pass through to volume.", 0, 10, 1, 5,
    ),
    "did_pre_days": Knob(
        "did_pre_days", "Pre-period length (trading days)", "int",
        "Window used to fit the Russian and Ukrainian trends. Longer windows "
        "are more stable and less local to the pre-war level.",
        20, 180, 5, 60,
    ),
}


@dataclass(frozen=True)
class AssumptionSpec:
    id: str
    label: str
    statement: str
    why: str
    status: str
    evidence: str
    knobs: tuple = ()

    def to_dict(self):
        return asdict(self)


A = AssumptionSpec

_SUTVA = A(
    "sutva",
    "Single, persistent, well-defined treatment",
    "The suspension is one intervention, it does not reverse inside the "
    "window, and every unit of the outcome is exposed to the same treatment.",
    "Without it there is no single counterfactual to recover; the estimand "
    "itself is undefined.",
    "holds",
    "Visa, MasterCard and peers suspended over 5-10 March 2022 and did not "
    "restore service inside the sample. The card-availability series go to "
    "zero and stay there through 30 December 2022.",
    ("intervention_date",),
)

_NO_ANTICIPATION = A(
    "no_anticipation",
    "No anticipation",
    "Behaviour before the announcement is unaffected by the coming "
    "suspension.",
    "Anticipatory stockpiling or card-switching before 5 March is learned by "
    "the model as normal pre-period behaviour, which drags the counterfactual "
    "down and understates the damage.",
    "strained",
    "Sanctions were widely trailed and the sample shows an unusual 3-4 March "
    "spike in volume consistent with pull-forward. The anticipation buffer "
    "removes those days from estimation.",
    ("anticipation_days", "intervention_date"),
)

_NO_INTERFERENCE = A(
    "exogenous_controls",
    "Controls unaffected by the treatment",
    "The covariates used to project the counterfactual would have taken the "
    "same path had the card networks stayed up.",
    "A control that the treatment moved is a post-treatment variable. "
    "Conditioning on it absorbs part of the payment effect into the "
    "counterfactual and biases the estimate toward zero.",
    "fails",
    "The ruble, the RTS oil and gas index and CPI all move violently after 5 "
    "March, and the card suspension was part of the same sanctions wave that "
    "moved them. The contamination slider is the direct stress test.",
    ("contamination", "controls"),
)

_COND_STATIONARITY = A(
    "cond_stationarity",
    "Conditional stationarity",
    "Given the controls, the outcome process is stable across the "
    "intervention date.",
    "The counterfactual is the pre-period process run forward. If the process "
    "itself broke at the invasion, that projection is not the counterfactual.",
    "fails",
    "The invasion is eleven trading days before the announcement. Volatility, "
    "level and the covariance with macro controls all shift. This is the "
    "weakest assumption in the whole exercise and the reason for the "
    "three-path decomposition rather than a single pre/post comparison.",
    ("contamination", "baseline_covariate_rule", "did_pre_days"),
)

_LINEAR = A(
    "linear_dynamics",
    "Linear, invertible ARMA dynamics",
    "The outcome follows a stable linear recursion in its own lags and the "
    "controls.",
    "The forecast is generated by iterating that recursion; a misspecified "
    "lag structure compounds with horizon.",
    "testable",
    "Ljung-Box on in-sample residuals is reported in the diagnostics panel. "
    "The AR block is constrained to the stationary region so the "
    "counterfactual cannot diverge.",
    ("arima_order",),
)

_HOMOSKED = A(
    "homoskedasticity",
    "Constant residual variance",
    "Innovation variance does not change across the sample.",
    "Interval coverage, not the point estimate, depends on this. Understated "
    "variance turns an insignificant effect into a significant one.",
    "fails",
    "Breusch-Pagan rejects decisively for the parametric models. Intervals "
    "here are built from a residual bootstrap rather than the analytic "
    "formula for that reason.",
    (),
)

_NO_EXTRAP = A(
    "support",
    "Counterfactual lies inside observed support",
    "The no-suspension path stays within the range of outcomes the model saw "
    "in training.",
    "Tree ensembles predict averages of training leaves. They cannot return a "
    "value outside the training range, whatever the covariates say.",
    "fails",
    "Early March volume ran above anything in 2021, so the forest's "
    "counterfactual is censored at the pre-period maximum and the payment "
    "effect it reports is a lower bound in magnitude.",
    ("rf_max_depth", "rf_lags"),
)

_IID_RESID = A(
    "independent_residuals",
    "Independent residuals",
    "Residuals carry no serial dependence.",
    "Bootstrap intervals built on resampled residuals are only valid if the "
    "residuals are exchangeable; dependence makes them far too narrow.",
    "testable",
    "The static forest fails Ljung-Box at every lag tested. Adding "
    "autoregressive lags is the repair, and the diagnostics panel shows "
    "whether it worked.",
    ("rf_lags",),
)

_PARALLEL = A(
    "parallel_trends",
    "Parallel trends, adjusted for exposure",
    "Absent the suspension, Russian and Ukrainian e-commerce would have moved "
    "together up to the exposure elasticity phi.",
    "This is what licenses reading the war channel off Ukraine instead of "
    "assuming it.",
    "strained",
    "Pre-war trends can be compared directly in the dashboard. The two "
    "economies differ in size, penetration and war role, which is exactly "
    "what phi is for, and why phi is exposed as a slider rather than fixed.",
    ("war_exposure", "did_pre_days"),
)

_CONTROL_UNTREATED = A(
    "control_untreated",
    "Control is untreated",
    "Ukrainian e-commerce was not itself hit by the Visa/MasterCard "
    "suspension in Russia.",
    "A contaminated control absorbs the treatment effect and collapses the "
    "estimate toward zero.",
    "holds",
    "The suspension applied to cards issued in or used in Russia. Ukrainian "
    "card rails stayed live; Visa and MasterCard continued operating in "
    "Ukraine throughout.",
    (),
)

_DOSE_EXOG = A(
    "dose_exogeneity",
    "Rollout timing unrelated to war news",
    "The four-day staircase in card availability is not systematically timed "
    "with the war shocks hitting demand on the same days.",
    "The dose coefficients would otherwise pick up whatever else moved on "
    "7-10 March.",
    "strained",
    "The rollout schedule was set by network operations, not by the front "
    "line, which supports the claim. But the same fortnight carried the "
    "central bank rate hike and capital controls, which did not.",
    ("dose_lags", "controls"),
)

_DOSE_LINEAR = A(
    "dose_linearity",
    "Effect proportional to availability",
    "Losing half of cross-border card acceptance does half the damage of "
    "losing all of it.",
    "The counterfactual is built by accumulating the dose response, so "
    "curvature in the response maps straight into bias.",
    "strained",
    "Plausible over a four-day window, weak over months as substitution to "
    "Mir and domestic rails sets in. The estimate is reported for the "
    "two-week window only.",
    ("dose_lags",),
)

_STATE_SPACE = A(
    "state_space_form",
    "Correct state decomposition",
    "Level, weekly cycle and control regression exhaust the systematic "
    "variation, with Gaussian errors.",
    "An omitted state component is absorbed into the level, which then drifts "
    "to fit it and carries that drift into the counterfactual.",
    "testable",
    "Residual diagnostics pass here where the other models fail, which is the "
    "main evidence for this specification. Over-specification is the standing "
    "risk: a very flexible level can track the post-shock path and report no "
    "effect.",
    ("bsts_time_varying", "bsts_seasonality", "bsts_level_scale"),
)

_NN_CAPACITY = A(
    "capacity",
    "Capacity matched to sample",
    "The network is small enough not to memorise roughly 420 pre-intervention "
    "days.",
    "An overfitted sequence model reproduces the training path and its "
    "counterfactual is noise with a confident band around it.",
    "strained",
    "Held down with weight decay, dropout and a 32-unit hidden state, and "
    "averaged over an ensemble. There is no identification argument here; the "
    "model is a flexibility benchmark, not a causal estimator on its own.",
    ("gru_hidden", "gru_members"),
)

_NN_NO_IDENT = A(
    "no_structural_identification",
    "No structural identification",
    "The model provides predictive accuracy, not a causal argument.",
    "Stated plainly so the number is not read as an estimate with the same "
    "standing as the design-based ones.",
    "fails",
    "Included to bound how much of the gap is explained by nonlinearity in "
    "the outcome-control mapping rather than by the intervention.",
    (),
)


MODEL_ASSUMPTIONS: dict[str, tuple[AssumptionSpec, ...]] = {
    "carima": (_SUTVA, _NO_ANTICIPATION, _NO_INTERFERENCE, _COND_STATIONARITY, _LINEAR, _HOMOSKED),
    "bsts": (_SUTVA, _NO_ANTICIPATION, _NO_INTERFERENCE, _STATE_SPACE, _COND_STATIONARITY),
    "rf": (_SUTVA, _NO_ANTICIPATION, _NO_INTERFERENCE, _NO_EXTRAP, _IID_RESID),
    "gru": (_SUTVA, _NO_ANTICIPATION, _NO_INTERFERENCE, _NN_CAPACITY, _NN_NO_IDENT),
    "did": (_SUTVA, _PARALLEL, _CONTROL_UNTREATED, _NO_ANTICIPATION),
    "dose": (_SUTVA, _DOSE_EXOG, _DOSE_LINEAR, _NO_INTERFERENCE),
}

#: Knobs surfaced for each model, in dashboard order. Global knobs first.
GLOBAL_KNOBS = (
    "intervention_date",
    "key_window",
    "anticipation_days",
    "contamination",
    "baseline_covariate_rule",
    "controls",
)

MODEL_KNOBS: dict[str, tuple[str, ...]] = {
    "carima": ("arima_order",),
    "bsts": ("bsts_time_varying", "bsts_seasonality", "bsts_level_scale"),
    "rf": ("rf_lags", "rf_max_depth"),
    "gru": ("gru_hidden", "gru_members"),
    "did": ("war_exposure", "did_pre_days"),
    "dose": ("dose_lags",),
}

#: Knobs that exist globally but do nothing for a given model, so the
#: dashboard can grey them out instead of letting a user move a dead control.
INERT_KNOBS: dict[str, tuple[str, ...]] = {
    # Identification runs off rollout timing, not off the control path, so the
    # counterfactual never touches the cleaned covariates. The dose regression
    # is fitted from the invasion onward rather than up to the announcement, so
    # trimming days before the announcement changes nothing either.
    "dose": ("contamination", "baseline_covariate_rule", "anticipation_days"),
    # Trends are fitted on the two outcome series over the pre-war window
    # alone, so neither the control path nor the announcement-side buffer
    # reaches this estimator.
    "did": ("contamination", "controls", "baseline_covariate_rule", "anticipation_days"),
}

STATUS_ORDER = {"fails": 0, "strained": 1, "testable": 2, "holds": 3}


def registry_payload() -> dict:
    """Everything the front end needs to render the assumption panels."""
    return {
        "knobs": {k: v.to_dict() for k, v in KNOBS.items()},
        "global_knobs": list(GLOBAL_KNOBS),
        "model_knobs": {k: list(v) for k, v in MODEL_KNOBS.items()},
        "inert_knobs": {k: list(v) for k, v in INERT_KNOBS.items()},
        "model_assumptions": {
            k: [a.to_dict() for a in sorted(v, key=lambda x: STATUS_ORDER[x.status])]
            for k, v in MODEL_ASSUMPTIONS.items()
        },
    }
