# Sanctions Effectiveness Estimator 
https://benh83.github.io/payments-shock-counterfactuals/

**Separating the Effect of Payment Sanctions and Wartime on Russian E-commerce.**

On 5 March 2022 Visa and MasterCard announced they were suspending service in
Russia. Cross-border card acceptance degraded over the next four trading days
and was gone by 10 March. Russian e-commerce volume fell 44% over the following
two weeks, as a wider range of financial sanctions took place, ending on March 23rd.

Russia had been at war since 24 February. So the primary question for anyone
assessing what the payment cutoff actually achieved is not *did volume fall,* but **how much of the e-commerce drop did the financial sanctions cause, and how much would have happened anyway.** A mistake here could mean over-attributing impact to weak sanctions or just concluding that Russian consumers simply made an unlikely collective buying decision at the same time the shocks were happening.

This repository builds six counterfactuals of the no-suspension world, states
the identifying assumptions each one needs, and lets you move those assumptions
and watch the attribution move with them. None of these are clear-cut. Data on similiar wartime sanctions combinations could let us construct a more realistic counterfactual, but that data is not available to me. Violent macro shocks make measuring things like this difficult, but through examining different methods with transparent assumptions, we can come closer to knowing the truth

---

## Results

Over **7–18 March 2022**, Russian e-commerce volume fell **43.7%** against its
pre-announcement level (32.8bn → 18.5bn, five-day reference against a three-day
window close).

Under the default assumption set, the estimators put the payments channel at
roughly **80–95% of the shock-attributable decline**, with the war channel
supplying the rest. Two estimators dissent depending on how the anticipation
buffer is set, which is the point of shipping all six rather than one.

One key fact:

> Between the invasion on 24 February and the suspension announcement on 5
> March — eleven trading days with the war live and the card rails intact —
> **Ukrainian e-commerce fell about 80%. Russian e-commerce rose 24%.**

The war on its own had not yet dented Russian online retail when the networks
pulled out. That is what pins the war-exposure elasticity at its floor and
pushes the attribution toward the payments channel. It is also, in isolation,
too convenient to trust, which is why the dashboard exposes the elasticity as a
slider and shows you what has to be true for the war to carry the decline
instead.

As even the end of this relatively short dataset shows, Russian e-commerce slowly moved up. The Mir card system continued its gradual rollout, and in the last few years Russia has created an autarkic financial technology system. Long before the Ukraine war, they had been preparing for financial independence from the west, and the war gave them the push that made it necessary. 

---

## Why six models

No single estimator is credible here. Each fails a different assumption, so the
useful output is the spread across them and the conditions under which they
converge.

| Estimator | Family | What it buys | Where it breaks |
|---|---|---|---|
| **Ukraine difference-in-differences** | Design-based | Observes the war channel on a war-exposed, payments-unexposed control instead of assuming it | Parallel trends across two very different economies |
| **Dose-response** | Continuous treatment | Reads the effect off the four-day staged collapse in card acceptance — the only identifying variation the war does not also produce | Rollout timing must be unrelated to same-week war news |
| **Bayesian structural time series** | State space | Drifting level, weekly cycle and drifting control loadings; best residual diagnostics in the set | Over-specification: a free enough level can absorb the shock and report nothing |
| **C-ARIMA** | Parametric time series | Transparent linear dynamics, the Menchetti et al. baseline | Conditional stationarity, which the invasion plainly breaks |
| **Random forest** | Nonparametric ensemble | Flexible control mapping, no functional form imposed | Cannot extrapolate past observed outcome levels; static residuals are serially correlated |
| **GRU ensemble** | Neural sequence model | Nonlinear, state-dependent response — the only one that can represent it | No identification argument at all; a flexibility benchmark, not an estimate |

Models deliberately **not** included, and why:

- **Synthetic control.** One donor series. A synthetic Russia built from a donor
  pool of one is a rescaled Ukraine, which is the DiD above with extra steps.
- **Causal forests / double ML.** Need cross-sectional treatment variation. There
  is one treated unit and no untreated Russian region in the data.
- **Markov switching.** Needs the suspension to reverse inside the sample so the
  second regime is observed. It does not.
- **VAR.** No credible second endogenous series; the authors' own note says the
  same.

---

## How the attribution works

Three worlds, one window, two differences.

```
observed    O_t    war and suspension both live
counterfact C_t    war live, card rails intact
baseline    B_t    neither shock
```

Each estimator is fitted twice: once on the sample ending at the announcement
(so the war is inside the training data and the model has priced it), then
projected on the counterfactual covariate path; and once on the strictly pre-war
sample, then projected on a frozen covariate path. The first projection carries
the war; the second does not.

The accounting is done on **declines from a common reference**, not on levels.
Reference `R` is the observed average over the five trading days before the
intervention; endpoint `E` is the average over the last three days of the
window.

```
decline       = (E_obs  - R) / R
payments      = decline_obs - decline_cf
war           = decline_cf  - decline_base
secular       = decline_base
```

Every counterfactual path is normalised onto the observed reference level before
differencing. The *level* of a long extrapolation is its least trustworthy
feature — a model that runs 8% high before the intervention runs 8% high after
it, and that bias would land squarely in the attribution. Normalising throws the
level away and keeps the shape, which is the part carrying the causal signal.

Shares are taken over the shock-attributable part (`payments + war`), not over
the raw decline, so a drifting baseline cannot push a share past 100% for
arithmetic reasons. When the two channels come out with opposite signs the
output is flagged `shares_well_posed: false` and the dashboard tells you to read
percentage points instead of shares.

---

## The assumption controls

Every control in the dashboard is an identifying assumption, not a display
option. The two that carry the argument:

**Covariate contamination (λ).** The ruble, the RTS oil and gas index and CPI
all move violently after 5 March — and the card suspension was part of the same
sanctions wave that moved them. Conditioning a counterfactual on them is
conditioning on a post-treatment variable, which launders payment effects into
the counterfactual and biases the estimate toward zero. λ walks those series
back toward their pre-announcement path: 0 treats them as pure war signal, 1
treats them as fully sanctions-driven.

**Anticipation buffer.** Volume on 2–4 March ran at 35–40bn against a February
norm nearer 25bn. That is pull-forward ahead of widely-trailed sanctions, not
ordinary demand. Left in the estimation sample it teaches every model that early
March was a boom, and the counterfactual inherits the boom. The default trims
three trading days; set it to zero to see what that decision is worth (for the
GRU and BSTS, a great deal).

Also exposed: intervention date (announcement vs. first observable degradation),
assessment window, no-shock covariate rule, control set, and the
specification knobs for each estimator. The war-exposure elasticity φ is a
slider on the DiD panel.

Controls that do nothing for a given model are greyed out rather than left live —
the dose-response model never touches the cleaned covariates, so λ is inert
there, and the dashboard says so.

---

## Data

`data/russia_ecom_daily.csv` — 502 trading days, 4 January 2021 to 30 December
2022.

| Column | Series |
|---|---|
| `ecom` | Russian e-commerce volume (mn) — the outcome |
| `UKR_ecom` | Ukrainian e-commerce volume (mn) — war-exposed control |
| `consum_index` | Russian consumer & retail equity index |
| `RTS_OGI` | RTS oil & gas index |
| `USD_XR` | USD/RUB |
| `CPI` | Russian CPI, % change |
| `vix_close` | CBOE VIX — global risk |
| `RCardinR` | Russian-issued cards used in Russia |
| `Dem_mthly` | Domestic demand indicator (monthly, interpolated) |
| `xr` | Ruble strength index — collinear with `USD_XR`, excluded from the default control set |
| `NRBankinR` | Non-Russian-issued cards usable in Russia |
| `Rbankoutside` | Russian-issued cards usable outside Russia |

`NRBankinR` and `Rbankoutside` **are the treatment.** They trace cross-border
card acceptance falling 39.60 / 274.0 on 4 March to 0 / 0 on 10 March. They are
never selectable as controls; only the dose-response model touches them, and it
treats them as the intervention.

Two modelling choices worth stating up front, both in `data.py`:

- Covariates are standardised on **full-sample** moments, not pre-war moments.
  Pre-war variation is tiny next to what 2022 did to these series — the ruble's
  log level sits roughly seventeen pre-war standard deviations out in March. Scaled
  on that yardstick every control is a vast outlier, linear projections explode,
  and the contamination slider loses all bite because every setting clips at the
  same bound. Full-sample scaling is a monotone rescaling of regressors; it uses
  no outcome information.
- Standardised covariates are winsorised at ±6.

---

## Install and run

```bash
git clone <repo>
cd payments-shock-counterfactuals
pip install -e .

# live dashboard, refits on every control change
streamlit run dashboard/app.py

# precompute the assumption grid, then open the standalone dashboard
python -m payments_shock.run_grid --out results/grid.json --workers 2
open dashboard/static/index.html
```

Programmatic use:

```python
from payments_shock import Assumptions, run

out = run(Assumptions(contamination=0.5, anticipation_days=3))
print(out["did"]["attribution"]["payments_share"])
```

Requires Python 3.10+, PyTorch (CPU is fine), scikit-learn, pandas, scipy.
The C-ARIMA and BSTS estimators are implemented directly in PyTorch — maximum
likelihood on the conditional ARMA recursion and on the exact Kalman filter
respectively — rather than wrapped from a stats package, so the assumption knobs
reach the estimator rather than stopping at an API.

---

## Diagnostics

Reported per model, per assumption set:

- In-sample RMSE (log scale)
- Ljung-Box at 1, 5 and 10 lags — structure left in the residuals means the
  counterfactual is missing dynamics and the bands are too narrow
- Breusch-Pagan — a moving variance breaks interval coverage but not the point
  estimate
- Diebold-Mariano for pairwise in-sample forecast comparison
- Placebo intervention at a fake date 120 days earlier; a well-behaved estimator
  should report roughly zero damage there

Only in-sample comparison is possible. Out-of-sample here *is* the
counterfactual, and there is no second reality to score it against.

---

## Layout

```
src/payments_shock/
  config.py          timeline, series definitions, the Assumptions dataclass
  data.py            three covariate worlds, design matrices, exposure calibration
  assumptions.py     the assumption registry — what each model needs and why
  decomposition.py   the war/payments split
  diagnostics.py     residual tests, DM, placebo
  run_grid.py        precompute the dashboard grid
  models/
    base.py          two-fit protocol shared by the projection estimators
    carima.py        C-ARIMA, PyTorch MLE
    bsts.py          Kalman-filter state space, PyTorch MLE
    forest.py        random forest with block-bootstrap bands
    gru.py           GRU ensemble
    did.py           Ukraine difference-in-differences
    dose.py          distributed-lag dose-response on rollout intensity
dashboard/
  app.py             Streamlit, live refit
  static/index.html  standalone, grid-driven
```

---

## What this does not establish

The estimates are conditional on the assumption set displayed beside them.
Several of those assumptions fail outright on this data — conditional
stationarity across an invasion, and exogeneity of macro controls that the same
sanctions package moved. The models are run anyway, with the failure stated,
because a bounded and labelled estimate is more useful than silence. Read the
spread across the six, not any one of them.

Substitution is out of scope. The dataset ends before the Mir network and
domestic rails absorbed much of the displaced volume, so nothing here speaks to
how durable the effect was past the two-week window. 

This was built as an extension of an original abstract *How Much Damage did Major
Payment Services Suspensions Cause to Russian E-commerce?* (Hinrichs & Yu). 
Methodological lineage: Brodersen et al. (2015) for the state-space
counterfactual; Menchetti, Cipollini & Mealli (2023) for C-ARIMA.
