"""Tests that would actually catch a wrong number.

The expensive estimators are exercised once each; the cheap ones are used to
check the accounting identities, which is where a silent sign error would hide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from payments_shock import config as C
from payments_shock.assumptions import (
    GLOBAL_KNOBS,
    KNOBS,
    MODEL_ASSUMPTIONS,
    MODEL_KNOBS,
    STATUS_ORDER,
)
from payments_shock.data import build_design, calibrate_war_exposure, load_raw
from payments_shock.decomposition import attribute, rescaled_paths
from payments_shock.diagnostics import evaluate, ljung_box
from payments_shock.models import ORDER, REGISTRY


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def design():
    a = C.Assumptions()
    a.war_exposure = calibrate_war_exposure()["phi"]
    return build_design(a)


# --- data ---------------------------------------------------------------


def test_sample_shape(raw):
    assert len(raw) == 502
    assert raw.index[0] == C.SAMPLE_START
    assert raw.index[-1] == C.SAMPLE_END
    assert not raw.isna().any().any()


def test_treatment_series_collapse(raw):
    """The rollout has to be in the data, or the dose model has nothing."""
    pre = raw.loc[C.ANNOUNCEMENT - pd.Timedelta(days=1), "NRBankinR"]
    post = raw.loc[C.ROLLOUT_COMPLETE, "NRBankinR"]
    assert pre > 30
    assert post == 0.0


def test_treatment_never_enters_controls():
    a = C.Assumptions(controls=C.DEFAULT_CONTROLS + C.TREATMENT_INTENSITY)
    d = build_design(a)
    for col in C.TREATMENT_INTENSITY:
        assert col not in d.columns


def test_dose_is_bounded_and_monotone_through_rollout(design):
    assert design.dose.min() >= 0.0
    assert design.dose.max() <= 1.0
    w = (design.index >= C.ANNOUNCEMENT) & (design.index <= C.ROLLOUT_COMPLETE)
    seq = design.dose[w]
    assert np.all(np.diff(seq) <= 1e-9), "card availability must not recover mid-rollout"
    assert design.dose[design.index > C.ROLLOUT_COMPLETE].max() == 0.0


def test_contamination_moves_the_counterfactual_covariates():
    lo = build_design(C.Assumptions(contamination=0.0))
    hi = build_design(C.Assumptions(contamination=1.0))
    w = lo.window_mask
    assert not np.allclose(lo.X_cf[w], hi.X_cf[w]), "lambda slider is inert"
    assert np.allclose(lo.X_cf[lo.train_mask], hi.X_cf[hi.train_mask]), (
        "lambda must not touch the pre-intervention sample"
    )


def test_anticipation_buffer_trims_the_training_sample():
    a0 = build_design(C.Assumptions(anticipation_days=0))
    a5 = build_design(C.Assumptions(anticipation_days=5))
    assert a0.train_mask.sum() - a5.train_mask.sum() == 5


def test_estimation_sample_is_a_contiguous_prefix(design):
    n = int(design.train_mask.sum())
    assert design.train_mask[:n].all()
    assert not design.train_mask[n:].any()


def test_war_exposure_calibration_reports_the_floor():
    cal = calibrate_war_exposure()
    assert cal["ukraine_log_change"] < -1.0, "Ukraine should collapse in the war window"
    assert cal["russia_log_change"] > 0.0, "Russia should not, before the suspension"
    assert cal["phi"] == 0.0 and cal["at_floor"] is True


# --- accounting ---------------------------------------------------------


@pytest.mark.parametrize("key", ["did", "dose"])
def test_decomposition_is_an_identity(design, key):
    p = REGISTRY[key](design.assumptions).paths(design)
    a = attribute(design, p)
    assert a.payments_pp + a.war_pp + a.secular_pp == pytest.approx(a.decline_pct, abs=1e-6)


@pytest.mark.parametrize("key", ["did", "dose"])
def test_paths_share_the_reference_level(design, key):
    p = REGISTRY[key](design.assumptions).paths(design)
    paths = rescaled_paths(design, p)
    pre = design.index < design.assumptions.intervention_date
    ref = np.where(pre)[0][-5:]
    for name in ("cf", "baseline"):
        assert paths[name][ref].mean() == pytest.approx(
            paths["observed"][ref].mean(), rel=1e-9
        )


def test_zero_exposure_collapses_did_onto_its_baseline(design):
    a = C.Assumptions(war_exposure=0.0, calibrate_exposure=False)
    d = build_design(a)
    p = REGISTRY["did"](a).paths(d)
    res = attribute(d, p)
    assert res.war_pp == pytest.approx(0.0, abs=1e-6)
    assert res.payments_share == pytest.approx(1.0, abs=1e-6)


def test_higher_exposure_moves_decline_from_payments_to_war(design):
    shares = []
    for phi in (0.0, 0.5, 1.0, 2.0):
        a = C.Assumptions(war_exposure=phi, calibrate_exposure=False)
        d = build_design(a)
        shares.append(attribute(d, REGISTRY["did"](a).paths(d)).war_pp)
    assert all(b < a for a, b in zip(shares, shares[1:])), (
        "raising war exposure must attribute more of the decline to the war"
    )


def test_observed_decline_is_stable_across_models(design):
    """Every model measures the same observed decline; only the split differs."""
    declines = []
    for key in ("did", "dose"):
        p = REGISTRY[key](design.assumptions).paths(design)
        declines.append(attribute(design, p).decline_pct)
    assert declines[0] == pytest.approx(declines[1], abs=1e-6)
    assert -60 < declines[0] < -30


# --- models -------------------------------------------------------------


@pytest.mark.parametrize("key", ORDER)
def test_model_runs_and_produces_finite_paths(design, key):
    p = REGISTRY[key](design.assumptions).paths(design)
    for arr in (p.cf, p.baseline, p.cf_sd, p.baseline_sd):
        assert np.all(np.isfinite(arr)), f"{key} produced non-finite output"
    assert np.all(p.cf_sd >= 0)
    res = attribute(design, p)
    assert -400 < res.payments_pp < 400, f"{key} counterfactual has diverged"


@pytest.mark.parametrize("key", ORDER)
def test_counterfactual_tracks_observation_before_the_intervention(design, key):
    p = REGISTRY[key](design.assumptions).paths(design)
    paths = rescaled_paths(design, p)
    pre = design.index < design.assumptions.intervention_date
    pre = pre & (design.index >= pd.Timestamp("2022-01-03"))
    err = np.abs(paths["cf"][pre] / paths["observed"][pre] - 1.0).mean()
    assert err < 0.45, f"{key} does not track the pre-period ({err:.2f})"


def test_uncertainty_widens_with_horizon(design):
    p = REGISTRY["bsts"](design.assumptions).paths(design)
    n = int(design.train_mask.sum())
    assert p.cf_sd[n + 30] > p.cf_sd[n + 1]


# --- diagnostics --------------------------------------------------------


def test_ljung_box_detects_known_dependence():
    rng = np.random.default_rng(0)
    white = rng.normal(size=400)
    ar = np.zeros(400)
    for t in range(1, 400):
        ar[t] = 0.8 * ar[t - 1] + rng.normal()
    assert ljung_box(white, 10) > 0.05
    assert ljung_box(ar, 10) < 0.01


def test_diagnostics_payload_is_complete(design):
    d = evaluate(design, REGISTRY["dose"](design.assumptions).paths(design))
    for k in ("rmse", "ljung_box_1", "ljung_box_5", "ljung_box_10", "breusch_pagan"):
        assert k in d


# --- registry -----------------------------------------------------------


def test_every_model_has_assumptions_and_knobs():
    for key in ORDER:
        assert key in MODEL_ASSUMPTIONS and MODEL_ASSUMPTIONS[key]
        assert key in MODEL_KNOBS


def test_assumption_statuses_are_known():
    for specs in MODEL_ASSUMPTIONS.values():
        for spec in specs:
            assert spec.status in STATUS_ORDER
            assert spec.statement and spec.why and spec.evidence


def test_every_referenced_knob_exists():
    named = {k for specs in MODEL_ASSUMPTIONS.values() for s in specs for k in s.knobs}
    named |= set(GLOBAL_KNOBS)
    named |= {k for v in MODEL_KNOBS.values() for k in v}
    assert named <= set(KNOBS), named - set(KNOBS)
