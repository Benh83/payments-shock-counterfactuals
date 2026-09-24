"""Precompute the assumption grid the dashboard reads.

Refitting six estimators on every slider move is not viable in a browser, so
the grid is computed once and shipped as JSON. The sweep is deliberately not a
full factorial: it crosses the two knobs that carry the identification
argument (contamination and the anticipation buffer) and moves everything else
one at a time around the default. That is what a sensitivity table is for --
showing whether the conclusion survives each assumption in turn, not
enumerating 180 combinations nobody will read.

    python -m payments_shock.run_grid --out results/grid.json
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .assumptions import registry_payload
from .data import build_design, calibrate_war_exposure, load_raw
from .decomposition import attribute, daily_series
from .diagnostics import evaluate
from .models import ORDER, REGISTRY

CONTAMINATION = [0.0, 0.25, 0.5, 0.75, 1.0]
ANTICIPATION = [0, 3, 5]
PHI_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

SERIES_FROM = "2022-01-03"
SERIES_TO = "2022-06-30"


def _base() -> C.Assumptions:
    a = C.Assumptions()
    a.war_exposure = calibrate_war_exposure()["phi"]
    return a


def _with(**kw) -> C.Assumptions:
    a = _base()
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _label(**kw) -> str:
    return ";".join(f"{k}={v}" for k, v in sorted(kw.items()))


def build_configs() -> list[dict]:
    cfgs: list[dict] = []

    def add(cid, label, group, models, **kw):
        cfgs.append(
            {"id": cid, "label": label, "group": group, "models": models, "kw": kw}
        )

    # Identification cross: the two knobs that decide what the number means.
    for lam in CONTAMINATION:
        for ant in ANTICIPATION:
            add(
                f"lam{lam}_ant{ant}",
                f"lambda={lam}, anticipation={ant}d",
                "identification",
                list(ORDER),
                contamination=lam,
                anticipation_days=ant,
            )

    # Timing and framing, one at a time.
    add("date_0307", "Intervention = 7 March", "timing", list(ORDER),
        intervention_date=pd.Timestamp("2022-03-07"))
    add("win_0305", "Window = 5-16 March", "timing", list(ORDER),
        key_window=(pd.Timestamp("2022-03-05"), pd.Timestamp("2022-03-16")))
    add("win_0310", "Window = 10-23 March", "timing", list(ORDER),
        key_window=(pd.Timestamp("2022-03-10"), pd.Timestamp("2022-03-23")))
    add("base_drift", "No-shock controls drift", "timing", list(ORDER),
        baseline_covariate_rule="drift")

    # Control set: drop the most contaminated macro series entirely.
    add("ctrl_clean", "Drop ruble, equities, CPI", "controls", list(ORDER),
        controls=["UKR_ecom", "vix_close", "RCardinR", "Dem_mthly"])
    add("ctrl_min", "Ukraine only", "controls", list(ORDER),
        controls=["UKR_ecom"])

    # Model-specific specification sweeps.
    for order, name in [((1, 0, 0), "AR(1)"), ((2, 0, 1), "ARMA(2,1)"), ((1, 1, 1), "ARIMA(1,1,1)")]:
        add(f"arima_{name}", f"C-ARIMA {name}", "model", ["carima"], arima_order=order)
    add("bsts_static", "BSTS, fixed loadings", "model", ["bsts"], bsts_time_varying=False)
    add("bsts_noseas", "BSTS, no weekly cycle", "model", ["bsts"], bsts_seasonality=False)
    add("bsts_loose", "BSTS, loose level prior", "model", ["bsts"], bsts_level_scale=0.20)
    add("bsts_tight", "BSTS, tight level prior", "model", ["bsts"], bsts_level_scale=0.01)
    for lag in (1, 3):
        add(f"rf_lag{lag}", f"Forest with {lag} AR lag(s)", "model", ["rf"], rf_lags=lag)
    add("rf_shallow", "Forest depth 5", "model", ["rf"], rf_max_depth=5)
    add("rf_deep", "Forest depth 14", "model", ["rf"], rf_max_depth=14)
    add("gru_small", "GRU 16 units", "model", ["gru"], gru_hidden=16)
    add("gru_large", "GRU 64 units", "model", ["gru"], gru_hidden=64)
    for lag in (0, 2, 10):
        add(f"dose_l{lag}", f"Dose lags = {lag}", "model", ["dose"], dose_lags=lag)
    for pre in (20, 120, 180):
        add(f"did_pre{pre}", f"DiD pre-period {pre}d", "model", ["did"], did_pre_days=pre)
    for phi in PHI_GRID:
        add(f"did_phi{phi}", f"DiD exposure phi={phi}", "exposure", ["did"],
            war_exposure=phi, calibrate_exposure=False)

    return cfgs


def _run_one(cfg: dict) -> tuple[str, dict]:
    a = _with(**cfg["kw"])
    d = build_design(a)
    out = {}
    for key in cfg["models"]:
        t0 = time.time()
        p = REGISTRY[key](a).paths(d)
        out[key] = {
            "attribution": attribute(d, p).to_dict(),
            "diagnostics": evaluate(d, p),
            "series": daily_series(d, p, start=SERIES_FROM),
            "seconds": round(time.time() - t0, 2),
        }
        s = out[key]["series"]
        keep = [i for i, x in enumerate(s["dates"]) if x <= SERIES_TO]
        for k in ("dates", "observed", "cf", "cf_lo", "cf_hi", "baseline", "dose"):
            s[k] = [s[k][i] for i in keep]
    return cfg["id"], out


def context() -> dict:
    raw = load_raw()
    cal = calibrate_war_exposure(raw)
    w = (raw.index >= C.KEY_WINDOW[0]) & (raw.index <= C.KEY_WINDOW[1])
    pre = raw.index < C.ANNOUNCEMENT
    ref = float(raw.loc[pre, C.OUTCOME].iloc[-5:].mean())
    end = float(raw.loc[w, C.OUTCOME].iloc[-3:].mean())
    return {
        "generated": pd.Timestamp.now("UTC").isoformat(),
        "n_obs": int(len(raw)),
        "sample": [str(raw.index[0].date()), str(raw.index[-1].date())],
        "war_start": str(C.WAR_START.date()),
        "announcement": str(C.ANNOUNCEMENT.date()),
        "rollout_complete": str(C.ROLLOUT_COMPLETE.date()),
        "key_window": [str(C.KEY_WINDOW[0].date()), str(C.KEY_WINDOW[1].date())],
        "reference_mn": ref,
        "endpoint_mn": end,
        "observed_decline_pct": (end - ref) / ref * 100.0,
        "exposure_calibration": cal,
        "series_labels": C.SERIES_LABELS,
        "treatment_intensity": C.TREATMENT_INTENSITY,
        "default_controls": C.DEFAULT_CONTROLS,
        "contamination_grid": CONTAMINATION,
        "anticipation_grid": ANTICIPATION,
        "phi_grid": PHI_GRID,
        "covariates": {
            col: {
                "label": C.SERIES_LABELS.get(col, col),
                "dates": [str(x.date()) for x in raw.index[raw.index >= SERIES_FROM]],
                "values": np.round(
                    raw.loc[raw.index >= SERIES_FROM, col].to_numpy(), 3
                ).tolist(),
            }
            for col in ["ecom", "UKR_ecom", "USD_XR", "RTS_OGI", "CPI", "vix_close",
                        "NRBankinR", "Rbankoutside"]
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/grid.json")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    cfgs = build_configs()
    print(f"{len(cfgs)} configurations")
    results: dict[str, dict] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (cid, res) in enumerate(ex.map(_run_one, cfgs), 1):
            results[cid] = res
            print(f"[{i}/{len(cfgs)}] {cid}  ({time.time() - t0:.0f}s)", flush=True)

    payload = {
        "context": context(),
        "registry": registry_payload(),
        "models": [
            {
                "key": k,
                "name": REGISTRY[k].name,
                "family": REGISTRY[k].family,
                "tagline": REGISTRY[k].tagline,
            }
            for k in ORDER
        ],
        "configs": [{k: v for k, v in c.items() if k != "kw"} | {"kw": _jsonable(c["kw"])}
                    for c in cfgs],
        "results": results,
        "base_config": "lam0.0_ant3",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":"), default=str))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) in {time.time() - t0:.0f}s")


def _jsonable(kw: dict) -> dict:
    out = {}
    for k, v in kw.items():
        if isinstance(v, tuple) and v and isinstance(v[0], pd.Timestamp):
            out[k] = [str(x.date()) for x in v]
        elif isinstance(v, pd.Timestamp):
            out[k] = str(v.date())
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
