"""Live assumption dashboard (Streamlit).

    streamlit run dashboard/app.py

Every control on the left is an identifying assumption, not a display option.
Moving one refits the selected estimator and recomputes the attribution, so the
number on screen is always the number that assumption set implies.

For a zero-dependency, shareable version driven by a precomputed grid, see
``dashboard/static/index.html`` and ``payments_shock.run_grid``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from payments_shock import config as C  # noqa: E402
from payments_shock.assumptions import (  # noqa: E402
    GLOBAL_KNOBS,
    INERT_KNOBS,
    KNOBS,
    MODEL_ASSUMPTIONS,
    MODEL_KNOBS,
    STATUS_ORDER,
)
from payments_shock.data import build_design, calibrate_war_exposure  # noqa: E402
from payments_shock.decomposition import attribute, rescaled_paths  # noqa: E402
from payments_shock.diagnostics import evaluate  # noqa: E402
from payments_shock.models import ORDER, REGISTRY  # noqa: E402

st.set_page_config(page_title="Payments Shock Attribution", layout="wide")

STATUS_COLOR = {
    "holds": "#1baf7a",
    "testable": "#2a78d6",
    "strained": "#eda100",
    "fails": "#e34948",
}


@st.cache_data(show_spinner=False)
def _calibration():
    return calibrate_war_exposure()


@st.cache_data(show_spinner="Refitting estimator...")
def _fit(model_key: str, payload: dict):
    a = C.Assumptions(**{k: v for k, v in payload.items()})
    d = build_design(a)
    p = REGISTRY[model_key](a).paths(d)
    return attribute(d, p).to_dict(), evaluate(d, p), {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in rescaled_paths(d, p).items()
    }, [str(x.date()) for x in d.index], d.dose.tolist()


def sidebar() -> tuple[str, dict]:
    st.sidebar.title("Assumptions")
    model_key = st.sidebar.selectbox(
        "Estimator", ORDER, format_func=lambda k: REGISTRY[k].name
    )
    inert = set(INERT_KNOBS.get(model_key, ()))
    kw: dict = {}

    st.sidebar.caption("Identification")
    for kid in GLOBAL_KNOBS:
        knob = KNOBS[kid]
        disabled = kid in inert
        label = knob.label + (" (inert for this model)" if disabled else "")
        if kid == "controls":
            kw["controls"] = st.sidebar.multiselect(
                label, C.DEFAULT_CONTROLS, default=C.DEFAULT_CONTROLS,
                help=knob.help, disabled=disabled,
            )
        elif kid == "intervention_date":
            kw["intervention_date"] = pd.Timestamp(
                st.sidebar.selectbox(label, list(knob.options), help=knob.help)
            )
        elif kid == "key_window":
            choice = st.sidebar.selectbox(label, list(knob.options), help=knob.help)
            lo, hi = choice.split(":")
            kw["key_window"] = (pd.Timestamp(lo), pd.Timestamp(hi))
        elif knob.kind == "slider":
            kw[kid] = st.sidebar.slider(
                label, float(knob.low), float(knob.high), float(knob.default),
                float(knob.step), help=knob.help, disabled=disabled,
            )
        elif knob.kind == "int":
            kw[kid] = st.sidebar.slider(
                label, int(knob.low), int(knob.high), int(knob.default),
                int(knob.step), help=knob.help, disabled=disabled,
            )
        else:
            kw[kid] = st.sidebar.selectbox(label, list(knob.options), help=knob.help)

    st.sidebar.caption(f"{REGISTRY[model_key].name} specification")
    for kid in MODEL_KNOBS[model_key]:
        knob = KNOBS[kid]
        if kid == "arima_order":
            kw["arima_order"] = tuple(
                int(x) for x in st.sidebar.selectbox(
                    knob.label, list(knob.options), help=knob.help
                ).split(",")
            )
        elif knob.kind == "toggle":
            kw[kid] = st.sidebar.checkbox(knob.label, bool(knob.default), help=knob.help)
        elif knob.kind == "slider":
            kw[kid] = st.sidebar.slider(
                knob.label, float(knob.low), float(knob.high), float(knob.default),
                float(knob.step), help=knob.help,
            )
        else:
            kw[kid] = st.sidebar.slider(
                knob.label, int(knob.low), int(knob.high), int(knob.default),
                int(knob.step), help=knob.help,
            )

    if "war_exposure" in kw:
        kw["calibrate_exposure"] = False
    else:
        kw["war_exposure"] = _calibration()["phi"]
        kw["calibrate_exposure"] = False
    return model_key, kw


def main() -> None:
    model_key, kw = sidebar()
    cal = _calibration()

    st.title("Payments shock attribution")
    st.write(
        "How much of the collapse in Russian e-commerce over 7-18 March 2022 came "
        "from the international card networks pulling out, and how much from the "
        "war that was already running?"
    )

    att, diag, paths, dates, dose = _fit(model_key, kw)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observed decline", f"{att['decline_pct']:.1f}%")
    c2.metric("Payments channel", f"{att['payments_pp']:.1f} pp",
              help="Percentage points of the decline attributable to the suspension")
    c3.metric("War channel", f"{att['war_pp']:.1f} pp")
    c4.metric("Payments share", f"{att['payments_share'] * 100:.0f}%")

    if not att["shares_well_posed"]:
        st.warning(
            "The two channels push in opposite directions under this assumption set, "
            "so the shares do not partition the decline. Read the percentage-point "
            "figures, not the shares."
        )

    idx = pd.to_datetime(dates)
    keep = (idx >= "2022-01-03") & (idx <= "2022-06-30")
    frame = pd.DataFrame(
        {
            "Observed": np.array(paths["observed"])[keep],
            "War only (counterfactual)": np.array(paths["cf"])[keep],
            "Neither shock (baseline)": np.array(paths["baseline"])[keep],
        },
        index=idx[keep],
    )
    st.line_chart(frame, height=380)

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Identifying assumptions")
        specs = sorted(MODEL_ASSUMPTIONS[model_key], key=lambda x: STATUS_ORDER[x.status])
        for spec in specs:
            colour = STATUS_COLOR[spec.status]
            with st.expander(f"{spec.label}  -  {spec.status.upper()}", expanded=spec.status == "fails"):
                st.markdown(
                    f"<span style='color:{colour};font-weight:600'>{spec.status.upper()}</span>",
                    unsafe_allow_html=True,
                )
                st.write(f"**Assumption.** {spec.statement}")
                st.write(f"**Why it is needed.** {spec.why}")
                st.write(f"**Evidence here.** {spec.evidence}")
                if spec.knobs:
                    st.caption(
                        "Stress it with: "
                        + ", ".join(KNOBS[k].label for k in spec.knobs if k in KNOBS)
                    )

    with right:
        st.subheader("Diagnostics")
        st.dataframe(
            pd.DataFrame(
                {
                    "value": {
                        "In-sample RMSE (log)": round(diag["rmse"], 4),
                        "Ljung-Box p (1 lag)": round(diag["ljung_box_1"], 4),
                        "Ljung-Box p (5 lags)": round(diag["ljung_box_5"], 4),
                        "Ljung-Box p (10 lags)": round(diag["ljung_box_10"], 4),
                        "Breusch-Pagan p": round(diag["breusch_pagan"], 4),
                    }
                }
            ),
            use_container_width=True,
        )
        st.caption(
            "Low Ljung-Box p means structure left in the residuals: the "
            "counterfactual is missing dynamics and the bands are too narrow. "
            "Low Breusch-Pagan p means the variance moves, which breaks interval "
            "coverage but not the point estimate."
        )

        st.subheader("Window arithmetic")
        st.write(
            pd.Series(
                {
                    "Reference level (5d pre-shock)": f"{att['reference_mn']:,.0f} mn",
                    "Observed at window close": f"{att['observed_end_mn']:,.0f} mn",
                    "War-only counterfactual": f"{att['counterfactual_end_mn']:,.0f} mn",
                    "No-shock baseline": f"{att['baseline_end_mn']:,.0f} mn",
                    "Cumulative volume lost": f"{att['lost_total_mn']:,.0f} mn",
                    "  of which payments": f"{att['lost_payments_mn']:,.0f} mn",
                    "  of which war": f"{att['lost_war_mn']:,.0f} mn",
                }
            )
        )

    st.divider()
    st.caption(
        f"War exposure calibrated on {cal['n_days']} trading days between the invasion "
        f"and the announcement: Ukrainian volume {cal['ukraine_log_change'] * 100:.0f} log %, "
        f"Russian {cal['russia_log_change'] * 100:+.0f} log %, implied elasticity "
        f"{cal['phi_raw']:.2f}"
        + (" (clipped to zero)" if cal["at_floor"] else "")
        + "."
    )


if __name__ == "__main__":
    main()
