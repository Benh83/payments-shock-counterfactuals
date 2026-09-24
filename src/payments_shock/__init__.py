"""Counterfactual attribution for the March 2022 Russian payments cutoff.

Separates the effect of the Visa/MasterCard suspension on Russian e-commerce
from the effect of the war that was already under way.
"""

from .assumptions import MODEL_ASSUMPTIONS, registry_payload
from .config import Assumptions
from .data import build_design, calibrate_war_exposure, load_raw
from .decomposition import attribute, daily_series
from .models import ORDER, REGISTRY

__version__ = "0.1.0"

__all__ = [
    "Assumptions",
    "build_design",
    "load_raw",
    "calibrate_war_exposure",
    "attribute",
    "daily_series",
    "REGISTRY",
    "ORDER",
    "MODEL_ASSUMPTIONS",
    "registry_payload",
    "run",
]


def run(assumptions: Assumptions | None = None, models: list[str] | None = None) -> dict:
    """Fit the selected models and return attribution plus diagnostics."""
    from .diagnostics import evaluate

    a = assumptions or Assumptions()
    if a.calibrate_exposure:
        a.war_exposure = calibrate_war_exposure()["phi"]
    d = build_design(a)
    out = {}
    for key in models or ORDER:
        model = REGISTRY[key](a)
        p = model.paths(d)
        out[key] = {
            "attribution": attribute(d, p).to_dict(),
            "diagnostics": evaluate(d, p),
            "series": daily_series(d, p),
        }
    return out
