"""Shared model interface.

Every counterfactual estimator in this package answers the same three
questions with the same machinery:

1. What would Russian e-commerce have done with the war but no payment
   suspension?  -> ``cf``
2. What would it have done with neither shock?  -> ``baseline``
3. What did it actually do?  -> ``observed``

The estimator is fitted twice. Once on the sample ending at the announcement
(so the war is inside the training data and the model has learned to price it),
then projected forward on the counterfactual covariate path. Once on the
strictly pre-war sample, then projected forward on the frozen covariate path.
The first projection carries the war, the second does not, and the gap between
them is the war channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import Design


@dataclass
class Paths:
    """Three worlds, on the model's transformed scale, plus uncertainty."""

    observed: np.ndarray
    cf: np.ndarray
    cf_sd: np.ndarray
    baseline: np.ndarray
    baseline_sd: np.ndarray
    fitted: np.ndarray          # in-sample fit over the estimation window
    residuals: np.ndarray       # in-sample residuals, for diagnostics
    train_mask: np.ndarray


class CounterfactualModel:
    """Base class. Subclasses implement ``_fit`` and ``_predict``."""

    key: str = "base"
    name: str = "Base"
    family: str = ""
    #: Short line the dashboard prints under the model name.
    tagline: str = ""
    #: Whether this estimator needs the two-fit protocol above. Models that
    #: identify the war channel directly (DiD, dose-response) set this False
    #: and override :meth:`paths`.
    two_stage = True

    def __init__(self, assumptions):
        self.a = assumptions
        self._fitted = False

    # -- to implement ------------------------------------------------------
    def _fit(self, y: np.ndarray, X: np.ndarray, mask: np.ndarray) -> None:
        raise NotImplementedError

    def _predict(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray):
        """Return (mean, sd) over the full sample."""
        raise NotImplementedError

    # -- protocol ----------------------------------------------------------
    def paths(self, d: Design) -> Paths:
        self._fit(d.y, d.X_obs, d.train_mask)
        cf, cf_sd = self._predict(d.X_cf, d.y, d.train_mask)
        fitted, _ = self._predict(d.X_obs, d.y, d.train_mask)
        resid = d.y[d.train_mask] - fitted[d.train_mask]

        self._fit(d.y, d.X_obs, d.prewar_mask)
        base, base_sd = self._predict(d.X_base, d.y, d.prewar_mask)

        return Paths(
            observed=d.y.copy(),
            cf=cf,
            cf_sd=cf_sd,
            baseline=base,
            baseline_sd=base_sd,
            fitted=fitted,
            residuals=resid,
            train_mask=d.train_mask,
        )


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def add_const(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])
