"""Model registry."""

from .base import CounterfactualModel, Paths
from .bsts import BSTS
from .carima import CARIMA
from .did import UkraineDiD
from .dose import DoseResponse
from .forest import Forest
from .gru import GRUForecaster

REGISTRY: dict[str, type[CounterfactualModel]] = {
    m.key: m for m in (CARIMA, BSTS, Forest, GRUForecaster, UkraineDiD, DoseResponse)
}

#: Dashboard display order: design-based first, then statistical, then learned.
ORDER = ["did", "dose", "bsts", "carima", "rf", "gru"]

__all__ = [
    "REGISTRY",
    "ORDER",
    "CounterfactualModel",
    "Paths",
    "CARIMA",
    "BSTS",
    "Forest",
    "GRUForecaster",
    "UkraineDiD",
    "DoseResponse",
]
