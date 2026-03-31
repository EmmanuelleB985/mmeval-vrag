"""
Metric implementations and registry.

Every metric subclasses :class:`BaseMetric` and is registered via the
``@register_metric`` decorator so the evaluator can look them up by name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Type

from mmeval_vrag.config import EvalConfig
from mmeval_vrag.types import EvalSample

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseMetric(ABC):
    """Base class for all metrics.

    Subclasses must implement :meth:`compute` which receives a single
    :class:`EvalSample` and returns a dict of score(s).
    """

    name: str = ""

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    def setup(self) -> None:
        """Optional one-time setup (load models, etc.)."""

    @abstractmethod
    def compute(self, sample: EvalSample) -> Dict[str, float]:
        """Return ``{metric_name: score}`` for *sample*."""
        ...

    def compute_batch(self, samples: List[EvalSample]) -> List[Dict[str, float]]:
        """Override for batched computation (default loops)."""
        return [self.compute(s) for s in samples]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_METRIC_REGISTRY: Dict[str, Type[BaseMetric]] = {}


def register_metric(cls: Type[BaseMetric]) -> Type[BaseMetric]:
    """Class decorator that registers a metric by its ``name``."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a `name` class attribute.")
    _METRIC_REGISTRY[cls.name] = cls
    return cls


def get_metric_class(name: str) -> Type[BaseMetric]:
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Metric '{name}' not registered. Available: {list(_METRIC_REGISTRY)}")
    return _METRIC_REGISTRY[name]


def list_metrics() -> List[str]:
    return sorted(_METRIC_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Import concrete metrics so they self-register
# ---------------------------------------------------------------------------
from mmeval_vrag.metrics.cross_modal import (  # noqa: E402, F401
    CrossModalAlignment,
    MultimodalConsistency,
    VisualGrounding,
)
from mmeval_vrag.metrics.faithfulness import (  # noqa: E402, F401
    AnswerRelevance,
    ContextRelevance,
    Faithfulness,
)
from mmeval_vrag.metrics.hallucination import HallucinationRate  # noqa: E402, F401
from mmeval_vrag.metrics.retrieval import (  # noqa: E402, F401
    RetrievalMRR,
    RetrievalNDCG,
    RetrievalPrecision,
    RetrievalRecall,
)
