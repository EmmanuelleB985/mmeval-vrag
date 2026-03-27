"""Main evaluator that orchestrates metric computation."""

from __future__ import annotations

from typing import List, Optional

from tqdm import tqdm

from mmeval_vrag.config import EvalConfig
from mmeval_vrag.metrics import BaseMetric, get_metric_class
from mmeval_vrag.results import EvalResult, EvalResultCollection
from mmeval_vrag.types import EvalSample


class MultimodalRAGEvaluator:
    """One-stop evaluator for multimodal vision-language RAG pipelines.

    Usage
    -----
    >>> from mmeval_vrag import MultimodalRAGEvaluator, EvalConfig
    >>> evaluator = MultimodalRAGEvaluator(
    ...     config=EvalConfig(metrics=["faithfulness", "cross_modal_alignment"])
    ... )
    >>> results = evaluator.evaluate(samples)
    >>> print(results.summary())

    Parameters
    ----------
    config : EvalConfig
        Evaluation configuration.
    """

    def __init__(self, config: Optional[EvalConfig] = None) -> None:
        self.config = config or EvalConfig()
        self._metrics: List[BaseMetric] = []
        self._setup_done = False

    # ---- lifecycle ----------------------------------------------------------

    def _ensure_setup(self) -> None:
        if self._setup_done:
            return
        for name in self.config.metrics:
            cls = get_metric_class(name)
            instance = cls(self.config)
            instance.setup()
            self._metrics.append(instance)
        self._setup_done = True

    # ---- public API ---------------------------------------------------------

    def evaluate(
        self,
        samples: List[EvalSample],
        *,
        show_progress: bool = True,
    ) -> EvalResultCollection:
        """Run all configured metrics on *samples*.

        Returns an :class:`EvalResultCollection` with per-sample scores and
        aggregate statistics.
        """
        self._ensure_setup()
        collection = EvalResultCollection()

        iterator = tqdm(samples, desc="Evaluating", disable=not show_progress)
        for sample in iterator:
            scores: dict[str, float] = {}
            details: dict = {}
            for metric in self._metrics:
                result = metric.compute(sample)
                scores.update(result)
            collection.results.append(
                EvalResult(
                    sample_id=sample.sample_id or str(len(collection.results)),
                    scores=scores,
                    details=details,
                )
            )
        return collection

    def evaluate_single(self, sample: EvalSample) -> EvalResult:
        """Evaluate a single sample (convenience wrapper)."""
        return self.evaluate([sample], show_progress=False).results[0]
