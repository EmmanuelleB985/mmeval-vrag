"""Classic information-retrieval metrics adapted for multimodal RAG."""

from __future__ import annotations

import math
from typing import Dict, List

from mmeval_vrag.metrics import BaseMetric, register_metric
from mmeval_vrag.types import EvalSample


def _relevance_labels(sample: EvalSample, k: int) -> List[int]:
    """Return binary relevance list for the top-k retrieved items."""
    labels: List[int] = []
    for item in sample.retrieved[:k]:
        if item.is_relevant is not None:
            labels.append(int(item.is_relevant))
        else:
            # fallback: treat score > 0 as relevant
            labels.append(1 if (item.score or 0) > 0 else 0)
    return labels


@register_metric
class RetrievalPrecision(BaseMetric):
    """Precision@K over retrieved contexts."""

    name = "retrieval_precision"

    def compute(self, sample: EvalSample) -> Dict[str, float]:
        k = self.config.top_k
        labels = _relevance_labels(sample, k)
        if not labels:
            return {self.name: 0.0}
        return {self.name: sum(labels) / len(labels)}


@register_metric
class RetrievalRecall(BaseMetric):
    """Recall@K — fraction of all relevant items that appear in top-K."""

    name = "retrieval_recall"

    def compute(self, sample: EvalSample) -> Dict[str, float]:
        k = self.config.top_k
        labels = _relevance_labels(sample, k)
        total_relevant = sum(1 for item in sample.retrieved if getattr(item, "is_relevant", False))
        if total_relevant == 0:
            return {self.name: 0.0}
        return {self.name: sum(labels) / total_relevant}


@register_metric
class RetrievalMRR(BaseMetric):
    """Mean Reciprocal Rank over retrieved contexts."""

    name = "retrieval_mrr"

    def compute(self, sample: EvalSample) -> Dict[str, float]:
        k = self.config.top_k
        labels = _relevance_labels(sample, k)
        for i, rel in enumerate(labels):
            if rel:
                return {self.name: 1.0 / (i + 1)}
        return {self.name: 0.0}


@register_metric
class RetrievalNDCG(BaseMetric):
    """Normalised Discounted Cumulative Gain @ K."""

    name = "retrieval_ndcg"

    @staticmethod
    def _dcg(relevances: List[int]) -> float:
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

    def compute(self, sample: EvalSample) -> Dict[str, float]:
        k = self.config.top_k
        labels = _relevance_labels(sample, k)
        if not labels or max(labels) == 0:
            return {self.name: 0.0}
        dcg = self._dcg(labels)
        ideal = self._dcg(sorted(labels, reverse=True))
        return {self.name: dcg / ideal if ideal > 0 else 0.0}
