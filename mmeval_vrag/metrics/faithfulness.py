"""Faithfulness, answer relevance, and context relevance metrics.

These metrics assess whether the generated answer is grounded in the
retrieved contexts and whether the contexts are relevant to the query.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from mmeval_vrag.config import EvalConfig
from mmeval_vrag.metrics import BaseMetric, register_metric
from mmeval_vrag.types import EvalSample
from mmeval_vrag.utils.text import (
    sentence_split,
    token_overlap,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embedding_similarity(
    text_a: str,
    text_b: str,
    model=None,
) -> float:
    """Cosine similarity between two texts using sentence-transformers.

    Falls back to token overlap when the model is unavailable.
    """
    if model is not None:
        emb = model.encode([text_a, text_b], convert_to_numpy=True)
        cos = float(
            np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-9)
        )
        return cos
    return token_overlap(text_a, text_b)


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


@register_metric
class Faithfulness(BaseMetric):
    """Measures how well the generated answer is supported by retrieved context.

    Strategy
    --------
    1. Split the generated answer into *claims* (sentences).
    2. For each claim, compute its maximum overlap / similarity with
       any retrieved passage.
    3. Average across claims → faithfulness score in [0, 1].

    When ``sentence-transformers`` is installed and the model can be loaded
    the metric uses embedding similarity; otherwise it falls back to
    unigram-overlap (still useful as a lightweight signal).
    """

    name = "faithfulness"

    def __init__(self, config: EvalConfig) -> None:
        super().__init__(config)
        self._model = None

    def setup(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.config.embedding_model, device=self.config.device
            )
        except Exception:
            self._model = None

    def compute(self, sample: EvalSample) -> Dict[str, float]:
        claims = sentence_split(sample.generated_answer)
        if not claims or not sample.retrieved_texts:
            return {self.name: 0.0}

        scores: List[float] = []
        for claim in claims:
            best = max(
                _embedding_similarity(claim, ctx, self._model) for ctx in sample.retrieved_texts
            )
            scores.append(best)
        return {self.name: float(np.mean(scores))}


# ---------------------------------------------------------------------------
# Answer Relevance
# ---------------------------------------------------------------------------


@register_metric
class AnswerRelevance(BaseMetric):
    """Similarity between the generated answer and the original query."""

    name = "answer_relevance"

    def __init__(self, config: EvalConfig) -> None:
        super().__init__(config)
        self._model = None

    def setup(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.config.embedding_model, device=self.config.device
            )
        except Exception:
            self._model = None

    def compute(self, sample: EvalSample) -> Dict[str, float]:
        if not sample.query_text or not sample.generated_answer:
            return {self.name: 0.0}
        score = _embedding_similarity(sample.query_text, sample.generated_answer, self._model)
        return {self.name: score}


# ---------------------------------------------------------------------------
# Context Relevance
# ---------------------------------------------------------------------------


@register_metric
class ContextRelevance(BaseMetric):
    """Average relevance of retrieved passages to the query."""

    name = "context_relevance"

    def __init__(self, config: EvalConfig) -> None:
        super().__init__(config)
        self._model = None

    def setup(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.config.embedding_model, device=self.config.device
            )
        except Exception:
            self._model = None

    def compute(self, sample: EvalSample) -> Dict[str, float]:
        if not sample.query_text or not sample.retrieved_texts:
            return {self.name: 0.0}
        scores = [
            _embedding_similarity(sample.query_text, ctx, self._model)
            for ctx in sample.retrieved_texts
        ]
        return {self.name: float(np.mean(scores))}
