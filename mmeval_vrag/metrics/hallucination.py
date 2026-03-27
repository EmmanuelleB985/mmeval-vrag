"""Hallucination rate metric for RAG systems.

Detects claims in the generated answer that are *not* entailed by
any retrieved context.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from mmeval_vrag.config import EvalConfig
from mmeval_vrag.metrics import BaseMetric, register_metric
from mmeval_vrag.types import EvalSample
from mmeval_vrag.utils.text import sentence_split, token_overlap


def _nli_entailment_scores(
    premises: List[str],
    hypothesis: str,
    model=None,
    tokenizer=None,
    device: str = "cpu",
) -> List[float]:
    """Return P(entailment) for each (premise, hypothesis) pair.

    Falls back to token-overlap heuristic when no NLI model is loaded.
    """
    if model is not None and tokenizer is not None:
        import torch

        scores: List[float] = []
        for premise in premises:
            inputs = tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            # standard NLI label order: contradiction, neutral, entailment
            scores.append(float(probs[0, 2]))
        return scores
    # Lightweight fallback
    return [token_overlap(p, hypothesis) for p in premises]


@register_metric
class HallucinationRate(BaseMetric):
    """Fraction of generated claims *not* supported by any retrieved context.

    Lower is better (0 = no hallucination).

    Methods
    -------
    ``entailment`` (default):
        Uses a cross-encoder NLI model (e.g. ``cross-encoder/nli-deberta-v3-base``).
        Falls back to unigram overlap if the model is unavailable.
    ``llm_judge``:
        Placeholder for LLM-as-judge hallucination scoring (requires
        ``config.llm_judge`` to be set).
    """

    name = "hallucination_rate"

    def __init__(self, config: EvalConfig) -> None:
        super().__init__(config)
        self._nli_model = None
        self._nli_tokenizer = None

    def setup(self) -> None:
        if self.config.hallucination_method == "entailment":
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                model_name = "cross-encoder/nli-deberta-v3-base"
                self._nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._nli_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name
                ).to(self.config.device)
                self._nli_model.eval()
            except Exception:
                pass  # will use fallback

    def compute(self, sample: EvalSample) -> Dict[str, float]:
        claims = sentence_split(sample.generated_answer)
        if not claims or not sample.retrieved_texts:
            return {self.name: 1.0}

        hallucinated = 0
        threshold = self.config.similarity_threshold

        for claim in claims:
            scores = _nli_entailment_scores(
                sample.retrieved_texts,
                claim,
                model=self._nli_model,
                tokenizer=self._nli_tokenizer,
                device=self.config.device,
            )
            if max(scores) < threshold:
                hallucinated += 1

        return {self.name: hallucinated / len(claims)}
