"""Pipeline evaluator — wraps a live RAG pipeline and evaluates it end-to-end.

Instead of pre-computed samples, you supply a *retriever* and a *generator*
callable, plus a dataset of (query, ground_truth) pairs, and EvalPipeline
runs everything and produces scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from mmeval_vrag.config import EvalConfig
from mmeval_vrag.evaluators.multimodal_rag import MultimodalRAGEvaluator
from mmeval_vrag.results import EvalResultCollection
from mmeval_vrag.types import EvalSample, ImageInput, RetrievedItem


# ---------------------------------------------------------------------------
# Protocols for user-supplied components
# ---------------------------------------------------------------------------


class Retriever(Protocol):
    """Any callable that takes a query and returns retrieved items."""

    def __call__(
        self,
        query_text: Optional[str] = None,
        query_image: Optional[ImageInput] = None,
        top_k: int = 5,
    ) -> List[RetrievedItem]:
        ...


class Generator(Protocol):
    """Any callable that takes a query + contexts and returns an answer."""

    def __call__(
        self,
        query_text: Optional[str],
        contexts: List[RetrievedItem],
    ) -> str:
        ...


# ---------------------------------------------------------------------------
# Query dataset item
# ---------------------------------------------------------------------------


@dataclass
class QueryItem:
    """A single benchmark item: a query plus optional ground truth."""

    query_text: Optional[str] = None
    query_image: Optional[ImageInput] = None
    reference_answer: Optional[str] = None
    reference_image: Optional[ImageInput] = None
    relevant_ids: Optional[List[str]] = None
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ---------------------------------------------------------------------------
# EvalPipeline
# ---------------------------------------------------------------------------


class EvalPipeline:
    """End-to-end evaluation: retriever → generator → metrics.

    Usage
    -----
    >>> pipeline = EvalPipeline(
    ...     retriever=my_retriever,
    ...     generator=my_generator,
    ...     config=EvalConfig(metrics=["all"]),
    ... )
    >>> results = pipeline.run(query_items)
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        config: Optional[EvalConfig] = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.config = config or EvalConfig()
        self._evaluator = MultimodalRAGEvaluator(config=self.config)

    def run(
        self,
        queries: Sequence[QueryItem],
        *,
        show_progress: bool = True,
    ) -> EvalResultCollection:
        """Execute the full pipeline and return evaluation results."""
        samples: List[EvalSample] = []

        for i, q in enumerate(queries):
            # 1. Retrieve
            retrieved = self.retriever(
                query_text=q.query_text,
                query_image=q.query_image,
                top_k=self.config.top_k,
            )

            # Tag relevance if ground-truth IDs are provided
            if q.relevant_ids:
                for item in retrieved:
                    doc_id = item.metadata.get("id", "")
                    item.is_relevant = doc_id in q.relevant_ids

            # 2. Generate
            answer = self.generator(q.query_text, retrieved)

            # 3. Package
            samples.append(
                EvalSample(
                    query_text=q.query_text,
                    query_image=q.query_image,
                    retrieved=retrieved,
                    generated_answer=answer,
                    reference_answer=q.reference_answer,
                    reference_image=q.reference_image,
                    sample_id=f"q_{i}",
                    metadata=q.metadata,
                )
            )

        return self._evaluator.evaluate(samples, show_progress=show_progress)
