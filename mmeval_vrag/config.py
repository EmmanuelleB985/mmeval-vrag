"""Configuration for evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

# ---------------------------------------------------------------------------
# Metric registry – canonical names the framework understands
# ---------------------------------------------------------------------------
ALL_METRICS = [
    "retrieval_precision",
    "retrieval_recall",
    "retrieval_mrr",
    "retrieval_ndcg",
    "cross_modal_alignment",
    "hallucination_rate",
    "faithfulness",
    "answer_relevance",
    "context_relevance",
    "visual_grounding",
    "multimodal_consistency",
]


def _resolve_metrics(names: List[str]) -> List[str]:
    """Expand ``"all"`` and validate metric names."""
    if "all" in names:
        return list(ALL_METRICS)
    unknown = set(names) - set(ALL_METRICS)
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}. Available: {ALL_METRICS}")
    return list(names)


@dataclass
class EvalConfig:
    """Top-level configuration for an evaluation run.

    Parameters
    ----------
    metrics : list of str
        Which metrics to compute. Pass ``["all"]`` for every metric.
    batch_size : int
        Batch size for embedding / model calls.
    device : str
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, …).
    embedding_model : str
        Name or path of sentence-transformer model for text embeddings.
    clip_model : str
        Name or path of CLIP model for cross-modal similarity.
    llm_judge : str or None
        Identifier for the LLM used as an automated judge
        (e.g. ``"gpt-4"`` or ``"claude-3-opus"``). ``None`` disables
        LLM-as-judge metrics.
    llm_judge_api_key : str or None
        API key for the judge model. Falls back to env vars if ``None``.
    top_k : int
        Number of retrieved items to consider for retrieval metrics.
    similarity_threshold : float
        Cosine similarity threshold for binary relevance decisions.
    hallucination_method : str
        Strategy for hallucination detection:
        ``"entailment"`` (NLI-based) or ``"llm_judge"``.
    cache_embeddings : bool
        If ``True``, cache computed embeddings to disk.
    cache_dir : str
        Directory for cached embeddings.
    seed : int
        Random seed for reproducibility.
    extra : dict
        Arbitrary extra configuration forwarded to custom metrics.
    """

    metrics: List[str] = field(default_factory=lambda: ["all"])
    batch_size: int = 32
    device: str = "cpu"

    # Embedding / CLIP models
    embedding_model: str = "all-MiniLM-L6-v2"
    clip_model: str = "openai/clip-vit-base-patch32"

    # LLM-as-judge
    llm_judge: Optional[str] = None
    llm_judge_api_key: Optional[str] = None

    # Retrieval settings
    top_k: int = 5
    similarity_threshold: float = 0.75

    # Hallucination
    hallucination_method: Literal["entailment", "llm_judge"] = "entailment"

    # Caching
    cache_embeddings: bool = False
    cache_dir: str = ".mmeval_cache"

    # Reproducibility
    seed: int = 42

    # Catch-all
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metrics = _resolve_metrics(self.metrics)
