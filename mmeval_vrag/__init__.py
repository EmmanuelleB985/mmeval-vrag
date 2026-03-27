"""
mmeval-vrag: Evaluation Framework for Multimodal Vision-Language RAG Systems
=============================================================================

Unified evaluation of retrieval quality, hallucination, faithfulness,
and cross-modal alignment for vision-language RAG pipelines.

Quick Start
-----------
>>> from mmeval_vrag import MultimodalRAGEvaluator, EvalConfig
>>> evaluator = MultimodalRAGEvaluator(config=EvalConfig(metrics=["all"]))
>>> results = evaluator.evaluate(samples)
>>> results.summary()
"""

__version__ = "0.1.0"

from mmeval_vrag.config import EvalConfig
from mmeval_vrag.evaluators.multimodal_rag import MultimodalRAGEvaluator
from mmeval_vrag.evaluators.pipeline import EvalPipeline
from mmeval_vrag.results import EvalResult, EvalResultCollection

__all__ = [
    "EvalConfig",
    "MultimodalRAGEvaluator",
    "EvalPipeline",
    "EvalResult",
    "EvalResultCollection",
]
