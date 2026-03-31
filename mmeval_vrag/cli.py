"""Minimal CLI for quick evaluation runs."""

from __future__ import annotations

import argparse
import json

from mmeval_vrag.config import ALL_METRICS, EvalConfig
from mmeval_vrag.evaluators.multimodal_rag import MultimodalRAGEvaluator
from mmeval_vrag.types import EvalSample, RetrievedItem


def _load_samples(path: str):
    """Load samples from a JSON-Lines file.

    Expected schema per line::

        {
            "query": "...",
            "retrieved": [{"text": "...", "is_relevant": true}, ...],
            "generated_answer": "...",
            "reference_answer": "..."  // optional
        }
    """
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            retrieved = [
                RetrievedItem(
                    text=r.get("text"),
                    score=r.get("score"),
                    is_relevant=r.get("is_relevant"),
                )
                for r in obj.get("retrieved", [])
            ]
            samples.append(
                EvalSample(
                    query_text=obj.get("query", ""),
                    retrieved=retrieved,
                    generated_answer=obj.get("generated_answer", ""),
                    reference_answer=obj.get("reference_answer"),
                    sample_id=obj.get("id", str(i)),
                )
            )
    return samples


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mmeval-vrag",
        description="Evaluate a multimodal RAG system from a JSONL file.",
    )
    parser.add_argument("input", help="Path to JSONL file with evaluation samples.")
    parser.add_argument(
        "-m",
        "--metrics",
        nargs="+",
        default=["all"],
        choices=["all"] + ALL_METRICS,
        help="Metrics to compute (default: all).",
    )
    parser.add_argument("-o", "--output", help="Path to write JSON results.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    config = EvalConfig(
        metrics=args.metrics,
        device=args.device,
        top_k=args.top_k,
    )
    evaluator = MultimodalRAGEvaluator(config=config)
    samples = _load_samples(args.input)
    results = evaluator.evaluate(samples)

    print(results)
    if args.output:
        results.to_json(args.output)
        print(f"\nDetailed results written to {args.output}")


if __name__ == "__main__":
    main()
