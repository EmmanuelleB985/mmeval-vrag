"""
Quickstart: Evaluate a multimodal RAG system in 10 lines
=========================================================

This example shows how to create evaluation samples and run the full
suite of metrics with mmeval-vrag.
"""

from mmeval_vrag import EvalConfig, MultimodalRAGEvaluator
from mmeval_vrag.types import EvalSample, RetrievedItem

# ── 1. Build evaluation samples ──────────────────────────────────────────

samples = [
    EvalSample(
        query_text="What does the chest X-ray show?",
        retrieved=[
            RetrievedItem(
                text="The chest X-ray reveals bilateral infiltrates consistent with pneumonia.",
                is_relevant=True,
                score=0.92,
            ),
            RetrievedItem(
                text="Patient history includes diabetes and hypertension.",
                is_relevant=False,
                score=0.45,
            ),
        ],
        generated_answer=(
            "The chest X-ray shows bilateral infiltrates, which are "
            "consistent with a diagnosis of pneumonia."
        ),
        reference_answer="Bilateral infiltrates indicating pneumonia.",
        sample_id="xray_001",
    ),
    EvalSample(
        query_text="Describe the MRI findings for this patient.",
        retrieved=[
            RetrievedItem(
                text="MRI shows a 2cm lesion in the right temporal lobe.",
                is_relevant=True,
                score=0.88,
            ),
            RetrievedItem(
                text="The lesion appears hyperintense on T2-weighted images.",
                is_relevant=True,
                score=0.85,
            ),
            RetrievedItem(
                text="Weather forecast for London: partly cloudy.",
                is_relevant=False,
                score=0.12,
            ),
        ],
        generated_answer=(
            "The MRI reveals a 2cm lesion in the right temporal lobe "
            "that appears hyperintense on T2-weighted images."
        ),
        reference_answer="2cm hyperintense lesion in right temporal lobe.",
        sample_id="mri_002",
    ),
]

# ── 2. Configure and run ─────────────────────────────────────────────────

# Use text-only metrics (no CLIP needed for this demo)
config = EvalConfig(
    metrics=[
        "retrieval_precision",
        "retrieval_recall",
        "retrieval_mrr",
        "retrieval_ndcg",
        "faithfulness",
        "hallucination_rate",
        "answer_relevance",
        "context_relevance",
    ],
    top_k=3,
    similarity_threshold=0.15,  # lower threshold for token-overlap fallback
)

evaluator = MultimodalRAGEvaluator(config=config)
results = evaluator.evaluate(samples)

# ── 3. Inspect results ───────────────────────────────────────────────────

print(results)
print()

# Per-metric breakdown
for metric, stats in results.summary().items():
    print(f"{metric:30s}  mean={stats['mean']:.3f}  std={stats['std']:.3f}")

print()

# Per-sample detail
for r in results.results:
    print(f"[{r.sample_id}]  scores={r.scores}")

# Export to JSON
results.to_json("quickstart_results.json")
print("\nResults saved to quickstart_results.json")
