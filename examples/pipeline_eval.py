"""
End-to-End Pipeline Evaluation
===============================

This example shows how to use ``EvalPipeline`` to evaluate a live
retriever + generator system, rather than pre-computed samples.
"""

from mmeval_vrag import EvalConfig
from mmeval_vrag.evaluators.pipeline import EvalPipeline, QueryItem
from mmeval_vrag.types import RetrievedItem

# ── Mock retriever & generator (replace with your own) ───────────────────

MOCK_CORPUS = {
    "doc_1": "Transformers use self-attention to process sequences in parallel.",
    "doc_2": "CLIP aligns images and text in a shared embedding space.",
    "doc_3": "RAG combines retrieval with generation for grounded answers.",
    "doc_4": "The weather in Paris is mild in spring.",
}


def my_retriever(query_text=None, query_image=None, top_k=5):
    """Toy retriever: returns docs sorted by word overlap."""
    if not query_text:
        return []
    q_tokens = set(query_text.lower().split())
    scored = []
    for doc_id, text in MOCK_CORPUS.items():
        overlap = len(q_tokens & set(text.lower().split()))
        scored.append((doc_id, text, overlap))
    scored.sort(key=lambda x: x[2], reverse=True)
    return [
        RetrievedItem(text=text, score=float(s), metadata={"id": did})
        for did, text, s in scored[:top_k]
    ]


def my_generator(query_text, contexts):
    """Toy generator: concatenates top context with the query."""
    if contexts and contexts[0].text:
        return f"Based on the evidence: {contexts[0].text}"
    return "I don't have enough information to answer."


# ── Run the pipeline ─────────────────────────────────────────────────────

pipeline = EvalPipeline(
    retriever=my_retriever,
    generator=my_generator,
    config=EvalConfig(
        metrics=[
            "retrieval_precision",
            "retrieval_mrr",
            "faithfulness",
            "hallucination_rate",
            "answer_relevance",
        ],
        top_k=3,
        similarity_threshold=0.1,
    ),
)

queries = [
    QueryItem(
        query_text="How do transformers process sequences?",
        reference_answer="Transformers use self-attention for parallel processing.",
        relevant_ids=["doc_1"],
    ),
    QueryItem(
        query_text="What does CLIP do with images and text?",
        reference_answer="CLIP maps images and text into a shared space.",
        relevant_ids=["doc_2"],
    ),
    QueryItem(
        query_text="How does RAG improve generation?",
        reference_answer="RAG grounds generation in retrieved documents.",
        relevant_ids=["doc_3"],
    ),
]

results = pipeline.run(queries)
print(results)
print()

for metric, stats in results.summary().items():
    print(f"{metric:30s}  mean={stats['mean']:.3f}")
