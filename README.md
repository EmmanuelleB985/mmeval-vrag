# mmeval-vrag

**Evaluation framework for Multimodal Vision-Language RAG systems.**

Measure retrieval quality, hallucination, faithfulness, and cross-modal alignment in one unified pipeline.

---

## Why mmeval-vrag?

Existing RAG evaluation tools focus on text-only pipelines. Real-world systems increasingly retrieve **images alongside text** — medical scans with clinical notes, product photos with descriptions, diagrams with documentation. `mmeval-vrag` is purpose-built for this multimodal setting:

- **11 metrics** spanning retrieval, generation, and cross-modal alignment
- **Graceful degradation** — works with CPU-only token overlap, scales up with sentence-transformers, CLIP, and NLI models
- **Pipeline evaluation** — plug in your retriever + generator and benchmark end-to-end
- **JSONL + VQA loaders** — start evaluating in minutes with standard formats
- **Extensible** — register custom metrics with a single decorator

## Installation

```bash
# Core (numpy + Pillow only)
pip install mmeval-vrag

# With sentence-transformers for embedding-based metrics
pip install mmeval-vrag[transformers]

# With PyTorch + CLIP for cross-modal metrics
pip install mmeval-vrag[torch]

# Everything
pip install mmeval-vrag[full]
```

## Quick Start

```python
from mmeval_vrag import MultimodalRAGEvaluator, EvalConfig
from mmeval_vrag.types import EvalSample, RetrievedItem

sample = EvalSample(
    query_text="What does the chest X-ray show?",
    retrieved=[
        RetrievedItem(
            text="Bilateral infiltrates consistent with pneumonia.",
            is_relevant=True,
        ),
    ],
    generated_answer="The X-ray shows bilateral infiltrates indicating pneumonia.",
    reference_answer="Bilateral infiltrates indicating pneumonia.",
)

evaluator = MultimodalRAGEvaluator(
    config=EvalConfig(metrics=["faithfulness", "hallucination_rate", "retrieval_precision"])
)
results = evaluator.evaluate([sample])
print(results.summary())
```

## Metrics

| Category | Metric | What it measures |
|---|---|---|
| **Retrieval** | `retrieval_precision` | Fraction of top-K items that are relevant |
| | `retrieval_recall` | Fraction of all relevant items in top-K |
| | `retrieval_mrr` | Reciprocal rank of first relevant item |
| | `retrieval_ndcg` | Normalised DCG accounting for rank positions |
| **Generation** | `faithfulness` | Are generated claims supported by context? |
| | `hallucination_rate` | Fraction of unsupported claims (lower = better) |
| | `answer_relevance` | Similarity between answer and query |
| | `context_relevance` | Relevance of retrieved passages to query |
| **Cross-Modal** | `cross_modal_alignment` | CLIP similarity: retrieved images ↔ query text |
| | `visual_grounding` | CLIP similarity: retrieved images ↔ generated answer |
| | `multimodal_consistency` | CLIP similarity within (image, text) pairs |

## End-to-End Pipeline Evaluation

Evaluate a live retriever + generator without pre-computing samples:

```python
from mmeval_vrag.evaluators.pipeline import EvalPipeline, QueryItem

pipeline = EvalPipeline(
    retriever=my_retriever,   # (query_text, query_image, top_k) → List[RetrievedItem]
    generator=my_generator,   # (query_text, contexts) → str
    config=EvalConfig(metrics=["all"]),
)

results = pipeline.run([
    QueryItem(query_text="Describe the tumor.", relevant_ids=["doc_42"]),
])
```

## CLI

```bash
# Evaluate from a JSONL file
mmeval-vrag samples.jsonl -m faithfulness hallucination_rate -o results.json

# All metrics
mmeval-vrag samples.jsonl -m all --device cuda
```

JSONL format (one object per line):
```json
{
  "query": "What is shown in the image?",
  "retrieved": [{"text": "A lesion is visible.", "is_relevant": true}],
  "generated_answer": "The image shows a lesion.",
  "reference_answer": "A lesion."
}
```

## Custom Metrics

```python
from mmeval_vrag.metrics import BaseMetric, register_metric

@register_metric
class MyCustomMetric(BaseMetric):
    name = "my_custom_metric"

    def compute(self, sample):
        score = len(sample.generated_answer) / 100  # toy example
        return {self.name: min(score, 1.0)}
```

Then use it: `EvalConfig(metrics=["my_custom_metric", "faithfulness"])`.

## Fallback Behaviour

| Component available | Faithfulness / Relevance | Hallucination | Cross-modal |
|---|---|---|---|
| Core only (numpy) | Token overlap (Jaccard) | Token overlap | Skipped (returns 0) |
| + sentence-transformers | Embedding cosine sim | Token overlap | Skipped |
| + transformers (NLI) | Embedding cosine sim | NLI entailment | Skipped |
| + transformers (CLIP) | Embedding cosine sim | NLI entailment | CLIP cosine sim |

## Export & Analysis

```python
# Summary statistics
results.summary()  # {metric: {mean, std, median, min, max, n}}

# Per-sample DataFrame
df = results.to_dataframe()

# JSON export
results.to_json("results.json")
```

## Project Structure

```
mmeval-vrag/
├── mmeval_vrag/
│   ├── __init__.py          # Public API
│   ├── config.py            # EvalConfig + metric registry
│   ├── types.py             # EvalSample, RetrievedItem, ImageInput
│   ├── results.py           # EvalResult, EvalResultCollection
│   ├── cli.py               # CLI entry point
│   ├── evaluators/
│   │   ├── multimodal_rag.py  # Main evaluator
│   │   └── pipeline.py        # End-to-end pipeline evaluator
│   ├── metrics/
│   │   ├── __init__.py        # BaseMetric + registry
│   │   ├── retrieval.py       # Precision, Recall, MRR, NDCG
│   │   ├── faithfulness.py    # Faithfulness, Answer/Context Relevance
│   │   ├── hallucination.py   # Hallucination Rate
│   │   └── cross_modal.py     # CLIP-based cross-modal metrics
│   ├── datasets/
│   │   └── loaders.py         # JSONL + VQA dataset loaders
│   └── utils/
│       └── text.py            # Sentence splitting, token overlap
├── tests/
│   └── test_core.py
├── examples/
│   ├── quickstart.py
│   └── pipeline_eval.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## Contributing

Contributions welcome! Please open an issue or PR on [GitHub](https://github.com/EmmanuelleB985/mmeval-vrag).

```bash
git clone https://github.com/EmmanuelleB985/mmeval-vrag.git
cd mmeval-vrag
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{bourigault2025mmeval,
  author = {Bourigault, Emmanuelle},
  title = {mmeval-vrag: Evaluation Framework for Multimodal Vision-Language RAG Systems},
  year = {2025},
  url = {https://github.com/EmmanuelleB985/mmeval-vrag},
}
```
