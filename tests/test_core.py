"""Tests for mmeval-vrag core functionality."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mmeval_vrag.config import EvalConfig, _resolve_metrics, ALL_METRICS
from mmeval_vrag.types import EvalSample, ImageInput, RetrievedItem
from mmeval_vrag.results import EvalResult, EvalResultCollection
from mmeval_vrag.evaluators.multimodal_rag import MultimodalRAGEvaluator
from mmeval_vrag.evaluators.pipeline import EvalPipeline, QueryItem
from mmeval_vrag.utils.text import sentence_split, token_overlap, ngram_overlap
from mmeval_vrag.metrics import list_metrics, get_metric_class
from mmeval_vrag.datasets.loaders import load_dataset


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def simple_sample() -> EvalSample:
    return EvalSample(
        query_text="What is the treatment for diabetes?",
        retrieved=[
            RetrievedItem(text="Diabetes is treated with insulin and lifestyle changes.", is_relevant=True, score=0.95),
            RetrievedItem(text="Regular exercise helps manage blood sugar levels.", is_relevant=True, score=0.85),
            RetrievedItem(text="The weather today is sunny and warm.", is_relevant=False, score=0.2),
        ],
        generated_answer="Diabetes is commonly treated with insulin therapy and lifestyle modifications including regular exercise.",
        reference_answer="Treatment includes insulin, diet, and exercise.",
        sample_id="test_001",
    )


@pytest.fixture
def no_context_sample() -> EvalSample:
    return EvalSample(
        query_text="What is quantum computing?",
        retrieved=[],
        generated_answer="Quantum computing uses qubits for computation.",
        sample_id="test_002",
    )


@pytest.fixture
def hallucinated_sample() -> EvalSample:
    return EvalSample(
        query_text="What causes headaches?",
        retrieved=[
            RetrievedItem(text="Headaches can be caused by stress and dehydration.", is_relevant=True),
        ],
        generated_answer="Headaches are caused by alien signals from space that disrupt brainwaves.",
        sample_id="test_003",
    )


@pytest.fixture
def config_all() -> EvalConfig:
    return EvalConfig(metrics=["all"], device="cpu")


@pytest.fixture
def config_retrieval_only() -> EvalConfig:
    return EvalConfig(
        metrics=["retrieval_precision", "retrieval_recall", "retrieval_mrr", "retrieval_ndcg"],
        device="cpu",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Config tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEvalConfig:
    def test_resolve_all(self):
        resolved = _resolve_metrics(["all"])
        assert set(resolved) == set(ALL_METRICS)

    def test_resolve_specific(self):
        resolved = _resolve_metrics(["faithfulness", "hallucination_rate"])
        assert resolved == ["faithfulness", "hallucination_rate"]

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            _resolve_metrics(["not_a_real_metric"])

    def test_default_config(self):
        cfg = EvalConfig()
        assert set(cfg.metrics) == set(ALL_METRICS)
        assert cfg.batch_size == 32
        assert cfg.device == "cpu"
        assert cfg.top_k == 5

    def test_custom_config(self):
        cfg = EvalConfig(
            metrics=["faithfulness"],
            batch_size=16,
            device="cuda",
            top_k=10,
            seed=123,
        )
        assert cfg.metrics == ["faithfulness"]
        assert cfg.batch_size == 16
        assert cfg.top_k == 10
        assert cfg.seed == 123


# ═══════════════════════════════════════════════════════════════════════════
# Types tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTypes:
    def test_eval_sample_retrieved_texts(self, simple_sample):
        texts = simple_sample.retrieved_texts
        assert len(texts) == 3
        assert "insulin" in texts[0]

    def test_eval_sample_has_ground_truth(self, simple_sample):
        assert simple_sample.has_ground_truth_relevance is True

    def test_eval_sample_no_ground_truth(self):
        sample = EvalSample(
            retrieved=[RetrievedItem(text="hello")],
            generated_answer="world",
        )
        assert sample.has_ground_truth_relevance is False

    def test_image_input_from_array(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        img = ImageInput(array=arr)
        pil = img.to_pil()
        assert pil.size == (64, 64)

    def test_image_input_no_source_raises(self):
        img = ImageInput()
        with pytest.raises(ValueError, match="needs either"):
            img.to_pil()

    def test_retrieved_item_defaults(self):
        item = RetrievedItem()
        assert item.text is None
        assert item.image is None
        assert item.score is None
        assert item.is_relevant is None
        assert item.metadata == {}


# ═══════════════════════════════════════════════════════════════════════════
# Text utilities tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTextUtils:
    def test_sentence_split_basic(self):
        text = "Hello world. This is a test. And another one!"
        sentences = sentence_split(text)
        assert len(sentences) == 3

    def test_sentence_split_empty(self):
        assert sentence_split("") == []
        assert sentence_split("   ") == []

    def test_sentence_split_single(self):
        assert len(sentence_split("Just one sentence.")) == 1

    def test_token_overlap_identical(self):
        assert token_overlap("hello world", "hello world") == 1.0

    def test_token_overlap_disjoint(self):
        assert token_overlap("hello world", "foo bar") == 0.0

    def test_token_overlap_partial(self):
        score = token_overlap("the cat sat", "the dog sat")
        assert 0.0 < score < 1.0
        # Jaccard: {the, sat} / {the, cat, sat, dog} = 2/4 = 0.5
        assert abs(score - 0.5) < 1e-6

    def test_token_overlap_empty(self):
        assert token_overlap("", "hello") == 0.0
        assert token_overlap("hello", "") == 0.0

    def test_ngram_overlap_identical(self):
        assert ngram_overlap("a b c d", "a b c d", n=2) == 1.0

    def test_ngram_overlap_disjoint(self):
        assert ngram_overlap("a b c", "x y z", n=2) == 0.0

    def test_ngram_overlap_partial(self):
        score = ngram_overlap("the cat sat on", "the dog sat on", n=2)
        assert 0.0 < score < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Metric registry tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMetricRegistry:
    def test_all_metrics_registered(self):
        registered = list_metrics()
        for m in ALL_METRICS:
            assert m in registered, f"{m} not registered"

    def test_get_metric_class(self):
        cls = get_metric_class("faithfulness")
        assert cls.name == "faithfulness"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            get_metric_class("nonexistent_metric")


# ═══════════════════════════════════════════════════════════════════════════
# Retrieval metrics tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRetrievalMetrics:
    def test_precision_perfect(self):
        sample = EvalSample(
            retrieved=[
                RetrievedItem(text="a", is_relevant=True),
                RetrievedItem(text="b", is_relevant=True),
            ],
            generated_answer="answer",
        )
        cfg = EvalConfig(metrics=["retrieval_precision"], top_k=2)
        metric = get_metric_class("retrieval_precision")(cfg)
        result = metric.compute(sample)
        assert result["retrieval_precision"] == 1.0

    def test_precision_half(self):
        sample = EvalSample(
            retrieved=[
                RetrievedItem(text="a", is_relevant=True),
                RetrievedItem(text="b", is_relevant=False),
            ],
            generated_answer="answer",
        )
        cfg = EvalConfig(metrics=["retrieval_precision"], top_k=2)
        metric = get_metric_class("retrieval_precision")(cfg)
        result = metric.compute(sample)
        assert result["retrieval_precision"] == 0.5

    def test_precision_empty(self):
        sample = EvalSample(retrieved=[], generated_answer="answer")
        cfg = EvalConfig(metrics=["retrieval_precision"], top_k=5)
        metric = get_metric_class("retrieval_precision")(cfg)
        result = metric.compute(sample)
        assert result["retrieval_precision"] == 0.0

    def test_recall_partial(self):
        sample = EvalSample(
            retrieved=[
                RetrievedItem(text="a", is_relevant=True),
                RetrievedItem(text="b", is_relevant=False),
                RetrievedItem(text="c", is_relevant=True),
            ],
            generated_answer="answer",
        )
        cfg = EvalConfig(metrics=["retrieval_recall"], top_k=2)
        metric = get_metric_class("retrieval_recall")(cfg)
        result = metric.compute(sample)
        # top-2 has 1 relevant out of 2 total relevant
        assert result["retrieval_recall"] == 0.5

    def test_mrr_first_relevant(self):
        sample = EvalSample(
            retrieved=[
                RetrievedItem(text="a", is_relevant=True),
                RetrievedItem(text="b", is_relevant=False),
            ],
            generated_answer="answer",
        )
        cfg = EvalConfig(metrics=["retrieval_mrr"], top_k=5)
        metric = get_metric_class("retrieval_mrr")(cfg)
        result = metric.compute(sample)
        assert result["retrieval_mrr"] == 1.0

    def test_mrr_second_relevant(self):
        sample = EvalSample(
            retrieved=[
                RetrievedItem(text="a", is_relevant=False),
                RetrievedItem(text="b", is_relevant=True),
            ],
            generated_answer="answer",
        )
        cfg = EvalConfig(metrics=["retrieval_mrr"], top_k=5)
        metric = get_metric_class("retrieval_mrr")(cfg)
        result = metric.compute(sample)
        assert result["retrieval_mrr"] == 0.5

    def test_mrr_none_relevant(self):
        sample = EvalSample(
            retrieved=[
                RetrievedItem(text="a", is_relevant=False),
                RetrievedItem(text="b", is_relevant=False),
            ],
            generated_answer="answer",
        )
        cfg = EvalConfig(metrics=["retrieval_mrr"], top_k=5)
        metric = get_metric_class("retrieval_mrr")(cfg)
        result = metric.compute(sample)
        assert result["retrieval_mrr"] == 0.0

    def test_ndcg_perfect(self):
        sample = EvalSample(
            retrieved=[
                RetrievedItem(text="a", is_relevant=True),
                RetrievedItem(text="b", is_relevant=True),
            ],
            generated_answer="answer",
        )
        cfg = EvalConfig(metrics=["retrieval_ndcg"], top_k=2)
        metric = get_metric_class("retrieval_ndcg")(cfg)
        result = metric.compute(sample)
        assert result["retrieval_ndcg"] == 1.0

    def test_ndcg_inverted(self):
        sample = EvalSample(
            retrieved=[
                RetrievedItem(text="a", is_relevant=False),
                RetrievedItem(text="b", is_relevant=True),
            ],
            generated_answer="answer",
        )
        cfg = EvalConfig(metrics=["retrieval_ndcg"], top_k=2)
        metric = get_metric_class("retrieval_ndcg")(cfg)
        result = metric.compute(sample)
        # DCG = 0/log2(2) + 1/log2(3) ; IDCG = 1/log2(2) + 0/log2(3)
        expected = (1 / math.log2(3)) / (1 / math.log2(2))
        assert abs(result["retrieval_ndcg"] - expected) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# Faithfulness metrics tests (token-overlap fallback)
# ═══════════════════════════════════════════════════════════════════════════


class TestFaithfulnessMetrics:
    def test_faithfulness_grounded(self, simple_sample):
        cfg = EvalConfig(metrics=["faithfulness"])
        metric = get_metric_class("faithfulness")(cfg)
        result = metric.compute(simple_sample)
        # Token overlap should be decent since answer uses similar words
        assert result["faithfulness"] > 0.0

    def test_faithfulness_no_context(self, no_context_sample):
        cfg = EvalConfig(metrics=["faithfulness"])
        metric = get_metric_class("faithfulness")(cfg)
        result = metric.compute(no_context_sample)
        assert result["faithfulness"] == 0.0

    def test_answer_relevance(self, simple_sample):
        cfg = EvalConfig(metrics=["answer_relevance"])
        metric = get_metric_class("answer_relevance")(cfg)
        result = metric.compute(simple_sample)
        assert result["answer_relevance"] > 0.0

    def test_answer_relevance_no_query(self):
        sample = EvalSample(generated_answer="hello")
        cfg = EvalConfig(metrics=["answer_relevance"])
        metric = get_metric_class("answer_relevance")(cfg)
        result = metric.compute(sample)
        assert result["answer_relevance"] == 0.0

    def test_context_relevance(self, simple_sample):
        cfg = EvalConfig(metrics=["context_relevance"])
        metric = get_metric_class("context_relevance")(cfg)
        result = metric.compute(simple_sample)
        assert result["context_relevance"] > 0.0

    def test_context_relevance_no_contexts(self, no_context_sample):
        cfg = EvalConfig(metrics=["context_relevance"])
        metric = get_metric_class("context_relevance")(cfg)
        result = metric.compute(no_context_sample)
        assert result["context_relevance"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Hallucination metric tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHallucinationMetric:
    def test_hallucination_grounded(self, simple_sample):
        cfg = EvalConfig(metrics=["hallucination_rate"], similarity_threshold=0.1)
        metric = get_metric_class("hallucination_rate")(cfg)
        result = metric.compute(simple_sample)
        # Answer reuses words from context → low hallucination
        assert result["hallucination_rate"] < 1.0

    def test_hallucination_ungrounded(self, hallucinated_sample):
        cfg = EvalConfig(metrics=["hallucination_rate"], similarity_threshold=0.3)
        metric = get_metric_class("hallucination_rate")(cfg)
        result = metric.compute(hallucinated_sample)
        # "alien signals from space" has very low overlap with context
        assert result["hallucination_rate"] > 0.0

    def test_hallucination_no_context(self, no_context_sample):
        cfg = EvalConfig(metrics=["hallucination_rate"])
        metric = get_metric_class("hallucination_rate")(cfg)
        result = metric.compute(no_context_sample)
        assert result["hallucination_rate"] == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Cross-modal metrics tests (fallback behaviour without CLIP)
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossModalMetrics:
    def test_cross_modal_no_images(self, simple_sample):
        cfg = EvalConfig(metrics=["cross_modal_alignment"])
        metric = get_metric_class("cross_modal_alignment")(cfg)
        result = metric.compute(simple_sample)
        assert result["cross_modal_alignment"] == 0.0

    def test_visual_grounding_no_images(self, simple_sample):
        cfg = EvalConfig(metrics=["visual_grounding"])
        metric = get_metric_class("visual_grounding")(cfg)
        result = metric.compute(simple_sample)
        assert result["visual_grounding"] == 0.0

    def test_multimodal_consistency_no_pairs(self, simple_sample):
        cfg = EvalConfig(metrics=["multimodal_consistency"])
        metric = get_metric_class("multimodal_consistency")(cfg)
        result = metric.compute(simple_sample)
        assert result["multimodal_consistency"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Results tests
# ═══════════════════════════════════════════════════════════════════════════


class TestResults:
    def test_eval_result_getitem(self):
        r = EvalResult(sample_id="s1", scores={"f": 0.8, "h": 0.1})
        assert r["f"] == 0.8
        assert r["h"] == 0.1

    def test_eval_result_to_dict(self):
        r = EvalResult(sample_id="s1", scores={"f": 0.8})
        d = r.to_dict()
        assert d["sample_id"] == "s1"
        assert d["scores"]["f"] == 0.8

    def test_collection_summary(self):
        col = EvalResultCollection(
            results=[
                EvalResult(sample_id="a", scores={"m1": 0.8, "m2": 0.6}),
                EvalResult(sample_id="b", scores={"m1": 0.6, "m2": 0.4}),
            ]
        )
        s = col.summary()
        assert abs(s["m1"]["mean"] - 0.7) < 1e-6
        assert abs(s["m2"]["mean"] - 0.5) < 1e-6
        assert s["m1"]["n"] == 2

    def test_collection_mean_std(self):
        col = EvalResultCollection(
            results=[
                EvalResult(sample_id="a", scores={"x": 1.0}),
                EvalResult(sample_id="b", scores={"x": 0.0}),
            ]
        )
        assert col.mean("x") == 0.5
        assert col.std("x") == 0.5

    def test_collection_to_json(self):
        col = EvalResultCollection(
            results=[EvalResult(sample_id="a", scores={"m": 0.9})]
        )
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        col.to_json(path)
        with open(path) as f:
            data = json.load(f)
        assert "summary" in data
        assert "per_sample" in data
        assert data["per_sample"][0]["scores"]["m"] == 0.9

    def test_collection_repr(self):
        col = EvalResultCollection(
            results=[EvalResult(sample_id="a", scores={"m": 0.5})]
        )
        r = repr(col)
        assert "n=1" in r
        assert "m:" in r

    def test_collection_empty_metric(self):
        col = EvalResultCollection(results=[])
        assert col.mean("anything") == 0.0
        assert col.std("anything") == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Evaluator integration tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMultimodalRAGEvaluator:
    def test_evaluate_retrieval_only(self, simple_sample, config_retrieval_only):
        evaluator = MultimodalRAGEvaluator(config=config_retrieval_only)
        results = evaluator.evaluate([simple_sample], show_progress=False)
        assert len(results.results) == 1
        scores = results.results[0].scores
        assert "retrieval_precision" in scores
        assert "retrieval_recall" in scores
        assert "retrieval_mrr" in scores
        assert "retrieval_ndcg" in scores

    def test_evaluate_multiple_samples(self, simple_sample, no_context_sample):
        cfg = EvalConfig(metrics=["faithfulness", "hallucination_rate"])
        evaluator = MultimodalRAGEvaluator(config=cfg)
        results = evaluator.evaluate(
            [simple_sample, no_context_sample], show_progress=False
        )
        assert len(results.results) == 2
        s = results.summary()
        assert "faithfulness" in s
        assert "hallucination_rate" in s

    def test_evaluate_single(self, simple_sample):
        cfg = EvalConfig(metrics=["retrieval_precision"])
        evaluator = MultimodalRAGEvaluator(config=cfg)
        result = evaluator.evaluate_single(simple_sample)
        assert isinstance(result, EvalResult)
        assert "retrieval_precision" in result.scores


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEvalPipeline:
    def test_pipeline_end_to_end(self):
        def mock_retriever(query_text=None, query_image=None, top_k=5):
            return [
                RetrievedItem(
                    text="Relevant context about the topic.",
                    is_relevant=True,
                    score=0.9,
                    metadata={"id": "doc_1"},
                ),
                RetrievedItem(
                    text="Irrelevant filler text.",
                    is_relevant=False,
                    score=0.3,
                    metadata={"id": "doc_2"},
                ),
            ]

        def mock_generator(query_text, contexts):
            return "This is a generated answer about the topic."

        pipeline = EvalPipeline(
            retriever=mock_retriever,
            generator=mock_generator,
            config=EvalConfig(metrics=["retrieval_precision", "faithfulness"]),
        )
        queries = [
            QueryItem(query_text="What is the topic?", reference_answer="The topic is X."),
            QueryItem(query_text="Another question?", relevant_ids=["doc_1"]),
        ]
        results = pipeline.run(queries, show_progress=False)
        assert len(results.results) == 2
        assert "retrieval_precision" in results.results[0].scores


# ═══════════════════════════════════════════════════════════════════════════
# Dataset loader tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDatasetLoaders:
    def test_jsonl_loader(self, tmp_path):
        jsonl = tmp_path / "test.jsonl"
        data = [
            {
                "query": "What is X?",
                "retrieved": [
                    {"text": "X is a thing.", "is_relevant": True},
                    {"text": "Unrelated.", "is_relevant": False},
                ],
                "generated_answer": "X is a thing that does stuff.",
                "reference_answer": "X is a thing.",
            },
            {
                "query": "What is Y?",
                "retrieved": [{"text": "Y is another thing."}],
                "generated_answer": "Y is another thing entirely.",
            },
        ]
        with open(jsonl, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        samples = load_dataset("jsonl", str(jsonl))
        assert len(samples) == 2
        assert samples[0].query_text == "What is X?"
        assert len(samples[0].retrieved) == 2
        assert samples[0].retrieved[0].is_relevant is True
        assert samples[1].reference_answer is None

    def test_jsonl_max_samples(self, tmp_path):
        jsonl = tmp_path / "test.jsonl"
        with open(jsonl, "w") as f:
            for i in range(10):
                f.write(json.dumps({"query": f"q{i}", "generated_answer": "a"}) + "\n")
        samples = load_dataset("jsonl", str(jsonl), max_samples=3)
        assert len(samples) == 3

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_format", "/tmp/x")


# ═══════════════════════════════════════════════════════════════════════════
# CLI tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCLI:
    def test_cli_basic(self, tmp_path):
        from mmeval_vrag.cli import main

        jsonl = tmp_path / "input.jsonl"
        out = tmp_path / "results.json"
        data = {
            "query": "What is AI?",
            "retrieved": [{"text": "AI is artificial intelligence.", "is_relevant": True}],
            "generated_answer": "AI stands for artificial intelligence.",
        }
        with open(jsonl, "w") as f:
            f.write(json.dumps(data) + "\n")

        main([str(jsonl), "-m", "retrieval_precision", "faithfulness", "-o", str(out)])
        assert out.exists()
        with open(out) as f:
            results = json.load(f)
        assert "summary" in results
