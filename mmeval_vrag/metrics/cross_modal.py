"""Cross-modal alignment, visual grounding, and multimodal consistency.

These metrics are the unique differentiator of *mmeval-vrag*: they evaluate
how well a multimodal RAG system bridges the gap between visual and textual
modalities, using CLIP-family models for zero-shot cross-modal scoring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from mmeval_vrag.config import EvalConfig
from mmeval_vrag.metrics import BaseMetric, register_metric
from mmeval_vrag.types import EvalSample, ImageInput


# ---------------------------------------------------------------------------
# Shared CLIP helper
# ---------------------------------------------------------------------------


class _CLIPScorer:
    """Thin wrapper around a CLIP model for image-text similarity."""

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.device: str = "cpu"

    def load(self, model_name: str, device: str) -> bool:
        try:
            from transformers import CLIPModel, CLIPProcessor

            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.model.eval()
            self.device = device
            return True
        except Exception:
            return False

    @property
    def available(self) -> bool:
        return self.model is not None

    def score(self, images: List[ImageInput], texts: List[str]) -> np.ndarray:
        """Return a (n_images, n_texts) cosine-similarity matrix."""
        import torch

        pil_images = [img.to_pil() for img in images]
        inputs = self.processor(
            text=texts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            img_emb = outputs.image_embeds  # (n_images, d)
            txt_emb = outputs.text_embeds  # (n_texts, d)

        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        sim = (img_emb @ txt_emb.T).cpu().numpy()
        return sim


# Module-level singleton so multiple metrics share one model load.
_clip = _CLIPScorer()


# ---------------------------------------------------------------------------
# CrossModalAlignment
# ---------------------------------------------------------------------------


@register_metric
class CrossModalAlignment(BaseMetric):
    """Measures alignment between retrieved images and the textual query.

    For each retrieved image, compute CLIP cosine similarity with the query
    text and return the **mean** score.  When no CLIP model is available,
    falls back to 0 (the metric is skipped gracefully).
    """

    name = "cross_modal_alignment"

    def setup(self) -> None:
        if not _clip.available:
            _clip.load(self.config.clip_model, self.config.device)

    def compute(self, sample: EvalSample) -> Dict[str, float]:
        images = sample.retrieved_images
        if not images or not sample.query_text or not _clip.available:
            return {self.name: 0.0}
        sim = _clip.score(images, [sample.query_text])  # (n, 1)
        return {self.name: float(sim[:, 0].mean())}


# ---------------------------------------------------------------------------
# VisualGrounding
# ---------------------------------------------------------------------------


@register_metric
class VisualGrounding(BaseMetric):
    """How well the generated answer is grounded in retrieved images.

    Computes CLIP similarity between each retrieved image and the generated
    answer, then returns the max score (best grounding evidence).
    """

    name = "visual_grounding"

    def setup(self) -> None:
        if not _clip.available:
            _clip.load(self.config.clip_model, self.config.device)

    def compute(self, sample: EvalSample) -> Dict[str, float]:
        images = sample.retrieved_images
        if not images or not sample.generated_answer or not _clip.available:
            return {self.name: 0.0}
        sim = _clip.score(images, [sample.generated_answer])
        return {self.name: float(sim[:, 0].max())}


# ---------------------------------------------------------------------------
# MultimodalConsistency
# ---------------------------------------------------------------------------


@register_metric
class MultimodalConsistency(BaseMetric):
    """Consistency between textual and visual contexts retrieved together.

    For each retrieved item that has *both* an image and text, compute
    the CLIP similarity between them.  The mean gives a sense of whether
    the retriever is returning coherent multimodal chunks.
    """

    name = "multimodal_consistency"

    def setup(self) -> None:
        if not _clip.available:
            _clip.load(self.config.clip_model, self.config.device)

    def compute(self, sample: EvalSample) -> Dict[str, float]:
        if not _clip.available:
            return {self.name: 0.0}

        pairs = [
            (item.image, item.text)
            for item in sample.retrieved
            if item.image is not None and item.text
        ]
        if not pairs:
            return {self.name: 0.0}

        images, texts = zip(*pairs)
        sims: List[float] = []
        for img, txt in zip(images, texts):
            sim = _clip.score([img], [txt])
            sims.append(float(sim[0, 0]))
        return {self.name: float(np.mean(sims))}
