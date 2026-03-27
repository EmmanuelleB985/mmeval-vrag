"""Core data types for evaluation samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


@dataclass
class ImageInput:
    """An image that can be lazily loaded.

    Provide *either* ``path`` (loaded on first access) or ``array``
    (already in memory).
    """

    path: Optional[Union[str, Path]] = None
    array: Optional[np.ndarray] = None
    _pil_image: Any = field(default=None, repr=False, compare=False)

    def to_pil(self):
        """Return a PIL Image, loading from disk if needed."""
        if self._pil_image is not None:
            return self._pil_image
        from PIL import Image

        if self.array is not None:
            self._pil_image = Image.fromarray(self.array)
        elif self.path is not None:
            self._pil_image = Image.open(self.path).convert("RGB")
        else:
            raise ValueError("ImageInput needs either `path` or `array`.")
        return self._pil_image


@dataclass
class RetrievedItem:
    """A single retrieved context item (text, image, or both).

    Parameters
    ----------
    text : str or None
        Retrieved text passage.
    image : ImageInput or None
        Retrieved image.
    score : float or None
        Retrieval score (e.g. cosine similarity).
    metadata : dict
        Arbitrary metadata (source, page number, …).
    is_relevant : bool or None
        Ground-truth relevance label, if available.
    """

    text: Optional[str] = None
    image: Optional[ImageInput] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_relevant: Optional[bool] = None


@dataclass
class EvalSample:
    """One evaluation example.

    This is the atomic unit fed into :class:`MultimodalRAGEvaluator`.

    Parameters
    ----------
    query_text : str or None
        The user's textual query.
    query_image : ImageInput or None
        An optional image accompanying the query.
    retrieved : list of RetrievedItem
        Contexts returned by the retrieval stage.
    generated_answer : str
        The model's generated answer.
    reference_answer : str or None
        Gold-standard answer for reference-based metrics.
    reference_image : ImageInput or None
        Gold-standard image for visual-grounding metrics.
    sample_id : str
        Unique identifier for logging / debugging.
    metadata : dict
        Extra fields (domain, difficulty, …).
    """

    query_text: Optional[str] = None
    query_image: Optional[ImageInput] = None
    retrieved: List[RetrievedItem] = field(default_factory=list)
    generated_answer: str = ""
    reference_answer: Optional[str] = None
    reference_image: Optional[ImageInput] = None
    sample_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- convenience helpers ------------------------------------------------

    @property
    def retrieved_texts(self) -> List[str]:
        return [r.text for r in self.retrieved if r.text]

    @property
    def retrieved_images(self) -> List[ImageInput]:
        return [r.image for r in self.retrieved if r.image]

    @property
    def has_ground_truth_relevance(self) -> bool:
        return any(r.is_relevant is not None for r in self.retrieved)
