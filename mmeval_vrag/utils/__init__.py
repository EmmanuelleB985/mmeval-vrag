"""Lightweight text utilities (no heavy dependencies)."""

from __future__ import annotations

import re
from typing import List


def sentence_split(text: str) -> List[str]:
    """Split text into sentences (simple regex-based)."""
    if not text:
        return []
    # Split on sentence-ending punctuation followed by space or end-of-string
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def token_overlap(text_a: str, text_b: str) -> float:
    """Unigram overlap score in [0, 1] (Jaccard-like)."""
    a_tokens = set(text_a.lower().split())
    b_tokens = set(text_b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(intersection) / len(union)


def ngram_overlap(text_a: str, text_b: str, n: int = 2) -> float:
    """N-gram overlap score in [0, 1]."""

    def _ngrams(text: str, n: int):
        tokens = text.lower().split()
        return set(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    a_ng = _ngrams(text_a, n)
    b_ng = _ngrams(text_b, n)
    if not a_ng or not b_ng:
        return 0.0
    return len(a_ng & b_ng) / len(a_ng | b_ng)
