"""Dataset loaders for popular multimodal benchmarks.

Provides a uniform interface for loading evaluation data from common
benchmarks so users can quickly test their systems.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from mmeval_vrag.types import EvalSample, ImageInput, RetrievedItem

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_LOADER_REGISTRY: Dict[str, type] = {}


def _register(cls):
    _LOADER_REGISTRY[cls.name] = cls
    return cls


class DatasetLoader:
    """Base class for dataset loaders."""

    name: str = ""

    def load(
        self,
        path: Union[str, Path],
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> List[EvalSample]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# JSONL loader (generic)
# ---------------------------------------------------------------------------


@_register
class JSONLLoader(DatasetLoader):
    """Load any dataset from a simple JSONL format.

    Expected fields per line: ``query``, ``retrieved`` (list),
    ``generated_answer``, ``reference_answer`` (optional).
    """

    name = "jsonl"

    def load(self, path, split="test", max_samples=None):
        samples = []
        with open(path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                obj = json.loads(line)
                retrieved = [
                    RetrievedItem(
                        text=r.get("text"),
                        image=ImageInput(path=r["image"]) if "image" in r else None,
                        score=r.get("score"),
                        is_relevant=r.get("is_relevant"),
                    )
                    for r in obj.get("retrieved", [])
                ]
                samples.append(
                    EvalSample(
                        query_text=obj.get("query"),
                        query_image=(
                            ImageInput(path=obj["query_image"]) if "query_image" in obj else None
                        ),
                        retrieved=retrieved,
                        generated_answer=obj.get("generated_answer", ""),
                        reference_answer=obj.get("reference_answer"),
                        sample_id=obj.get("id", str(i)),
                    )
                )
        return samples


# ---------------------------------------------------------------------------
# VQA-style loader (VQAv2 / Medical VQA format)
# ---------------------------------------------------------------------------


@_register
class VQALoader(DatasetLoader):
    """Load VQA-style datasets (e.g. VQAv2, PathVQA, SLAKE).

    Expects a JSON file with ``questions`` and an ``annotations`` file,
    following the VQAv2 schema.
    """

    name = "vqa"

    def load(self, path, split="test", max_samples=None):
        base = Path(path)
        q_file = base / f"{split}_questions.json"
        a_file = base / f"{split}_annotations.json"

        with open(q_file) as f:
            questions = json.load(f)["questions"]
        annotations = {}
        if a_file.exists():
            with open(a_file) as f:
                for ann in json.load(f)["annotations"]:
                    annotations[ann["question_id"]] = ann

        samples = []
        for i, q in enumerate(questions):
            if max_samples and i >= max_samples:
                break
            qid = q["question_id"]
            ann = annotations.get(qid, {})
            img_path = base / "images" / q.get("image_filename", f"{q.get('image_id', qid)}.jpg")
            samples.append(
                EvalSample(
                    query_text=q["question"],
                    query_image=ImageInput(path=str(img_path)) if img_path.exists() else None,
                    reference_answer=ann.get("multiple_choice_answer", ""),
                    sample_id=str(qid),
                )
            )
        return samples


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(
    name: str,
    path: Union[str, Path],
    split: str = "test",
    max_samples: Optional[int] = None,
) -> List[EvalSample]:
    """Load an evaluation dataset.

    Parameters
    ----------
    name : str
        Loader name (``"jsonl"``, ``"vqa"``).
    path : str or Path
        Path to dataset root or file.
    split : str
        Dataset split.
    max_samples : int or None
        Cap on number of samples to load.
    """
    if name not in _LOADER_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(_LOADER_REGISTRY)}")
    return _LOADER_REGISTRY[name]().load(path, split, max_samples)
