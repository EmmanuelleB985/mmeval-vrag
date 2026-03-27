"""Structured containers for evaluation results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EvalResult:
    """Result for a single sample."""

    sample_id: str
    scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, metric: str) -> float:
        return self.scores[metric]

    def to_dict(self) -> Dict[str, Any]:
        return {"sample_id": self.sample_id, "scores": self.scores, "details": self.details}


@dataclass
class EvalResultCollection:
    """Aggregated results across samples."""

    results: List[EvalResult] = field(default_factory=list)

    # ---- aggregate helpers --------------------------------------------------

    @property
    def metric_names(self) -> List[str]:
        names: set[str] = set()
        for r in self.results:
            names.update(r.scores.keys())
        return sorted(names)

    def mean(self, metric: str) -> float:
        vals = [r.scores[metric] for r in self.results if metric in r.scores]
        return float(np.mean(vals)) if vals else 0.0

    def std(self, metric: str) -> float:
        vals = [r.scores[metric] for r in self.results if metric in r.scores]
        return float(np.std(vals)) if vals else 0.0

    def median(self, metric: str) -> float:
        vals = [r.scores[metric] for r in self.results if metric in r.scores]
        return float(np.median(vals)) if vals else 0.0

    def percentile(self, metric: str, q: float) -> float:
        vals = [r.scores[metric] for r in self.results if metric in r.scores]
        return float(np.percentile(vals, q)) if vals else 0.0

    # ---- display / export ---------------------------------------------------

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return ``{metric: {mean, std, median, min, max}}``."""
        out: Dict[str, Dict[str, float]] = {}
        for m in self.metric_names:
            vals = [r.scores[m] for r in self.results if m in r.scores]
            if not vals:
                continue
            arr = np.array(vals)
            out[m] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "median": float(np.median(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n": len(vals),
            }
        return out

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        payload = {
            "summary": self.summary(),
            "per_sample": [r.to_dict() for r in self.results],
        }
        text = json.dumps(payload, indent=indent)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text

    def to_dataframe(self):
        """Return a ``pandas.DataFrame`` (raises if pandas is missing)."""
        import pandas as pd

        rows = []
        for r in self.results:
            row = {"sample_id": r.sample_id, **r.scores}
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        lines = [f"EvalResultCollection(n={len(self.results)})"]
        for m, stats in self.summary().items():
            lines.append(f"  {m}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        return "\n".join(lines)
