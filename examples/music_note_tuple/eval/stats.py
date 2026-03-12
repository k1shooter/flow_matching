from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

try:
    from scipy.stats import t as student_t
except Exception:  # pragma: no cover
    student_t = None


def _t_critical(confidence: float, df: int) -> float:
    alpha = 1.0 - confidence
    if student_t is not None and df > 0:
        return float(student_t.ppf(1.0 - alpha / 2.0, df=df))
    return 1.96


def mean_and_ci(values: Iterable[float], confidence: float = 0.95) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std": 0.0, "n": 0}

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    sem = std / math.sqrt(arr.size) if arr.size > 1 else 0.0
    margin = _t_critical(confidence=confidence, df=max(1, arr.size - 1)) * sem

    return {
        "mean": mean,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
        "std": std,
        "n": int(arr.size),
    }


def within_subject_ci(
    records: List[Dict[str, object]],
    subject_key: str,
    condition_key: str,
    value_key: str,
    confidence: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Morey-corrected within-subject CI (ANOVA-style repeated-measures CI)."""
    if len(records) == 0:
        return {}

    condition_to_values: Dict[str, List[float]] = defaultdict(list)
    subject_values: Dict[str, List[float]] = defaultdict(list)

    for row in records:
        subject_values[str(row[subject_key])].append(float(row[value_key]))

    subject_mean = {k: float(np.mean(v)) for k, v in subject_values.items()}
    global_mean = float(np.mean([float(r[value_key]) for r in records]))

    n_conditions = len({str(r[condition_key]) for r in records})
    correction = math.sqrt(n_conditions / max(1, (n_conditions - 1)))

    normalized_rows = []
    for row in records:
        s = str(row[subject_key])
        normalized = float(row[value_key]) - subject_mean[s] + global_mean
        normalized_rows.append(
            {
                "condition": str(row[condition_key]),
                "raw": float(row[value_key]),
                "normalized": normalized,
            }
        )

    for row in normalized_rows:
        condition_to_values[row["condition"]].append(row["normalized"])

    result = {}
    for condition in sorted(condition_to_values):
        raw_values = [r["raw"] for r in normalized_rows if r["condition"] == condition]
        norm_values = np.asarray(condition_to_values[condition], dtype=np.float64)

        mean = float(np.mean(raw_values))
        if norm_values.size <= 1:
            margin = 0.0
            std = 0.0
        else:
            std = float(np.std(norm_values, ddof=1)) * correction
            sem = std / math.sqrt(norm_values.size)
            margin = _t_critical(confidence=confidence, df=norm_values.size - 1) * sem

        result[condition] = {
            "mean": mean,
            "ci_low": mean - margin,
            "ci_high": mean + margin,
            "std": std,
            "n": int(norm_values.size),
        }

    return result
