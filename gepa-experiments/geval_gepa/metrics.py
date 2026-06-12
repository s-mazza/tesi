"""Metrics for comparing judge scores with USR human annotations."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class RegressionMetrics:
    n: int
    pearson: float
    spearman: float
    kendall_tau: float
    mae: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "pearson": self.pearson,
            "spearman": self.spearman,
            "kendall_tau": self.kendall_tau,
            "mae": self.mae,
        }


def parse_discrete_score(value: Any, *, min_score: int, max_score: int) -> int | None:
    """Extract a single integer score from a DSPy prediction or raw text."""

    if hasattr(value, "score"):
        value = getattr(value, "score")
    text = str(value).strip()
    if text.isdigit():
        score = int(text)
        return score if min_score <= score <= max_score else None

    matches = [int(match) for match in re.findall(r"(?<!\d)(\d+)(?!\d)", text)]
    valid = [score for score in matches if min_score <= score <= max_score]
    if len(valid) == 1:
        return valid[0]
    return None


def normalized_absolute_score(predicted: float, target: float, *, min_score: int, max_score: int) -> float:
    """Return 1 for exact agreement and 0 for the largest possible scale error."""

    scale_width = max_score - min_score
    if scale_width <= 0:
        raise ValueError("max_score must be greater than min_score")
    return max(0.0, 1.0 - abs(predicted - target) / scale_width)


def compute_regression_metrics(predictions: Iterable[float], targets: Iterable[float]) -> RegressionMetrics:
    pred = list(predictions)
    gold = list(targets)
    if len(pred) != len(gold):
        raise ValueError(f"Prediction/target length mismatch: {len(pred)} != {len(gold)}")
    if not pred:
        raise ValueError("At least one prediction is required")

    errors = [abs(a - b) for a, b in zip(pred, gold)]
    return RegressionMetrics(
        n=len(pred),
        pearson=_pearson(pred, gold),
        spearman=_pearson(_rank(pred), _rank(gold)),
        kendall_tau=_kendall_tau_b(pred, gold),
        mae=sum(errors) / len(errors),
    )


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Inputs must have equal length")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom == 0:
        return 0.0
    return sum(x * y for x, y in zip(dx, dy)) / denom


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        average_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = average_rank
        i = j
    return ranks


def _kendall_tau_b(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Inputs must have equal length")

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = _sign(xs[i] - xs[j])
            dy = _sign(ys[i] - ys[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1

    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
