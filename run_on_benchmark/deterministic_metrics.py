"""Deterministic, local-only metrics for insight predictions."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

TOKEN_RE = re.compile(r"\w+(?:[.-]\w+)*", re.UNICODE)
NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def harmonic(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def token_f1(left: str, right: str) -> float:
    a, b = Counter(tokenize(left)), Counter(tokenize(right))
    if not a or not b:
        return 0.0
    overlap = sum((a & b).values())
    return harmonic(overlap / sum(a.values()), overlap / sum(b.values()))


def two_way_token_overlap(predictions: list[str], references: list[str]) -> dict:
    if not predictions or not references:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    matrix = [[token_f1(pred, ref) for pred in predictions] for ref in references]
    recall = sum(max(row) for row in matrix) / len(matrix)
    precision = sum(max(matrix[i][j] for i in range(len(references))) for j in range(len(predictions))) / len(predictions)
    return {"recall": recall, "precision": precision, "f1": harmonic(precision, recall)}


def lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if token == other else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> dict:
    pred, ref = tokenize(prediction), tokenize(reference)
    if not pred or not ref:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    common = lcs_length(pred, ref)
    precision, recall = common / len(pred), common / len(ref)
    return {"recall": recall, "precision": precision, "f1": harmonic(precision, recall)}


def rouge_1(prediction: str, reference: str) -> dict:
    pred, ref = Counter(tokenize(prediction)), Counter(tokenize(reference))
    if not pred or not ref:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    common = sum((pred & ref).values())
    precision, recall = common / sum(pred.values()), common / sum(ref.values())
    return {"recall": recall, "precision": precision, "f1": harmonic(precision, recall)}


def _numbers(text: str) -> list[tuple[float, bool]]:
    values = []
    for raw in NUMBER_RE.findall(text):
        percent = raw.endswith("%")
        try:
            values.append((float(raw.rstrip("%").replace(",", "")), percent))
        except ValueError:
            continue
    return values


def numeric_coverage(predictions: Iterable[str], references: Iterable[str], tolerance: float = 0.01) -> dict:
    predicted = _numbers(" ".join(predictions))
    expected = _numbers(" ".join(references))
    matched = 0
    for value, percent in expected:
        threshold = max(1e-9, abs(value) * tolerance)
        if any(pct == percent and math.isclose(candidate, value, abs_tol=threshold) for candidate, pct in predicted):
            matched += 1
    return {
        "coverage": matched / len(expected) if expected else None,
        "matched": matched,
        "n_reference_numbers": len(expected),
        "n_prediction_numbers": len(predicted),
        "relative_tolerance": tolerance,
    }


def output_profile(predictions: list[str]) -> dict:
    normalized = [" ".join(tokenize(item)) for item in predictions if item.strip()]
    unique = len(set(normalized))
    return {
        "n_insights": len(predictions),
        "n_empty": sum(not item.strip() for item in predictions),
        "duplicate_rate": 1 - unique / len(normalized) if normalized else 0.0,
        "mean_tokens": sum(len(tokenize(item)) for item in predictions) / len(predictions) if predictions else 0.0,
    }


def evaluate_deterministic(prediction: dict, ground_truth: dict) -> dict:
    insights = prediction.get("pred_insights") or []
    references = ground_truth.get("insights") or []
    summary = prediction.get("pred_summary") or ""
    gt_summary = ground_truth.get("summary") or ""
    return {
        "token_overlap": two_way_token_overlap(insights, references),
        "summary_rouge_1": rouge_1(summary, gt_summary),
        "summary_rouge_l": rouge_l(summary, gt_summary),
        "numeric_coverage": numeric_coverage(insights, references),
        "output_profile": output_profile(insights),
    }
