"""
evaluation/metrics.py
─────────────────────
Precision, recall, F1, accuracy, ROC-AUC, average precision, confusion
matrix, and threshold sweep utilities.

All functions accept plain Python lists so they can be used with or
without scikit-learn being the primary caller.

Usage
-----
    from prompt_injection.evaluation.metrics import compute_metrics, threshold_sweep

    report = compute_metrics(y_true, y_pred, y_scores=scores)
    print(report.summary())

    sweep = threshold_sweep(y_true, y_scores)
    # → list of ThresholdPoint for PR-curve plotting
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import hashlib
import random
from typing import Optional


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ConfusionMatrix:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def to_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn}


@dataclass
class MetricsReport:
    """
    Full evaluation report for one detector / threshold configuration.

    Attributes
    ----------
    precision : float
    recall : float
    f1 : float
    accuracy : float
    roc_auc : float | None
        None if continuous scores are not available (binary mode).
    average_precision : float | None
        Area under the precision-recall curve.
    confusion : ConfusionMatrix
    threshold : float
        Decision threshold used.
    n_samples : int
    config_name : str
        Human-readable label for this configuration.
    """

    precision: float
    recall: float
    f1: float
    accuracy: float
    roc_auc: Optional[float]
    average_precision: Optional[float]
    confusion: ConfusionMatrix
    threshold: float
    n_samples: int
    config_name: str = "unnamed"

    def summary(self) -> str:
        lines = [
            f"Config      : {self.config_name}",
            f"Samples     : {self.n_samples}",
            f"Threshold   : {self.threshold:.3f}",
            f"Precision   : {self.precision:.4f}",
            f"Recall      : {self.recall:.4f}",
            f"F1          : {self.f1:.4f}",
            f"Accuracy    : {self.accuracy:.4f}",
        ]
        if self.roc_auc is not None:
            lines.append(f"ROC-AUC     : {self.roc_auc:.4f}")
        if self.average_precision is not None:
            lines.append(f"Avg Prec    : {self.average_precision:.4f}")
        cm = self.confusion
        lines += [
            f"TP={cm.tp}  FP={cm.fp}  TN={cm.tn}  FN={cm.fn}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "config_name": self.config_name,
            "n_samples": self.n_samples,
            "threshold": round(self.threshold, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "average_precision": (
                round(self.average_precision, 4)
                if self.average_precision is not None
                else None
            ),
            "confusion": self.confusion.to_dict(),
        }


@dataclass
class ThresholdPoint:
    """One point on a precision-recall or ROC curve."""

    threshold: float
    precision: float
    recall: float
    f1: float
    fpr: float          # false positive rate (for ROC)
    tp: int
    fp: int
    tn: int
    fn: int


# ---------------------------------------------------------------------------
# Core metric helpers (pure Python — no sklearn required for basic metrics)
# ---------------------------------------------------------------------------


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _precision_recall_f1(
    tp: int, fp: int, fn: int
) -> tuple[float, float, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def _confusion(y_true: list[int], y_pred: list[int]) -> ConfusionMatrix:
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        else:
            fn += 1
    return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)


def _roc_auc(y_true: list[int], y_scores: list[float]) -> float:
    """Trapezoid-rule AUC from scratch (no sklearn required)."""
    pairs = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    tp = fp = 0
    prev_fpr = prev_tpr = 0.0
    auc = 0.0
    prev_score = None

    for score, label in pairs:
        if prev_score is not None and score != prev_score:
            fpr = _safe_div(fp, n_neg)
            tpr = _safe_div(tp, n_pos)
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
            prev_fpr, prev_tpr = fpr, tpr
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score

    fpr = _safe_div(fp, n_neg)
    tpr = _safe_div(tp, n_pos)
    auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
    return round(min(auc, 1.0), 4)


def _average_precision(y_true: list[int], y_scores: list[float]) -> float:
    """Area under the precision-recall curve (interpolated)."""
    pairs = sorted(zip(y_scores, y_true), key=lambda x: -x[0])
    tp = 0
    fp = 0
    n_pos = sum(y_true)
    if n_pos == 0:
        return 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    prev_recall = 0.0
    ap = 0.0

    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, n_pos)
        precisions.append(p)
        recalls.append(r)

    # Interpolate (step function, right-to-left).
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    prev_r = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_r)
        prev_r = r

    return round(min(ap, 1.0), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_scores: list[float] | None = None,
    threshold: float = 0.50,
    config_name: str = "unnamed",
) -> MetricsReport:
    """
    Compute a full ``MetricsReport``.

    Parameters
    ----------
    y_true : list[int]
        Ground-truth labels (1 = injection, 0 = benign).
    y_pred : list[int]
        Predicted binary labels.
    y_scores : list[float] | None
        Continuous risk scores for AUC computation.  If None, AUC fields
        will be None in the report.
    threshold : float
        Decision threshold (informational; y_pred is already binarised).
    config_name : str
        Label for this configuration.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    cm = _confusion(y_true, y_pred)
    precision, recall, f1 = _precision_recall_f1(cm.tp, cm.fp, cm.fn)
    accuracy = _safe_div(cm.tp + cm.tn, cm.total)

    roc_auc = None
    avg_prec = None
    if y_scores is not None:
        if len(y_scores) != len(y_true):
            raise ValueError("y_scores must have the same length as y_true.")
        roc_auc = _roc_auc(y_true, y_scores)
        avg_prec = _average_precision(y_true, y_scores)

    return MetricsReport(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        accuracy=round(accuracy, 4),
        roc_auc=roc_auc,
        average_precision=avg_prec,
        confusion=cm,
        threshold=threshold,
        n_samples=len(y_true),
        config_name=config_name,
    )


def threshold_sweep(
    y_true: list[int],
    y_scores: list[float],
    n_thresholds: int = 100,
) -> list[ThresholdPoint]:
    """
    Sweep thresholds and return PR / ROC curve data.

    Parameters
    ----------
    y_true : list[int]
        Ground-truth labels.
    y_scores : list[float]
        Continuous risk scores.
    n_thresholds : int
        Number of evenly-spaced threshold values to evaluate.

    Returns
    -------
    list[ThresholdPoint]
        One point per threshold, sorted ascending by threshold.
    """
    min_score = min(y_scores)
    max_score = max(y_scores)
    step = (max_score - min_score) / max(n_thresholds - 1, 1)
    thresholds = [min_score + i * step for i in range(n_thresholds)]

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    points: list[ThresholdPoint] = []

    for thr in thresholds:
        y_pred = [1 if s >= thr else 0 for s in y_scores]
        cm = _confusion(y_true, y_pred)
        precision, recall, f1 = _precision_recall_f1(cm.tp, cm.fp, cm.fn)
        fpr = _safe_div(cm.fp, n_neg) if n_neg > 0 else 0.0
        points.append(ThresholdPoint(
            threshold=round(thr, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            fpr=round(fpr, 4),
            tp=cm.tp, fp=cm.fp, tn=cm.tn, fn=cm.fn,
        ))

    return points


def _category_seed(seed: int, category: str) -> int:
    digest = hashlib.sha256(f"{seed}:{category}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _balanced_category_indices(
    records: list,
    category: str,
    *,
    seed: int,
) -> list[int]:
    positive_indices = [i for i, record in enumerate(records) if record.attack_category == category]
    negative_indices = [i for i, record in enumerate(records) if record.label == 0]

    if not positive_indices or not negative_indices:
        return positive_indices

    sample_size = min(len(positive_indices), len(negative_indices))
    rng = random.Random(_category_seed(seed, category))
    sampled_positives = positive_indices if len(positive_indices) <= sample_size else rng.sample(positive_indices, sample_size)
    sampled_negatives = rng.sample(negative_indices, sample_size)
    return sorted(sampled_positives + sampled_negatives)


def per_category_metrics(
    records: list,          # list[DataRecord]
    y_pred: list[int],
    y_scores: list[float] | None = None,
    *,
    seed: int = 42,
) -> dict[str, MetricsReport]:
    """
    Compute metrics broken down by attack category.

    Parameters
    ----------
    records : list[DataRecord]
        Dataset records aligned with y_pred and y_scores.
    y_pred : list[int]
        Predicted binary labels.
    y_scores : list[float] | None
        Continuous scores (optional).

    Returns
    -------
    dict[str, MetricsReport]
        Map from category name to MetricsReport.
    """
    categories = sorted({r.attack_category for r in records if r.attack_category})
    results: dict[str, MetricsReport] = {}

    for cat in categories:
        indices = _balanced_category_indices(records, cat, seed=seed)
        if not indices:
            continue
        yt = [records[i].label for i in indices]
        yp = [y_pred[i] for i in indices]
        ys = [y_scores[i] for i in indices] if y_scores else None
        results[cat] = compute_metrics(yt, yp, ys, config_name=cat)

    return results
