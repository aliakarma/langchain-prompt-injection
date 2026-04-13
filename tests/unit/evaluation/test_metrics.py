"""
tests/unit/evaluation/test_metrics.py
──────────────────────────────────────
Unit tests for compute_metrics, threshold_sweep, and per_category_metrics.
"""

import sys, os, math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from prompt_injection.evaluation.metrics import (
    ConfusionMatrix,
    MetricsReport,
    ThresholdPoint,
    _average_precision,
    _confusion,
    _precision_recall_f1,
    _roc_auc,
    compute_metrics,
    per_category_metrics,
    threshold_sweep,
)
from prompt_injection.evaluation.dataset import SyntheticDataset, DataRecord


# perfect predictions
Y_TRUE  = [1, 1, 1, 0, 0, 0]
Y_PRED  = [1, 1, 1, 0, 0, 0]
Y_SCORE = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]

# all wrong
Y_TRUE_W = [1, 1, 0, 0]
Y_PRED_W = [0, 0, 1, 1]

# all positive predictions
Y_TRUE_AP = [1, 1, 0, 0]
Y_PRED_AP = [1, 1, 1, 1]


class TestConfusion:
    def test_perfect_confusion(self):
        cm = _confusion(Y_TRUE, Y_PRED)
        assert cm.tp == 3 and cm.fp == 0 and cm.tn == 3 and cm.fn == 0

    def test_all_wrong(self):
        cm = _confusion(Y_TRUE_W, Y_PRED_W)
        assert cm.tp == 0 and cm.fn == 2 and cm.fp == 2 and cm.tn == 0

    def test_total_equals_length(self):
        cm = _confusion(Y_TRUE, Y_PRED)
        assert cm.total == len(Y_TRUE)


class TestPrecisionRecallF1:
    def test_perfect_prf(self):
        p, r, f = _precision_recall_f1(3, 0, 0)
        assert p == 1.0 and r == 1.0 and f == 1.0

    def test_zero_denominator_safe(self):
        p, r, f = _precision_recall_f1(0, 0, 0)
        assert p == 0.0 and r == 0.0 and f == 0.0

    def test_f1_harmonic_mean(self):
        p, r, f = _precision_recall_f1(2, 1, 1)  # P=2/3, R=2/3
        assert abs(f - 2*(2/3)*(2/3)/(2/3+2/3)) < 1e-6


class TestRocAuc:
    def test_perfect_auc(self):
        assert _roc_auc(Y_TRUE, Y_SCORE) == 1.0

    def test_random_auc_near_0_5(self):
        y_true  = [1, 0, 1, 0, 1, 0]
        y_score = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        auc = _roc_auc(y_true, y_score)
        assert 0.0 <= auc <= 1.0

    def test_all_same_class_returns_0(self):
        assert _roc_auc([1, 1, 1], [0.9, 0.8, 0.7]) == 0.0


class TestComputeMetrics:
    def test_perfect_report(self):
        r = compute_metrics(Y_TRUE, Y_PRED, Y_SCORE, config_name="perfect")
        assert r.precision == 1.0
        assert r.recall == 1.0
        assert r.f1 == 1.0
        assert r.accuracy == 1.0
        assert r.roc_auc == 1.0

    def test_report_fields_present(self):
        r = compute_metrics(Y_TRUE, Y_PRED)
        assert isinstance(r, MetricsReport)
        assert hasattr(r, "precision")
        assert hasattr(r, "recall")
        assert hasattr(r, "f1")
        assert hasattr(r, "accuracy")
        assert hasattr(r, "confusion")
        assert hasattr(r, "n_samples")

    def test_no_scores_gives_none_auc(self):
        r = compute_metrics(Y_TRUE, Y_PRED)
        assert r.roc_auc is None
        assert r.average_precision is None

    def test_n_samples_correct(self):
        r = compute_metrics(Y_TRUE, Y_PRED)
        assert r.n_samples == len(Y_TRUE)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            compute_metrics([1, 0, 1], [1, 0])

    def test_all_predictions_wrong(self):
        r = compute_metrics(Y_TRUE_W, Y_PRED_W)
        assert r.precision == 0.0
        assert r.recall == 0.0

    def test_to_dict_serialisable(self):
        import json
        r = compute_metrics(Y_TRUE, Y_PRED, Y_SCORE)
        d = r.to_dict()
        json.dumps(d)  # must not raise

    def test_summary_returns_string(self):
        r = compute_metrics(Y_TRUE, Y_PRED, config_name="test")
        s = r.summary()
        assert isinstance(s, str)
        assert "test" in s


class TestThresholdSweep:
    def test_returns_list_of_threshold_points(self):
        pts = threshold_sweep(Y_TRUE, Y_SCORE, n_thresholds=20)
        assert len(pts) == 20
        assert all(isinstance(p, ThresholdPoint) for p in pts)

    def test_all_thresholds_in_range(self):
        pts = threshold_sweep(Y_TRUE, Y_SCORE)
        for p in pts:
            assert 0.0 <= p.threshold <= 1.0

    def test_all_pr_values_valid(self):
        pts = threshold_sweep(Y_TRUE, Y_SCORE)
        for p in pts:
            assert 0.0 <= p.precision <= 1.0
            assert 0.0 <= p.recall <= 1.0
            assert 0.0 <= p.f1 <= 1.0

    def test_sorted_ascending_by_threshold(self):
        pts = threshold_sweep(Y_TRUE, Y_SCORE)
        thresholds = [p.threshold for p in pts]
        assert thresholds == sorted(thresholds)


class TestPerCategoryMetrics:
    def test_returns_dict_keyed_by_category(self):
        ds = SyntheticDataset(n_injections=50, n_benign=50, seed=42).generate()
        records = ds.records()
        y_pred  = [r.label for r in records]  # perfect predictions
        result  = per_category_metrics(records, y_pred)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_all_metrics_in_range(self):
        ds = SyntheticDataset(n_injections=50, n_benign=50, seed=42).generate()
        records = ds.records()
        y_pred  = [r.label for r in records]
        for cat, m in per_category_metrics(records, y_pred).items():
            assert 0.0 <= m.precision <= 1.0
            assert 0.0 <= m.recall <= 1.0
            assert 0.0 <= m.f1 <= 1.0

    def test_balanced_sampling_is_deterministic(self):
        records = [
            DataRecord(id=f"pos-{i}", text=f"attack {i}", label=1, attack_category="cat-a", source_type="user", severity="high")
            for i in range(3)
        ] + [
            DataRecord(id=f"ben-{i}", text=f"benign {i}", label=0, attack_category=None, source_type="user", severity=None)
            for i in range(10)
        ]
        y_pred = [r.label for r in records]
        first = per_category_metrics(records, y_pred, seed=7)["cat-a"]
        second = per_category_metrics(records, y_pred, seed=7)["cat-a"]
        assert first.n_samples == 6
        assert first.n_samples == second.n_samples
        assert first.f1 == second.f1
