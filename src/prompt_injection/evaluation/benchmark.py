"""
evaluation/benchmark.py
───────────────────────
Three-configuration ablation benchmark.

Configuration A – Regex only       (InjectionDetector mode="rules")
Configuration B – Regex + Scoring  (InjectionDetector mode="hybrid")
Configuration C – Regex + Scoring + Classifier  (mode="full", fitted LR)

The ``BenchmarkRunner`` evaluates all three configurations on the same
dataset partition, collects per-config ``MetricsReport`` and
``LatencyReport``, and assembles a ``BenchmarkResult`` for reporting.

Usage
-----
    from prompt_injection.evaluation.benchmark import BenchmarkRunner
    from prompt_injection.evaluation.dataset import SyntheticDataset

    ds = SyntheticDataset().generate()
    train_ds, test_ds = ds.train_test_split(test_size=0.2)

    runner = BenchmarkRunner()
    result = runner.run(train_ds, test_ds)
    print(result.summary_table())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sklearn.model_selection import StratifiedKFold

from prompt_injection.detector import DetectionResult, HitRecord, InjectionDetector, LogisticRegressionScorer
from prompt_injection.evaluation.dataset import SyntheticDataset, DataRecord
from prompt_injection.evaluation.metrics import (
    MetricsReport,
    compute_metrics,
    per_category_metrics,
    threshold_sweep,
    ThresholdPoint,
)
from prompt_injection.evaluation.performance import LatencyReport, PerformanceProfiler

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Baseline / summary containers
# ---------------------------------------------------------------------------


@dataclass
class CrossValidationSummary:
    """Aggregate cross-validation statistics for one configuration."""

    config_name: str
    folds: int
    f1_mean: float
    f1_std: float
    auc_mean: float | None
    auc_std: float | None


class KeywordBaseline:
    """Simple binary keyword baseline for comparison with the detector configs."""

    keywords = ["ignore", "system prompt", "reveal", "instructions", "bypass"]

    def __init__(self, keywords: list[str] | None = None) -> None:
        if keywords is not None:
            self.keywords = keywords
        self.mode = "baseline"

    def scan(self, text: str, source_type: str = "user") -> DetectionResult:
        lowered = text.lower()
        hits: list[HitRecord] = []
        for keyword in self.keywords:
            if keyword in lowered:
                hits.append(
                    HitRecord(
                        pattern_id=f"KW-{keyword.replace(' ', '_').upper()}",
                        category="keyword_baseline",
                        severity="medium",
                        description=f"Keyword '{keyword}' matched",
                    )
                )

        risk_score = min(len(hits) / max(len(self.keywords), 1), 1.0)
        return DetectionResult(
            text_preview=text[:200],
            source_type=source_type,
            is_injection=bool(hits),
            risk_score=risk_score,
            hits=hits,
            hit_categories=["keyword_baseline"] if hits else [],
            classifier_score=None,
            latency_ms=0.0,
        )


# ---------------------------------------------------------------------------
# Per-config result
# ---------------------------------------------------------------------------


@dataclass
class ConfigResult:
    """Result for a single detector configuration."""

    config_name: str        # "A: Regex only" | "B: Regex + Scoring" | "C: + Classifier"
    mode: str               # "rules" | "hybrid" | "full"
    metrics: MetricsReport
    latency: LatencyReport
    synthetic_metrics: MetricsReport | None = None
    per_category: dict[str, MetricsReport] = field(default_factory=dict)
    synthetic_per_category: dict[str, MetricsReport] = field(default_factory=dict)
    sweep: list[ThresholdPoint] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Aggregate result
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """
    Full three-configuration ablation result.

    Attributes
    ----------
    config_a : ConfigResult
    config_b : ConfigResult
    config_c : ConfigResult
    n_train : int
    n_synthetic_test : int
    n_real_world_test : int
    synthetic_metrics : dict[str, MetricsReport]
    real_world_metrics : dict[str, MetricsReport]
    synthetic_baseline_metrics : dict[str, MetricsReport]
    real_world_baseline_metrics : dict[str, MetricsReport]
    benign_fpr : dict[str, float]
    white_box_metrics : dict[str, MetricsReport]
    cross_validation : dict[str, CrossValidationSummary]
    """

    config_a: ConfigResult
    config_b: ConfigResult
    config_c: ConfigResult
    n_train: int
    n_synthetic_test: int
    n_real_world_test: int
    synthetic_metrics: dict[str, MetricsReport] = field(default_factory=dict)
    real_world_metrics: dict[str, MetricsReport] = field(default_factory=dict)
    synthetic_baseline_metrics: dict[str, MetricsReport] = field(default_factory=dict)
    real_world_baseline_metrics: dict[str, MetricsReport] = field(default_factory=dict)
    benign_fpr: dict[str, float] = field(default_factory=dict)
    white_box_metrics: dict[str, MetricsReport] = field(default_factory=dict)
    cross_validation: dict[str, CrossValidationSummary] = field(default_factory=dict)

    def configs(self) -> list[ConfigResult]:
        return [self.config_a, self.config_b, self.config_c]

    def summary_table(self) -> str:
        """
        Render a plain-text comparison table.
        """
        header = (
            f"{'Configuration':<28} {'P':>6} {'R':>6} {'F1':>6} "
            f"{'Acc':>6} {'AUC':>6} {'Latency':>10}"
        )
        sep = "-" * len(header)
        rows = [header, sep]
        for cfg in self.configs():
            m = cfg.metrics
            l = cfg.latency
            auc = f"{m.roc_auc:.4f}" if m.roc_auc is not None else "  N/A"
            rows.append(
                f"{cfg.config_name:<28} "
                f"{m.precision:>6.4f} {m.recall:>6.4f} {m.f1:>6.4f} "
                f"{m.accuracy:>6.4f} {auc:>6} "
                f"{l.mean_ms:>8.2f} ms"
            )
        rows.append(sep)
        rows.append(f"  Train samples: {self.n_train}   Test samples: {self.n_test}")
        return "\n".join(rows)

    def best_f1(self) -> ConfigResult:
        return max(self.configs(), key=lambda c: c.metrics.f1)

    def fastest(self) -> ConfigResult:
        return min(self.configs(), key=lambda c: c.latency.mean_ms)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """
    Orchestrates the three-configuration ablation.

    Parameters
    ----------
    threshold : float
        Shared decision threshold for configurations B and C.
    n_latency_runs : int
        Timing repetitions per text (passed to PerformanceProfiler).
    latency_budget_ms : float | None
        Optional latency budget for budget-check reporting.
    sweep_thresholds : bool
        Whether to compute full threshold sweep for PR/ROC curves.
    """

    CONFIG_NAMES = {
        "rules":  "A: Regex only",
        "hybrid": "B: Regex + Scoring",
        "full":   "C: + Classifier",
    }

    def __init__(
        self,
        threshold: float = 0.50,
        n_latency_runs: int = 30,
        latency_budget_ms: float | None = 10.0,
        sweep_thresholds: bool = True,
    ) -> None:
        self.threshold = threshold
        self.n_latency_runs = n_latency_runs
        self.latency_budget_ms = latency_budget_ms
        self.sweep_thresholds = sweep_thresholds

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        train_dataset: SyntheticDataset,
        test_dataset: SyntheticDataset,
        real_world_dataset: SyntheticDataset | None = None,
    ) -> BenchmarkResult:
        """
        Run the full three-configuration benchmark.

        Parameters
        ----------
        train_dataset : SyntheticDataset
            Training split (used to fit the Config C classifier).
        test_dataset : SyntheticDataset
            Test split (used for all metric computation).
        real_world_dataset : SyntheticDataset | None
            Optional real-world labelled sample for a separate evaluation.
        """
        test_records = test_dataset.records()
        test_texts   = test_dataset.texts()
        test_labels  = test_dataset.labels()

        # ── Configuration A ──────────────────────────────────────────────
        det_a = InjectionDetector(mode="rules", threshold=self.threshold)
        cfg_a = self._evaluate(det_a, test_records, test_texts, test_labels)

        # ── Configuration B ──────────────────────────────────────────────
        det_b = InjectionDetector(mode="hybrid", threshold=self.threshold)
        cfg_b = self._evaluate(det_b, test_records, test_texts, test_labels)

        # ── Configuration C ──────────────────────────────────────────────
        clf = LogisticRegressionScorer()
        clf.fit(train_dataset.texts(), train_dataset.labels())
        det_c = InjectionDetector(mode="full", threshold=self.threshold, classifier=clf)
        cfg_c = self._evaluate(det_c, test_records, test_texts, test_labels)

        # ── Real-world evaluation ─────────────────────────────────────────
        rw_metrics: dict[str, MetricsReport] = {}
        if real_world_dataset is not None and len(real_world_dataset) > 0:
            rw_records = real_world_dataset.records()
            rw_texts   = real_world_dataset.texts()
            rw_labels  = real_world_dataset.labels()
            for mode, det, name in [
                ("rules",  det_a, "A: Regex only"),
                ("hybrid", det_b, "B: Regex + Scoring"),
                ("full",   det_c, "C: + Classifier"),
            ]:
                y_pred, y_scores = self._predict(det, rw_texts)
                rw_metrics[name] = compute_metrics(
                    rw_labels, y_pred, y_scores,
                    threshold=self.threshold,
                    config_name=f"[Real] {name}",
                )

        return BenchmarkResult(
            config_a=cfg_a,
            config_b=cfg_b,
            config_c=cfg_c,
            n_train=len(train_dataset),
            n_test=len(test_dataset),
            real_world_metrics=rw_metrics,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict(
        self,
        detector: InjectionDetector,
        texts: list[str],
    ) -> tuple[list[int], list[float]]:
        """Run detector on all texts, return (y_pred, y_scores)."""
        y_pred: list[int] = []
        y_scores: list[float] = []
        for text in texts:
            result = detector.scan(text)
            y_pred.append(1 if result.is_injection else 0)
            y_scores.append(result.risk_score)
        return y_pred, y_scores

    def _evaluate(
        self,
        detector: InjectionDetector,
        records: list[DataRecord],
        texts: list[str],
        labels: list[int],
    ) -> ConfigResult:
        """Full evaluation for one detector configuration."""
        name = self.CONFIG_NAMES[detector.mode]

        # Predictions
        y_pred, y_scores = self._predict(detector, texts)

        # Core metrics
        scores_arg = y_scores if detector.mode != "rules" else None
        metrics = compute_metrics(
            labels, y_pred, scores_arg,
            threshold=self.threshold,
            config_name=name,
        )

        # Per-category metrics
        cat_metrics = per_category_metrics(records, y_pred, scores_arg)

        # Threshold sweep (B and C only)
        sweep: list[ThresholdPoint] = []
        if self.sweep_thresholds and detector.mode != "rules":
            sweep = threshold_sweep(labels, y_scores)

        # Latency profiling
        profiler = PerformanceProfiler()
        latency = profiler.profile(
            detector,
            texts[:min(50, len(texts))],   # sample for speed
            n_runs=self.n_latency_runs,
            budget_ms=self.latency_budget_ms,
            config_name=name,
        )

        return ConfigResult(
            config_name=name,
            mode=detector.mode,
            metrics=metrics,
            latency=latency,
            per_category=cat_metrics,
            sweep=sweep,
        )
