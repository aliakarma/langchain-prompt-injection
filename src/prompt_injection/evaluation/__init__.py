"""
prompt_injection.evaluation
────────────────────────────
Evaluation subpackage: synthetic dataset, metrics, benchmark, and
performance profiling.

Public API
----------
    SyntheticDataset      – Labelled dataset loader and generator.
    DataRecord            – Single labelled record.
    compute_metrics       – Precision / recall / F1 / AUC computation.
    MetricsReport         – Structured metrics output.
    threshold_sweep       – PR / ROC curve data.
    per_category_metrics  – Per-attack-category breakdown.
    BenchmarkRunner       – Three-configuration ablation orchestrator.
    BenchmarkResult       – Full ablation output.
    ConfigResult          – Per-configuration result.
    PerformanceProfiler   – Latency and throughput profiling.
    LatencyReport         – Structured latency output.
    time_detector         – Convenience latency helper.
    ReportSerializer      – Console / JSON / CSV report output.
"""

from prompt_injection.evaluation.dataset import SyntheticDataset, DataRecord
from prompt_injection.evaluation.metrics import (
    compute_metrics,
    MetricsReport,
    ConfusionMatrix,
    ThresholdPoint,
    threshold_sweep,
    per_category_metrics,
)
from prompt_injection.evaluation.benchmark import (
    BenchmarkRunner,
    BenchmarkResult,
    ConfigResult,
)
from prompt_injection.evaluation.performance import (
    PerformanceProfiler,
    LatencyReport,
    time_detector,
)
from prompt_injection.evaluation.report import ReportSerializer

__all__ = [
    "SyntheticDataset", "DataRecord",
    "compute_metrics", "MetricsReport", "ConfusionMatrix",
    "ThresholdPoint", "threshold_sweep", "per_category_metrics",
    "BenchmarkRunner", "BenchmarkResult", "ConfigResult",
    "PerformanceProfiler", "LatencyReport", "time_detector",
    "ReportSerializer",
]
