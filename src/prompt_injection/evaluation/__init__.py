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
    CrossValidationSummary – Mean/std summary across folds.
    KeywordBaseline       – Simple external baseline.
    PerformanceProfiler   – Latency and throughput profiling.
    LatencyReport         – Structured latency output.
    time_detector         – Convenience latency helper.
    ReportSerializer      – Console / JSON / CSV report output.
"""

from prompt_injection.evaluation.dataset import SyntheticDataset, DataRecord
from prompt_injection.evaluation.real_dataset import (
    REQUIRED_EXTERNAL_RAW_FILES,
    DatasetSplit,
    ExternalRawDatasetBundle,
    ExternalRawFileSummary,
    ExternalRawRecord,
    ExternalRawSplitBundle,
    load_required_external_raw_datasets,
    split_external_raw_dataset,
)
from prompt_injection.evaluation.metrics import (
    compute_metrics,
    MetricsReport,
    ConfusionMatrix,
    MetricCI,
    ThresholdPoint,
    threshold_sweep,
    per_category_metrics,
    compute_false_positive_rate,
    bootstrap_confidence_intervals,
)
from prompt_injection.evaluation.benchmark import (
    BenchmarkRunner,
    BenchmarkResult,
    ConfigResult,
    CrossValidationSummary,
    KeywordBaseline,
)
from prompt_injection.evaluation.performance import (
    PerformanceProfiler,
    LatencyReport,
    time_detector,
)
from prompt_injection.evaluation.report import ReportSerializer

__all__ = [
    "SyntheticDataset", "DataRecord",
    "ExternalRawRecord", "ExternalRawFileSummary", "ExternalRawDatasetBundle",
    "DatasetSplit", "ExternalRawSplitBundle",
    "REQUIRED_EXTERNAL_RAW_FILES",
    "load_required_external_raw_datasets", "split_external_raw_dataset",
    "compute_metrics", "MetricsReport", "ConfusionMatrix",
    "MetricCI", "compute_false_positive_rate", "bootstrap_confidence_intervals",
    "ThresholdPoint", "threshold_sweep", "per_category_metrics",
    "BenchmarkRunner", "BenchmarkResult", "ConfigResult",
    "CrossValidationSummary", "KeywordBaseline",
    "PerformanceProfiler", "LatencyReport", "time_detector",
    "ReportSerializer",
]
