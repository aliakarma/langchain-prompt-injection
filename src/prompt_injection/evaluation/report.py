"""
evaluation/report.py
─────────────────────
Report serialisation: console table, JSON, and CSV.

Works with ``BenchmarkResult`` (three-config ablation) and individual
``MetricsReport`` / ``LatencyReport`` objects.

Usage
-----
    from prompt_injection.evaluation.report import ReportSerializer

    s = ReportSerializer(result)
    s.print_summary()
    s.to_json("reports/benchmark.json")
    s.to_csv("reports/benchmark.csv")
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_injection.evaluation.benchmark import BenchmarkResult


class ReportSerializer:
    """
    Serialise a ``BenchmarkResult`` to multiple output formats.

    Parameters
    ----------
    result : BenchmarkResult
        The benchmark result to serialise.
    """

    def __init__(self, result: "BenchmarkResult") -> None:
        self.result = result

    # ------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print the full benchmark summary to stdout."""
        print("\n" + "=" * 70)
        print("  PROMPT-INJECTION DETECTION — ABLATION BENCHMARK")
        print("=" * 70)
        print(self.result.summary_table())

        if self.result.synthetic_metrics:
            print("\n── In-Distribution Upper Bound ────────────────────────────────────")
            print(self.result.dataset_table("Synthetic test", self.result.synthetic_metrics))

        if self.result.real_world_metrics:
            print("\n── Out-of-Distribution Primary Result ─────────────────────────────")
            print(self.result.dataset_table("Real-world test", self.result.real_world_metrics))
            print("  Note: AUC may be unreliable for small datasets")

        if self.result.external_metrics:
            print("\n── External Dataset Result ───────────────────────────────────────")
            print(self.result.dataset_table("External test", self.result.external_metrics))

        if self.result.synthetic_baseline_metrics:
            print("\n── Synthetic Baseline ─────────────────────────────────────────────")
            print(self.result.baseline_table("Keyword baseline", self.result.synthetic_baseline_metrics))

        if self.result.real_world_baseline_metrics:
            print("\n── Real-World Baseline ────────────────────────────────────────────")
            print(self.result.baseline_table("Keyword baseline", self.result.real_world_baseline_metrics))

        print("\n── Per-Category Breakdown ──────────────────────────────────────────")
        for cfg in self.result.configs():
            print(f"\n  [{cfg.config_name}]")
            if not cfg.per_category:
                print("    (no per-category data)")
                continue
            header = f"    {'Category':<30} {'P':>6} {'R':>6} {'F1':>6}"
            print(header)
            print("    " + "-" * (len(header) - 4))
            for cat, m in sorted(cfg.per_category.items()):
                print(f"    {cat:<30} {m.precision:>6.4f} {m.recall:>6.4f} {m.f1:>6.4f}")

        print("\n── Latency Profiles ────────────────────────────────────────────────")
        for cfg in self.result.configs():
            print(f"\n  [{cfg.config_name}]")
            for line in cfg.latency.summary().split("\n"):
                print(f"    {line}")

        if self.result.cross_validation:
            print("\n── 5-Fold Cross-Validation (Synthetic) ───────────────────────────")
            for name, summary in self.result.cross_validation.items():
                print(
                    f"  [{name}] F1={summary.f1_mean:.4f} ± {summary.f1_std:.4f}"
                    + (
                        f"   AUC={summary.auc_mean:.4f} ± {summary.auc_std:.4f}"
                        if summary.auc_mean is not None and summary.auc_std is not None
                        else ""
                    )
                )

        if self.result.confidence_intervals:
            print("\n── Bootstrap Confidence Intervals (Primary) ─────────────────────")
            for name, ci_map in self.result.confidence_intervals.items():
                f1_ci = ci_map.get("f1")
                auc_ci = ci_map.get("roc_auc")
                if f1_ci is None or auc_ci is None:
                    continue
                print(
                    f"  [{name}] F1={f1_ci.mean:.4f} [{f1_ci.lower:.4f}, {f1_ci.upper:.4f}]"
                    f"   AUC={auc_ci.mean:.4f} [{auc_ci.lower:.4f}, {auc_ci.upper:.4f}]"
                )

        if self.result.domain_shift:
            print("\n── Cross-Dataset Generalization Gap ─────────────────────────────")
            for name, gap in self.result.domain_shift.items():
                print(
                    f"  [{name}] ΔF1(syn→primary)={gap.get('f1_drop_vs_synthetic', 0.0):.4f}"
                    f"   ΔAUC(syn→primary)={gap.get('auc_drop_vs_synthetic', 0.0):.4f}"
                    f"   ΔF1(primary→external)={gap.get('f1_drop_primary_to_external', 0.0):.4f}"
                )

        if self.result.benign_fpr:
            print("\n── False Positive Rate on Benign Data ─────────────────────────────")
            for name, value in self.result.benign_fpr.items():
                print(f"  [{name}] FPR={value:.4f}")

        if self.result.white_box_metrics:
            print("\n── White-Box Evasion Benchmark ───────────────────────────────────")
            print(self.result.dataset_table("White-box evasion", self.result.white_box_metrics))

        if self.result.failure_cases:
            print("\n── Failure Analysis (Top Cases) ─────────────────────────────────")
            for key in ("missed_attacks", "false_positives"):
                cases = self.result.failure_cases.get(key, [])
                print(f"  {key}: {len(cases)} examples")
                for case in cases[:5]:
                    print(
                        f"    - {case.get('id')} | risk={float(case.get('risk', 0.0)):.3f}"
                        f" | {str(case.get('text', ''))[:90]}"
                    )

        if self.result.default_threshold_results:
            print("\n── Default Threshold Results ─────────────────────────────────────")
            print(self.result.dataset_table("Default threshold", self.result.default_threshold_results))

        if self.result.optimized_threshold_results:
            print("\n── Optimized Threshold Results ───────────────────────────────────")
            print(self.result.dataset_table("Optimized threshold", self.result.optimized_threshold_results))

        best = self.result.best_f1()
        fast = self.result.fastest()
        print(f"\n  ✓ Best F1       : {best.config_name}  (F1={best.metrics.f1:.4f})")
        print(f"  ✓ Fastest       : {fast.config_name}  ({fast.latency.mean_ms:.2f} ms/req)")
        print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def to_json(self, path: str | Path) -> None:
        """Serialise full result to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = self._build_payload()
        with p.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[ReportSerializer] JSON written → {p}")

    def to_json_str(self) -> str:
        """Return the JSON payload as a string."""
        return json.dumps(self._build_payload(), indent=2)

    def _build_payload(self) -> dict:
        result = self.result
        return {
            "summary": {
                "n_train": result.n_train,
                "n_synthetic_test": result.n_synthetic_test,
                "n_real_world_test": result.n_real_world_test,
                "best_f1_config": result.best_f1().config_name,
                "fastest_config": result.fastest().config_name,
                "auc_small_dataset_note": "Note: AUC may be unreliable for small datasets",
            },
            "configurations": [
                {
                    "config_name": cfg.config_name,
                    "mode": cfg.mode,
                    "metrics": cfg.metrics.to_dict(),
                    "latency": cfg.latency.to_dict(),
                    "per_category": {
                        cat: m.to_dict()
                        for cat, m in cfg.per_category.items()
                    },
                }
                for cfg in result.configs()
            ],
            "synthetic_metrics": {
                name: m.to_dict() for name, m in result.synthetic_metrics.items()
            },
            "real_world_metrics": {
                name: m.to_dict()
                for name, m in result.real_world_metrics.items()
            },
            "external_metrics": {
                name: m.to_dict() for name, m in result.external_metrics.items()
            },
            "synthetic_baseline_metrics": {
                name: m.to_dict() for name, m in result.synthetic_baseline_metrics.items()
            },
            "real_world_baseline_metrics": {
                name: m.to_dict() for name, m in result.real_world_baseline_metrics.items()
            },
            "benign_fpr": result.benign_fpr,
            "white_box_metrics": {
                name: m.to_dict() for name, m in result.white_box_metrics.items()
            },
            "cross_validation": {
                name: {
                    "config_name": summary.config_name,
                    "folds": summary.folds,
                    "f1_mean": summary.f1_mean,
                    "f1_std": summary.f1_std,
                    "auc_mean": summary.auc_mean,
                    "auc_std": summary.auc_std,
                }
                for name, summary in result.cross_validation.items()
            },
            "confidence_intervals": {
                name: {
                    metric: {
                        "metric": ci.metric,
                        "mean": ci.mean,
                        "lower": ci.lower,
                        "upper": ci.upper,
                        "confidence": ci.confidence,
                    }
                    for metric, ci in ci_map.items()
                }
                for name, ci_map in result.confidence_intervals.items()
            },
            "domain_shift": result.domain_shift,
            "failure_cases": result.failure_cases,
            "default_threshold_results": {
                name: m.to_dict() for name, m in result.default_threshold_results.items()
            },
            "optimized_threshold_results": {
                name: m.to_dict() for name, m in result.optimized_threshold_results.items()
            },
            "threshold_recommendations": result.threshold_recommendations,
        }

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def to_csv(self, path: str | Path) -> None:
        """
        Write a flat CSV with one row per configuration.

        Columns: config_name, mode, precision, recall, f1, accuracy,
                 roc_auc, average_precision, mean_latency_ms,
                 p95_latency_ms, throughput_rps.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "config_name", "mode",
            "precision", "recall", "f1", "accuracy",
            "roc_auc", "average_precision",
            "mean_latency_ms", "p95_latency_ms", "throughput_rps",
            "tp", "fp", "tn", "fn",
        ]
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for cfg in self.result.configs():
                m = cfg.metrics
                l = cfg.latency
                writer.writerow({
                    "config_name": cfg.config_name,
                    "mode": cfg.mode,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "accuracy": m.accuracy,
                    "roc_auc": m.roc_auc if m.roc_auc is not None else "",
                    "average_precision": (
                        m.average_precision if m.average_precision is not None else ""
                    ),
                    "mean_latency_ms": l.mean_ms,
                    "p95_latency_ms": l.p95_ms,
                    "throughput_rps": l.throughput_rps,
                    "tp": m.confusion.tp,
                    "fp": m.confusion.fp,
                    "tn": m.confusion.tn,
                    "fn": m.confusion.fn,
                })
        print(f"[ReportSerializer] CSV written → {p}")

    # ------------------------------------------------------------------
    # Per-category CSV
    # ------------------------------------------------------------------

    def category_csv(self, path: str | Path) -> None:
        """Write a per-category breakdown CSV."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["config_name", "category", "precision", "recall", "f1", "accuracy"]
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for cfg in self.result.configs():
                for cat, m in sorted(cfg.per_category.items()):
                    writer.writerow({
                        "config_name": cfg.config_name,
                        "category": cat,
                        "precision": m.precision,
                        "recall": m.recall,
                        "f1": m.f1,
                        "accuracy": m.accuracy,
                    })
        print(f"[ReportSerializer] Category CSV written → {p}")
