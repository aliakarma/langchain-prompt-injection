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

        if self.result.real_world_metrics:
            print("\n── Real-World Sample Metrics ───────────────────────────────────────")
            for name, m in self.result.real_world_metrics.items():
                print(f"\n  [{name}]")
                for line in m.summary().split("\n"):
                    print(f"    {line}")

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
                "n_test": result.n_test,
                "best_f1_config": result.best_f1().config_name,
                "fastest_config": result.fastest().config_name,
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
            "real_world_metrics": {
                name: m.to_dict()
                for name, m in result.real_world_metrics.items()
            },
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
