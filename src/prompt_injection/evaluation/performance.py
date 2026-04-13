"""
evaluation/performance.py
─────────────────────────
Latency measurement and throughput estimation for all three detector
configurations.

Metrics collected per run
--------------------------
  mean_ms   – arithmetic mean latency across all texts × n_runs.
  std_ms    – standard deviation.
  p50_ms    – 50th percentile (median).
  p95_ms    – 95th percentile.
  p99_ms    – 99th percentile.
  min_ms    – fastest observed scan.
  max_ms    – slowest observed scan.
  throughput_rps – estimated requests/second (single-threaded, 1 / mean_ms × 1000).

Usage
-----
    from prompt_injection.evaluation.performance import PerformanceProfiler
    from prompt_injection.detector import InjectionDetector

    profiler = PerformanceProfiler()
    report   = profiler.profile(InjectionDetector(mode="hybrid"), texts, n_runs=50)
    print(report.summary())
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_injection.detector import InjectionDetector


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LatencyReport:
    """
    Latency statistics for a profiling run.

    All time values are in milliseconds (ms).
    """

    config_name: str
    mode: str
    n_texts: int
    n_runs: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_rps: float
    budget_ms: float | None
    budget_ok: bool | None

    def summary(self) -> str:
        lines = [
            f"Config       : {self.config_name}",
            f"Mode         : {self.mode}",
            f"Texts        : {self.n_texts} × {self.n_runs} runs",
            f"Mean         : {self.mean_ms:.3f} ms",
            f"Std          : {self.std_ms:.3f} ms",
            f"P50          : {self.p50_ms:.3f} ms",
            f"P95          : {self.p95_ms:.3f} ms",
            f"P99          : {self.p99_ms:.3f} ms",
            f"Min          : {self.min_ms:.3f} ms",
            f"Max          : {self.max_ms:.3f} ms",
            f"Throughput   : {self.throughput_rps:.0f} req/s (single-threaded)",
        ]
        if self.budget_ms is not None:
            status = "✓ within budget" if self.budget_ok else "✗ exceeds budget"
            lines.append(f"Budget check : {self.budget_ms:.1f} ms → {status}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "config_name": self.config_name,
            "mode": self.mode,
            "n_texts": self.n_texts,
            "n_runs": self.n_runs,
            "mean_ms": round(self.mean_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "throughput_rps": round(self.throughput_rps, 1),
            "budget_ms": self.budget_ms,
            "budget_ok": self.budget_ok,
        }


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class PerformanceProfiler:
    """
    Measures per-request latency for an ``InjectionDetector`` instance.

    Parameters
    ----------
    warmup_runs : int
        Number of warmup scans discarded before measurement begins.
        Warmup amortises JIT / import overhead.
    """

    def __init__(self, warmup_runs: int = 3) -> None:
        self.warmup_runs = warmup_runs

    def profile(
        self,
        detector: "InjectionDetector",
        texts: list[str],
        *,
        n_runs: int = 20,
        budget_ms: float | None = None,
        config_name: str | None = None,
    ) -> LatencyReport:
        """
        Time ``detector.scan()`` across all *texts* for *n_runs* repetitions.

        Parameters
        ----------
        detector : InjectionDetector
            Pre-configured detector to profile.
        texts : list[str]
            Representative texts (mix of injections and benign is ideal).
        n_runs : int
            Number of timing repetitions per text.
        budget_ms : float | None
            Optional per-request latency budget.  Populates ``budget_ok``
            in the report.
        config_name : str | None
            Label for this profile run.

        Returns
        -------
        LatencyReport
        """
        if not texts:
            raise ValueError("texts must be non-empty.")

        name = config_name or f"detector[{detector.mode}]"

        # Warmup
        for _ in range(self.warmup_runs):
            detector.scan(texts[0])

        # Timed runs
        sample_ms: list[float] = []
        for _ in range(n_runs):
            for text in texts:
                t0 = time.perf_counter()
                detector.scan(text)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                sample_ms.append(elapsed_ms)

        sample_ms.sort()
        mean_ms = statistics.mean(sample_ms)
        std_ms = statistics.stdev(sample_ms) if len(sample_ms) > 1 else 0.0
        p50 = _percentile(sample_ms, 50)
        p95 = _percentile(sample_ms, 95)
        p99 = _percentile(sample_ms, 99)
        throughput = (1000.0 / mean_ms) if mean_ms > 0 else float("inf")

        budget_ok = (mean_ms <= budget_ms) if budget_ms is not None else None

        return LatencyReport(
            config_name=name,
            mode=detector.mode,
            n_texts=len(texts),
            n_runs=n_runs,
            mean_ms=round(mean_ms, 3),
            std_ms=round(std_ms, 3),
            p50_ms=round(p50, 3),
            p95_ms=round(p95, 3),
            p99_ms=round(p99, 3),
            min_ms=round(sample_ms[0], 3),
            max_ms=round(sample_ms[-1], 3),
            throughput_rps=round(throughput, 1),
            budget_ms=budget_ms,
            budget_ok=budget_ok,
        )

    def compare(
        self,
        detectors: list[tuple[str, "InjectionDetector"]],
        texts: list[str],
        *,
        n_runs: int = 20,
        budget_ms: float | None = None,
    ) -> list[LatencyReport]:
        """
        Profile multiple detectors and return a list of ``LatencyReport``
        objects sorted by mean latency (fastest first).

        Parameters
        ----------
        detectors : list[tuple[str, InjectionDetector]]
            List of (name, detector) pairs.
        texts : list[str]
            Shared representative texts.
        n_runs : int
            Timing repetitions.
        budget_ms : float | None
            Shared latency budget for budget checks.
        """
        reports = [
            self.profile(det, texts, n_runs=n_runs,
                         budget_ms=budget_ms, config_name=name)
            for name, det in detectors
        ]
        return sorted(reports, key=lambda r: r.mean_ms)


def _percentile(sorted_data: list[float], p: int) -> float:
    """Linear interpolation percentile on a sorted list."""
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


# ---------------------------------------------------------------------------
# Quick one-shot helper
# ---------------------------------------------------------------------------


def time_detector(
    detector: "InjectionDetector",
    texts: list[str],
    n_runs: int = 20,
) -> LatencyReport:
    """
    Convenience wrapper: profile *detector* on *texts* and return a report.

    Example::

        report = time_detector(InjectionDetector(mode="rules"), sample_texts)
        print(report.summary())
    """
    return PerformanceProfiler().profile(detector, texts, n_runs=n_runs)
