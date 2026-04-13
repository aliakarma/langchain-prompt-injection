"""Run research-grade evaluation with external and benign corpora."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, "src")

from prompt_injection.evaluation.benchmark import BenchmarkRunner
from prompt_injection.evaluation.dataset import SyntheticDataset
from prompt_injection.evaluation.report import ReportSerializer


def main() -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    ds = SyntheticDataset(n_injections=250, n_benign=250, seed=42).generate()
    train_ds, synthetic_test_ds = ds.train_test_split(test_size=0.20, seed=42)

    real_ds = SyntheticDataset()
    real_ds.load_from_path("data/real/injections_real.jsonl")
    real_ds.load_from_path("data/real/benign_real.jsonl")

    external_ds = SyntheticDataset()
    ext_path = Path("data/external/hackaprompt_like.jsonl")
    if ext_path.exists():
        external_ds.load_external_dataset(ext_path, train_texts=set(train_ds.texts()))

    benign_ds = SyntheticDataset()
    benign_path = Path("data/benign/benign_corpus.jsonl")
    if benign_path.exists():
        benign_ds.load_from_path(benign_path)

    runner = BenchmarkRunner(threshold=0.50, n_latency_runs=30, latency_budget_ms=10.0, sweep_thresholds=True)
    result = runner.run(
        train_ds,
        real_ds,
        synthetic_test_ds,
        external_eval_dataset=(external_ds if len(external_ds) else None),
        benign_dataset=(benign_ds if len(benign_ds) else None),
    )

    serializer = ReportSerializer(result)
    serializer.print_summary()
    serializer.to_json(reports_dir / "benchmark_research.json")
    serializer.to_csv(reports_dir / "benchmark_research.csv")
    serializer.category_csv(reports_dir / "category_breakdown_research.csv")

    with (reports_dir / "failure_analysis.json").open("w", encoding="utf-8") as fh:
        json.dump(result.failure_cases, fh, indent=2)


if __name__ == "__main__":
    main()
