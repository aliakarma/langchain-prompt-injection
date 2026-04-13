"""
tests/unit/evaluation/test_performance.py
──────────────────────────────────────────
Unit tests for ReportSerializer (console, JSON, CSV outputs).
"""

import sys, os, json, csv, tempfile, io
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from prompt_injection.evaluation.benchmark import BenchmarkRunner
from prompt_injection.evaluation.dataset import SyntheticDataset
from prompt_injection.evaluation.report import ReportSerializer


@pytest.fixture(scope="module")
def report_serializer():
    ds = SyntheticDataset(n_injections=60, n_benign=60, seed=42).generate()
    train_ds, test_ds = ds.train_test_split(test_size=0.20, seed=42)
    result = BenchmarkRunner(n_latency_runs=3, sweep_thresholds=False).run(train_ds, test_ds)
    return ReportSerializer(result)


class TestReportSerializerJSON:
    def test_to_json_str_valid_json(self, report_serializer):
        s = report_serializer.to_json_str()
        parsed = json.loads(s)
        assert isinstance(parsed, dict)

    def test_json_top_level_keys(self, report_serializer):
        parsed = json.loads(report_serializer.to_json_str())
        assert set(parsed.keys()) == {"summary", "configurations", "real_world_metrics"}

    def test_json_has_three_configurations(self, report_serializer):
        parsed = json.loads(report_serializer.to_json_str())
        assert len(parsed["configurations"]) == 3

    def test_json_each_config_has_metrics_and_latency(self, report_serializer):
        parsed = json.loads(report_serializer.to_json_str())
        for cfg in parsed["configurations"]:
            assert "metrics" in cfg
            assert "latency" in cfg

    def test_to_json_writes_file(self, report_serializer):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            report_serializer.to_json(path)
            with open(path) as f:
                parsed = json.load(f)
            assert "configurations" in parsed

    def test_json_summary_has_best_and_fastest(self, report_serializer):
        parsed = json.loads(report_serializer.to_json_str())
        assert "best_f1_config" in parsed["summary"]
        assert "fastest_config" in parsed["summary"]


class TestReportSerializerCSV:
    def test_to_csv_writes_three_rows(self, report_serializer):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bench.csv")
            report_serializer.to_csv(path)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 3

    def test_csv_has_required_columns(self, report_serializer):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bench.csv")
            report_serializer.to_csv(path)
            with open(path) as f:
                reader = csv.DictReader(f)
                for required in ["config_name", "f1", "precision", "recall",
                                  "mean_latency_ms", "tp", "fp", "tn", "fn"]:
                    assert required in reader.fieldnames

    def test_category_csv_has_multiple_rows(self, report_serializer):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cats.csv")
            report_serializer.category_csv(path)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) > 3   # 3 configs × N categories

    def test_category_csv_columns(self, report_serializer):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cats.csv")
            report_serializer.category_csv(path)
            with open(path) as f:
                reader = csv.DictReader(f)
                for col in ["config_name", "category", "precision", "recall", "f1"]:
                    assert col in reader.fieldnames


class TestReportSerializerConsole:
    def test_print_summary_runs_without_error(self, report_serializer, capsys):
        report_serializer.print_summary()
        captured = capsys.readouterr()
        assert "ABLATION BENCHMARK" in captured.out
        assert "A: Regex only" in captured.out
        assert "B: Regex + Scoring" in captured.out
        assert "C: + Classifier" in captured.out

    def test_print_summary_shows_best_and_fastest(self, report_serializer, capsys):
        report_serializer.print_summary()
        captured = capsys.readouterr()
        assert "Best F1" in captured.out
        assert "Fastest" in captured.out
