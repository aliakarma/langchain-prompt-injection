"""
tests/unit/evaluation/test_dataset.py
──────────────────────────────────────
Unit tests for SyntheticDataset and DataRecord.
"""

import sys, os, json, tempfile, pathlib
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from prompt_injection.evaluation.dataset import SyntheticDataset, DataRecord


@pytest.fixture(scope="module")
def ds():
    return SyntheticDataset(n_injections=100, n_benign=100, seed=42).generate()


class TestGeneration:
    def test_total_count(self, ds):
        assert len(ds) == 200

    def test_class_balance(self, ds):
        assert ds.n_positive == 100
        assert ds.n_negative == 100

    def test_labels_binary(self, ds):
        assert all(l in (0, 1) for l in ds.labels())

    def test_ids_unique(self, ds):
        ids = [r.id for r in ds]
        assert len(ids) == len(set(ids))

    def test_reproducible(self):
        ds1 = SyntheticDataset(n_injections=50, n_benign=50, seed=7).generate()
        ds2 = SyntheticDataset(n_injections=50, n_benign=50, seed=7).generate()
        assert ds1.texts() == ds2.texts()

    def test_different_seeds_differ(self):
        ds1 = SyntheticDataset(n_injections=50, n_benign=50, seed=1).generate()
        ds2 = SyntheticDataset(n_injections=50, n_benign=50, seed=2).generate()
        assert ds1.texts() != ds2.texts()

    def test_all_source_types_valid(self, ds):
        valid = {"user", "rag", "tool", "file", "web", "unknown"}
        for r in ds:
            assert r.source_type in valid, f"Invalid source_type: {r.source_type}"

    def test_attack_category_none_for_benign(self, ds):
        for r in ds:
            if r.label == 0:
                assert r.attack_category is None

    def test_attack_category_set_for_injection(self, ds):
        for r in ds:
            if r.label == 1:
                assert r.attack_category is not None and len(r.attack_category) > 0

    def test_severity_none_for_benign(self, ds):
        for r in ds:
            if r.label == 0:
                assert r.severity is None


class TestSplits:
    def test_no_overlap(self, ds):
        train, test = ds.train_test_split(test_size=0.20, seed=42)
        train_ids = {r.id for r in train}
        test_ids  = {r.id for r in test}
        assert len(train_ids & test_ids) == 0

    def test_all_records_accounted_for(self, ds):
        train, test = ds.train_test_split(test_size=0.20, seed=42)
        assert len(train) + len(test) == len(ds)

    def test_test_size_approximate(self, ds):
        train, test = ds.train_test_split(test_size=0.20, seed=42)
        assert abs(len(test) / len(ds) - 0.20) < 0.05

    def test_stratified_balance(self, ds):
        _, test = ds.train_test_split(test_size=0.20, seed=42)
        pos = sum(1 for r in test if r.label == 1)
        neg = sum(1 for r in test if r.label == 0)
        assert abs(pos - neg) <= 5


class TestIO:
    def test_save_and_reload(self, ds):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ds.jsonl")
            ds.save_to_path(path)
            ds2 = SyntheticDataset().load_from_path(path)
            assert len(ds2) == len(ds)
            assert ds2.texts() == ds.texts()

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            SyntheticDataset().load_from_path("/nonexistent/path/file.jsonl")

    def test_load_real_world_files(self):
        base = pathlib.Path(__file__).parents[3] / "data" / "real"
        if not base.exists():
            pytest.skip("data/real not found")
        ds = SyntheticDataset()
        ds.load_from_path(base / "injections_real.jsonl")
        ds.load_from_path(base / "benign_real.jsonl")
        assert len(ds) == 50
        assert ds.n_positive == 25
        assert ds.n_negative == 25


class TestFilters:
    def test_filter_by_source(self, ds):
        user_ds = ds.filter_by_source("user")
        assert all(r.source_type == "user" for r in user_ds)

    def test_filter_by_category(self, ds):
        cat_ds = ds.filter_by_category("dan_jailbreak")
        assert all(r.attack_category == "dan_jailbreak" for r in cat_ds)

    def test_data_record_to_dict(self, ds):
        r = ds.records()[0]
        d = r.to_dict()
        assert "id" in d and "text" in d and "label" in d
