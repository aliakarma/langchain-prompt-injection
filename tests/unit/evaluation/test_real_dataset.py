import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from prompt_injection.evaluation.real_dataset import (
    REQUIRED_EXTERNAL_RAW_FILES,
    load_required_external_raw_datasets,
    split_external_raw_dataset,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@pytest.fixture
def populated_external_raw_dir() -> Path:
    tmp_dir = tempfile.TemporaryDirectory()
    base = Path(tmp_dir.name)

    per_file_counts = {
        "hackaprompt.jsonl": {"label_1": 20},
        "prompt-injections.jsonl": {"label_0": 20, "label_1": 20},
        "jailbreak.jsonl": {"label_0": 20, "label_1": 20},
    }

    for file_name, config in per_file_counts.items():
        rows: list[dict] = []
        for label, count in ((0, config.get("label_0", 0)), (1, config.get("label_1", 0))):
            for index in range(count):
                rows.append(
                    {
                        "id": f"{Path(file_name).stem}-{label}-{index}",
                        "text": f"{file_name} sample label={label} index={index}",
                        "label": label,
                    }
                )
        _write_jsonl(base / file_name, rows)

    try:
        yield base
    finally:
        tmp_dir.cleanup()


class TestLoadRequiredExternalRawDatasets:
    def test_loads_only_required_files(self, populated_external_raw_dir):
        bundle = load_required_external_raw_datasets(populated_external_raw_dir, seed=42)
        assert sorted(bundle.file_summaries.keys()) == sorted(REQUIRED_EXTERNAL_RAW_FILES)
        assert len(bundle.records) == 100

    def test_removes_empty_and_exact_duplicates_within_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_jsonl(
                base / "hackaprompt.jsonl",
                [
                    {"id": "a1", "text": "attack 1", "label": 1},
                    {"id": "a2", "text": "attack 1", "label": 1},
                    {"id": "a3", "text": "   ", "label": 1},
                ],
            )
            _write_jsonl(
                base / "prompt-injections.jsonl",
                [
                    {"id": "b1", "text": "benign 1", "label": 0},
                    {"id": "b2", "text": "attack 2", "label": 1},
                ],
            )
            _write_jsonl(
                base / "jailbreak.jsonl",
                [
                    {"id": "c1", "text": "benign 2", "label": 0},
                    {"id": "c2", "text": "attack 3", "label": 1},
                ],
            )

            bundle = load_required_external_raw_datasets(base, seed=42)
            summary = bundle.file_summaries["hackaprompt.jsonl"]
            assert summary.removed_duplicates == 1
            assert summary.removed_empty == 1
            assert len(bundle.records) == 5

    def test_rejects_invalid_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            _write_jsonl(base / "hackaprompt.jsonl", [{"id": "a1", "text": "attack 1"}])
            _write_jsonl(base / "prompt-injections.jsonl", [{"id": "b1", "text": "benign 1", "label": 0}])
            _write_jsonl(base / "jailbreak.jsonl", [{"id": "c1", "text": "benign 2", "label": 0}])

            with pytest.raises(ValueError, match="missing required fields"):
                load_required_external_raw_datasets(base, seed=42)


class TestSplitExternalRawDataset:
    def test_strict_split_is_text_disjoint(self, populated_external_raw_dir):
        bundle = load_required_external_raw_datasets(populated_external_raw_dir, seed=42)
        split = split_external_raw_dataset(bundle.records, seed=42)
        split.assert_text_disjoint()

        all_sizes = split.train.size() + split.validation.size() + split.test.size()
        assert all_sizes == len(bundle.records)

    def test_split_preserves_composite_distribution(self, populated_external_raw_dir):
        bundle = load_required_external_raw_datasets(populated_external_raw_dir, seed=42)
        split = split_external_raw_dataset(bundle.records, seed=42)

        for subset in (split.train, split.validation, split.test):
            source_labels = subset.source_label_distribution()
            assert "hackaprompt" in source_labels
            assert "prompt-injections" in source_labels
            assert "jailbreak" in source_labels
            assert subset.label_distribution()[1] > 0

