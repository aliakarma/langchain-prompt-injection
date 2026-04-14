"""
Strict loaders and splitters for the required external raw datasets.

These helpers intentionally support only the production research workflow
that operates on:

    - hackaprompt.jsonl
    - prompt-injections.jsonl
    - jailbreak.jsonl

Each record must contain:
    {id: str, text: str, label: int}
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split


REQUIRED_EXTERNAL_RAW_FILES = (
    "hackaprompt.jsonl",
    "prompt-injections.jsonl",
    "jailbreak.jsonl",
)


@dataclass(frozen=True)
class ExternalRawRecord:
    id: str
    text: str
    label: int
    source_dataset: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "source_dataset": self.source_dataset,
        }


@dataclass(frozen=True)
class ExternalRawFileSummary:
    file_name: str
    total_samples: int
    kept_samples: int
    removed_empty: int
    removed_duplicates: int
    label_distribution: dict[int, int]
    random_samples: list[ExternalRawRecord]

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "total_samples": self.total_samples,
            "kept_samples": self.kept_samples,
            "removed_empty": self.removed_empty,
            "removed_duplicates": self.removed_duplicates,
            "label_distribution": self.label_distribution,
            "random_samples": [sample.to_dict() for sample in self.random_samples],
        }


@dataclass(frozen=True)
class ExternalRawDatasetBundle:
    records: list[ExternalRawRecord]
    file_summaries: dict[str, ExternalRawFileSummary]
    removed_cross_dataset_duplicates: int

    def label_distribution(self) -> dict[int, int]:
        return dict(Counter(record.label for record in self.records))

    def source_distribution(self) -> dict[str, int]:
        return dict(Counter(record.source_dataset for record in self.records))

    def texts(self) -> list[str]:
        return [record.text for record in self.records]


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    records: list[ExternalRawRecord]

    def size(self) -> int:
        return len(self.records)

    def label_distribution(self) -> dict[int, int]:
        return dict(Counter(record.label for record in self.records))

    def source_distribution(self) -> dict[str, int]:
        return dict(Counter(record.source_dataset for record in self.records))

    def source_label_distribution(self) -> dict[str, dict[int, int]]:
        distribution: dict[str, Counter[int]] = {}
        for record in self.records:
            distribution.setdefault(record.source_dataset, Counter())[record.label] += 1
        return {
            source: dict(sorted(counter.items()))
            for source, counter in sorted(distribution.items())
        }

    def texts(self) -> list[str]:
        return [record.text for record in self.records]

    def labels(self) -> list[int]:
        return [record.label for record in self.records]


@dataclass(frozen=True)
class ExternalRawSplitBundle:
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit

    def assert_text_disjoint(self) -> None:
        train_texts = set(self.train.texts())
        validation_texts = set(self.validation.texts())
        test_texts = set(self.test.texts())
        if train_texts & validation_texts:
            raise ValueError("Train and validation splits overlap by text.")
        if train_texts & test_texts:
            raise ValueError("Train and test splits overlap by text.")
        if validation_texts & test_texts:
            raise ValueError("Validation and test splits overlap by text.")


def load_required_external_raw_datasets(
    base_dir: str | Path,
    *,
    seed: int = 42,
) -> ExternalRawDatasetBundle:
    """Load and validate the three required external raw JSONL files."""
    root = Path(base_dir)
    summaries: dict[str, ExternalRawFileSummary] = {}
    merged_records: list[ExternalRawRecord] = []

    for file_name in REQUIRED_EXTERNAL_RAW_FILES:
        path = root / file_name
        file_records, summary = _load_one_external_raw_file(path, seed=seed)
        summaries[file_name] = summary
        merged_records.extend(file_records)

    deduped_records: list[ExternalRawRecord] = []
    seen_texts: set[str] = set()
    removed_cross_dataset_duplicates = 0

    for record in merged_records:
        if record.text in seen_texts:
            removed_cross_dataset_duplicates += 1
            continue
        seen_texts.add(record.text)
        deduped_records.append(record)

    return ExternalRawDatasetBundle(
        records=deduped_records,
        file_summaries=summaries,
        removed_cross_dataset_duplicates=removed_cross_dataset_duplicates,
    )


def split_external_raw_dataset(
    records: list[ExternalRawRecord],
    *,
    seed: int = 42,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
) -> ExternalRawSplitBundle:
    """Create strict train/validation/test splits with composite stratification."""
    if not records:
        raise ValueError("Cannot split an empty dataset.")

    total = train_size + validation_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_size + validation_size + test_size must equal 1.0")

    indices = list(range(len(records)))
    strata = [_stratification_key(record) for record in records]

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(validation_size + test_size),
        random_state=seed,
        stratify=strata,
    )
    temp_records = [records[index] for index in temp_indices]
    temp_indices_relative = list(range(len(temp_records)))
    temp_strata = [_stratification_key(record) for record in temp_records]

    validation_relative, test_relative = train_test_split(
        temp_indices_relative,
        test_size=(test_size / (validation_size + test_size)),
        random_state=seed,
        stratify=temp_strata,
    )

    validation_indices = [temp_indices[index] for index in validation_relative]
    test_indices = [temp_indices[index] for index in test_relative]

    split_bundle = ExternalRawSplitBundle(
        train=DatasetSplit("train", [records[index] for index in train_indices]),
        validation=DatasetSplit("validation", [records[index] for index in validation_indices]),
        test=DatasetSplit("test", [records[index] for index in test_indices]),
    )
    split_bundle.assert_text_disjoint()
    return split_bundle


def _load_one_external_raw_file(
    path: Path,
    *,
    seed: int,
) -> tuple[list[ExternalRawRecord], ExternalRawFileSummary]:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")

    total_samples = 0
    removed_empty = 0
    removed_duplicates = 0
    label_counter: Counter[int] = Counter()
    deduped_records: list[ExternalRawRecord] = []
    seen_texts: set[str] = set()

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_samples += 1
            raw = json.loads(line)
            record = _validate_external_raw_record(raw, line_number=line_number, source_dataset=path.stem)
            label_counter[record.label] += 1

            if not record.text.strip():
                removed_empty += 1
                continue
            if record.text in seen_texts:
                removed_duplicates += 1
                continue

            seen_texts.add(record.text)
            deduped_records.append(record)

    rng = random.Random(seed)
    sample_count = min(3, len(deduped_records))
    random_samples = rng.sample(deduped_records, sample_count) if sample_count else []

    summary = ExternalRawFileSummary(
        file_name=path.name,
        total_samples=total_samples,
        kept_samples=len(deduped_records),
        removed_empty=removed_empty,
        removed_duplicates=removed_duplicates,
        label_distribution=dict(sorted(label_counter.items())),
        random_samples=random_samples,
    )
    return deduped_records, summary


def _validate_external_raw_record(
    raw: object,
    *,
    line_number: int,
    source_dataset: str,
) -> ExternalRawRecord:
    if not isinstance(raw, dict):
        raise ValueError(f"{source_dataset}:{line_number} must be a JSON object.")

    required_fields = ("id", "text", "label")
    missing = [field for field in required_fields if field not in raw]
    if missing:
        raise ValueError(f"{source_dataset}:{line_number} missing required fields: {missing}")

    raw_id = raw["id"]
    raw_text = raw["text"]
    raw_label = raw["label"]

    if not isinstance(raw_id, str):
        raise ValueError(f"{source_dataset}:{line_number} field 'id' must be a string.")
    if not isinstance(raw_text, str):
        raise ValueError(f"{source_dataset}:{line_number} field 'text' must be a string.")
    if isinstance(raw_label, bool) or not isinstance(raw_label, int):
        raise ValueError(f"{source_dataset}:{line_number} field 'label' must be an int.")
    if raw_label not in {0, 1}:
        raise ValueError(f"{source_dataset}:{line_number} field 'label' must be 0 or 1.")

    return ExternalRawRecord(
        id=raw_id,
        text=raw_text,
        label=raw_label,
        source_dataset=source_dataset,
    )


def _stratification_key(record: ExternalRawRecord) -> str:
    return f"{record.source_dataset}::{record.label}"
