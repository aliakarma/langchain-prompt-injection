from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_KEYS = {"id", "text", "label", "attack_category"}


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i} invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{i} row must be a JSON object")
            rows.append(row)
    return rows


def validate_schema(path: Path, rows: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for i, row in enumerate(rows, start=1):
        keys = set(row.keys())
        if keys != REQUIRED_KEYS:
            issues.append(
                f"{path}:{i} schema mismatch; expected {sorted(REQUIRED_KEYS)}, got {sorted(keys)}"
            )
            continue
        if not isinstance(row["id"], str) or not row["id"].strip():
            issues.append(f"{path}:{i} invalid id")
        if not isinstance(row["text"], str) or not row["text"].strip():
            issues.append(f"{path}:{i} invalid text")
        if row["label"] not in (0, 1):
            issues.append(f"{path}:{i} label must be 0 or 1")
        if row["label"] == 0 and row["attack_category"] is not None:
            issues.append(f"{path}:{i} benign row must have attack_category=null")
        if row["label"] == 1 and (not isinstance(row["attack_category"], str) or not row["attack_category"].strip()):
            issues.append(f"{path}:{i} injection row must have a non-empty attack_category")
    return issues


def duplicate_stats(rows: list[dict[str, Any]]) -> tuple[int, list[str]]:
    counts: Counter[str] = Counter(normalize_text(str(r.get("text", ""))) for r in rows)
    dups = [text for text, c in counts.items() if c > 1]
    return len(dups), dups


def print_report(path: Path, rows: list[dict[str, Any]]) -> int:
    schema_issues = validate_schema(path, rows)
    n_dup, dup_texts = duplicate_stats(rows)

    label_counts = Counter(int(r["label"]) for r in rows if "label" in r)
    print(f"\n== {path} ==")
    print(f"rows: {len(rows)}")
    print(f"labels: {dict(label_counts)}")
    print(f"duplicate_texts: {n_dup}")
    if n_dup:
        for t in dup_texts[:5]:
            print(f"  duplicate sample: {t[:120]}")
    if schema_issues:
        print(f"schema_issues: {len(schema_issues)}")
        for issue in schema_issues[:10]:
            print(f"  {issue}")
    else:
        print("schema_issues: 0")

    return len(schema_issues) + n_dup


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dataset schema, duplicates, and label distribution.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="JSONL files to validate. Defaults to the repository dataset targets.",
    )
    args = parser.parse_args()

    default_paths = [
        Path("data/benign/benign_corpus_v2.jsonl"),
        Path("data/external/synthetic_stress_test.jsonl"),
        Path("data/real/injections_real_v2.jsonl"),
    ]
    paths = [Path(p) for p in args.paths] if args.paths else default_paths

    failures = 0
    for path in paths:
        if not path.exists():
            print(f"\n== {path} ==")
            print("missing file")
            failures += 1
            continue
        rows = load_jsonl(path)
        failures += print_report(path, rows)

    if failures:
        print(f"\nIntegrity check FAILED with {failures} issue(s).")
        return 1

    print("\nIntegrity check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
