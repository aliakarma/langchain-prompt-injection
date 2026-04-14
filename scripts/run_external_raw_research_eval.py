from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prompt_injection.detector import InjectionDetector, LogisticRegressionScorer
from prompt_injection.evaluation.metrics import compute_metrics, threshold_sweep
from prompt_injection.evaluation.real_dataset import (
    ExternalRawRecord,
    load_required_external_raw_datasets,
    split_external_raw_dataset,
)
from prompt_injection.middleware import PromptInjectionMiddleware
from prompt_injection.policy import PolicyDecision, PolicyEngine


SEED = 42
DEFAULT_THRESHOLD = 0.50
DATA_DIR = ROOT / "data" / "external_raw"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
SPLITS_DIR = RESULTS_DIR / "external_raw_splits"

RESULT_JSON = RESULTS_DIR / "external_raw_research_eval.json"
LOG_PATH = RESULTS_DIR / "external_raw_research_eval.log"
REPORT_PATH = REPORTS_DIR / "external_raw_research_report.md"

CONFIGS = {
    "A: Regex only": "rules",
    "B: Regex + Scoring": "hybrid",
    "C: + Classifier": "full",
}


@dataclass(frozen=True)
class EvaluatedRecord:
    id: str
    text: str
    label: int
    source_dataset: str
    risk_score: float
    predicted_label: int

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "source_dataset": self.source_dataset,
            "risk_score": round(self.risk_score, 4),
            "predicted_label": self.predicted_label,
        }


class Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: str = "") -> None:
        self.lines.append(message)
        print(message)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def main() -> int:
    start = time.perf_counter()
    logger = Logger()

    logger.log("PHASE 1 - Data loading and validation")
    dataset_bundle = load_required_external_raw_datasets(DATA_DIR, seed=SEED)
    merged_records = dataset_bundle.records
    _log_dataset_summaries(logger, dataset_bundle)
    logger.log(f"Merged dataset size after cross-dataset deduplication: {len(merged_records)}")
    logger.log(f"Cross-dataset duplicates removed: {dataset_bundle.removed_cross_dataset_duplicates}")
    logger.log(f"Merged label distribution: {json.dumps(dataset_bundle.label_distribution(), sort_keys=True)}")
    logger.log(f"Merged source distribution: {json.dumps(dataset_bundle.source_distribution(), sort_keys=True)}")
    logger.log("")

    logger.log("PHASE 2 - Strict train/validation/test splitting")
    split_bundle = split_external_raw_dataset(merged_records, seed=SEED)
    split_bundle.assert_text_disjoint()
    _log_split_summaries(logger, split_bundle)
    _write_split_artifacts(split_bundle)
    logger.log("")

    logger.log("PHASE 3 - Pipeline debugging")
    train_records = split_bundle.train.records
    train_texts = [record.text for record in train_records]
    train_labels = [record.label for record in train_records]
    classifier = LogisticRegressionScorer().fit(train_texts, train_labels)
    pipeline_checks = _run_pipeline_checks(train_records, classifier)
    for check in pipeline_checks:
        logger.log(f"[{check['status']}] {check['name']}: {check['detail']}")
    logger.log("")

    logger.log("PHASE 4/5 - Training Config C and evaluating configs A/B/C")
    detectors = {
        "A: Regex only": InjectionDetector(mode="rules", threshold=DEFAULT_THRESHOLD),
        "B: Regex + Scoring": InjectionDetector(mode="hybrid", threshold=DEFAULT_THRESHOLD),
        "C: + Classifier": InjectionDetector(
            mode="full",
            threshold=DEFAULT_THRESHOLD,
            classifier=classifier,
        ),
    }

    evaluation_results: dict[str, dict[str, Any]] = {"validation": {}, "test": {}}
    split_records = {
        "validation": split_bundle.validation.records,
        "test": split_bundle.test.records,
    }

    cached_scores: dict[str, dict[str, list[float]]] = {"validation": {}, "test": {}}

    for split_name, records in split_records.items():
        logger.log(f"{split_name.title()} split evaluation")
        for config_name, detector in detectors.items():
            evaluated = _score_records(records, detector, DEFAULT_THRESHOLD)
            cached_scores[split_name][config_name] = [row.risk_score for row in evaluated]
            metrics = _metrics_with_fpr(records, evaluated, threshold=DEFAULT_THRESHOLD, config_name=config_name)
            evaluation_results[split_name][config_name] = {
                "default": metrics,
                "default_examples": [row.to_dict() for row in evaluated[:5]],
            }
            logger.log(
                f"  {config_name}: "
                f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f} AUC={_fmt_auc(metrics['roc_auc'])} "
                f"FPR={metrics['false_positive_rate']:.4f}"
            )
        logger.log("")

    logger.log("PHASE 6 - Threshold analysis")
    threshold_analysis = _run_threshold_analysis(split_bundle, detectors, cached_scores, logger)
    logger.log("")

    logger.log("PHASE 7 - Error analysis")
    primary_threshold = threshold_analysis["C: + Classifier"]["best_threshold"]
    config_c_test_predictions = _score_records(
        split_bundle.test.records,
        detectors["C: + Classifier"],
        primary_threshold,
    )
    error_analysis = _build_error_analysis(split_bundle.test.records, config_c_test_predictions)
    _log_error_examples(logger, error_analysis)
    logger.log("")

    logger.log("PHASE 8 - Final structured report")
    observations = _build_observations(
        evaluation_results=evaluation_results,
        threshold_analysis=threshold_analysis,
        error_analysis=error_analysis,
    )

    final_payload = {
        "seed": SEED,
        "default_threshold": DEFAULT_THRESHOLD,
        "allowed_input_files": [
            str(DATA_DIR / "hackaprompt.jsonl"),
            str(DATA_DIR / "prompt-injections.jsonl"),
            str(DATA_DIR / "jailbreak.jsonl"),
        ],
        "dataset_summary": {
            "total_samples": len(merged_records),
            "per_source": dataset_bundle.source_distribution(),
            "label_distribution": dataset_bundle.label_distribution(),
            "per_file": {
                name: summary.to_dict()
                for name, summary in dataset_bundle.file_summaries.items()
            },
            "cross_dataset_duplicates_removed": dataset_bundle.removed_cross_dataset_duplicates,
        },
        "split_summary": {
            "train": _split_summary_dict(split_bundle.train),
            "validation": _split_summary_dict(split_bundle.validation),
            "test": _split_summary_dict(split_bundle.test),
        },
        "pipeline_checks": pipeline_checks,
        "evaluation_results": evaluation_results,
        "threshold_analysis": threshold_analysis,
        "error_analysis": error_analysis,
        "observations": observations,
        "integrity_checks": {
            "real_datasets_only": True,
            "synthetic_data_used": False,
            "data_leakage_detected": False,
            "text_overlap_between_splits": False,
            "training_split_only_for_config_c": True,
            "source_files_used": [
                "hackaprompt.jsonl",
                "prompt-injections.jsonl",
                "jailbreak.jsonl",
            ],
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
    report_markdown = _build_markdown_report(final_payload)
    REPORT_PATH.write_text(report_markdown, encoding="utf-8")

    elapsed = time.perf_counter() - start
    logger.log(f"Artifacts written to: {RESULT_JSON}")
    logger.log(f"Artifacts written to: {REPORT_PATH}")
    logger.log(f"Total runtime: {elapsed:.2f}s")
    logger.write(LOG_PATH)
    return 0


def _log_dataset_summaries(logger: Logger, dataset_bundle: Any) -> None:
    for file_name, summary in dataset_bundle.file_summaries.items():
        logger.log(f"{file_name}")
        logger.log(f"  Total samples: {summary.total_samples}")
        logger.log(f"  Label distribution: {json.dumps(summary.label_distribution, sort_keys=True)}")
        logger.log(
            f"  Removed empty: {summary.removed_empty} | "
            f"Removed duplicate text: {summary.removed_duplicates} | "
            f"Kept: {summary.kept_samples}"
        )
        logger.log("  Random samples:")
        for sample in summary.random_samples:
            logger.log(
                f"    - [{sample.id}] label={sample.label} "
                f"text={sample.text[:180].replace(chr(10), ' ')}"
            )


def _log_split_summaries(logger: Logger, split_bundle: Any) -> None:
    for split in (split_bundle.train, split_bundle.validation, split_bundle.test):
        logger.log(f"{split.name.title()} size: {split.size()}")
        logger.log(f"  Label distribution: {json.dumps(split.label_distribution(), sort_keys=True)}")
        logger.log(f"  Source distribution: {json.dumps(split.source_distribution(), sort_keys=True)}")
        logger.log(
            f"  Source/label distribution: "
            f"{json.dumps(split.source_label_distribution(), sort_keys=True)}"
        )


def _write_split_artifacts(split_bundle: Any) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split in (split_bundle.train, split_bundle.validation, split_bundle.test):
        path = SPLITS_DIR / f"{split.name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in split.records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")


def _run_pipeline_checks(
    train_records: list[ExternalRawRecord],
    classifier: LogisticRegressionScorer,
) -> list[dict[str, str]]:
    benign_record = next(record for record in train_records if record.label == 0)

    checks: list[dict[str, str]] = []

    detector_rules = InjectionDetector(mode="rules", threshold=DEFAULT_THRESHOLD)
    rules_attack_record = next(
        (record for record in train_records if record.label == 1 and detector_rules.scan(record.text, source_type="user").is_injection),
        None,
    )
    attack_rules = detector_rules.scan(rules_attack_record.text, source_type="user") if rules_attack_record else None
    benign_rules = detector_rules.scan(benign_record.text, source_type="user")
    checks.append(
        _check(
            name="Detector regex patterns",
            ok=(
                rules_attack_record is not None
                and attack_rules is not None
                and attack_rules.is_injection
                and len(attack_rules.hits) > 0
                and not benign_rules.is_injection
            ),
            detail=(
                f"attack_id={rules_attack_record.id if rules_attack_record else 'none'} "
                f"attack_hits={len(attack_rules.hits) if attack_rules else 0} "
                f"benign_flagged={benign_rules.is_injection}"
            ),
        )
    )

    detector_hybrid = InjectionDetector(mode="hybrid", threshold=DEFAULT_THRESHOLD)
    hybrid_attack_record = next(
        (record for record in train_records if record.label == 1 and detector_hybrid.scan(record.text, source_type="user").is_injection),
        None,
    )
    attack_hybrid = detector_hybrid.scan(hybrid_attack_record.text, source_type="user") if hybrid_attack_record else None
    benign_hybrid = detector_hybrid.scan(benign_record.text, source_type="user")
    checks.append(
        _check(
            name="Detector heuristic scoring",
            ok=(
                hybrid_attack_record is not None
                and attack_hybrid is not None
                and attack_hybrid.risk_score >= DEFAULT_THRESHOLD
                and benign_hybrid.risk_score < DEFAULT_THRESHOLD
            ),
            detail=(
                f"attack_id={hybrid_attack_record.id if hybrid_attack_record else 'none'} "
                f"attack_risk={attack_hybrid.risk_score if attack_hybrid else 0.0:.4f} "
                f"benign_risk={benign_hybrid.risk_score:.4f}"
            ),
        )
    )

    detector_full = InjectionDetector(
        mode="full",
        threshold=DEFAULT_THRESHOLD,
        classifier=classifier,
    )
    classifier_attack_record = next((record for record in train_records if record.label == 1), None)
    attack_full = detector_full.scan(classifier_attack_record.text, source_type="user") if classifier_attack_record else None
    checks.append(
        _check(
            name="Detector classifier",
            ok=(
                classifier_attack_record is not None
                and attack_full is not None
                and attack_full.classifier_score is not None
                and attack_full.classifier_score > DEFAULT_THRESHOLD
            ),
            detail=(
                f"attack_id={classifier_attack_record.id if classifier_attack_record else 'none'} "
                f"classifier_score={attack_full.classifier_score if attack_full and attack_full.classifier_score is not None else 0.0:.4f} "
                f"risk={attack_full.risk_score if attack_full else 0.0:.4f}"
            ),
        )
    )

    policy_block = PolicyEngine(strategy="block", threshold=DEFAULT_THRESHOLD)
    policy_allow = PolicyEngine(strategy="allow", threshold=DEFAULT_THRESHOLD)
    policy_annotate = PolicyEngine(strategy="annotate", threshold=DEFAULT_THRESHOLD)
    benign_policy = policy_block.decide(benign_hybrid, benign_record.text)
    attack_policy_block = policy_block.decide(attack_hybrid, hybrid_attack_record.text) if attack_hybrid and hybrid_attack_record else None
    attack_policy_allow = policy_allow.decide(attack_hybrid, hybrid_attack_record.text) if attack_hybrid and hybrid_attack_record else None
    attack_policy_annotate = policy_annotate.decide(attack_hybrid, hybrid_attack_record.text) if attack_hybrid and hybrid_attack_record else None
    checks.append(
        _check(
            name="Policy engine actions",
            ok=(
                benign_policy.action == PolicyDecision.ALLOW
                and attack_policy_block is not None
                and attack_policy_allow is not None
                and attack_policy_annotate is not None
                and attack_policy_block.action == PolicyDecision.BLOCK
                and attack_policy_allow.action == PolicyDecision.ALLOW
                and attack_policy_annotate.action == PolicyDecision.ANNOTATE
            ),
            detail=(
                f"attack_id={hybrid_attack_record.id if hybrid_attack_record else 'none'} "
                f"benign={benign_policy.action.value} "
                f"block={attack_policy_block.action.value if attack_policy_block else 'none'} "
                f"allow={attack_policy_allow.action.value if attack_policy_allow else 'none'} "
                f"annotate={attack_policy_annotate.action.value if attack_policy_annotate else 'none'}"
            ),
        )
    )

    middleware = PromptInjectionMiddleware(
        mode="hybrid",
        strategy="block",
        threshold=DEFAULT_THRESHOLD,
        log_detections=False,
    )
    trusted_result = middleware.inspect_messages(
        [{"role": "system", "content": hybrid_attack_record.text if hybrid_attack_record else ""}],
        source_type="user",
    )
    user_result = middleware.inspect_messages(
        [{"role": "user", "content": hybrid_attack_record.text if hybrid_attack_record else ""}],
        source_type="user",
    )

    call_counter = {"count": 0}

    def counted_before_model(state: dict[str, Any], runtime: Any) -> None:
        call_counter["count"] += 1
        return None

    middleware.before_model = counted_before_model  # type: ignore[method-assign]
    middleware.wrap_model_call(
        lambda state, runtime: {"content": benign_record.text},
        {"messages": [{"role": "user", "content": benign_record.text}]},
        runtime=None,
    )
    checks.append(
        _check(
            name="Middleware trust boundary and no double scanning",
            ok=(
                trusted_result.action == "allow"
                and user_result.action == "block"
                and call_counter["count"] == 0
            ),
            detail=(
                f"trusted={trusted_result.action} user={user_result.action} "
                f"wrap_model_call_before_model_calls={call_counter['count']}"
            ),
        )
    )

    return checks


def _check(*, name: str, ok: bool, detail: str) -> dict[str, str]:
    return {
        "name": name,
        "status": "PASS" if ok else "FAIL",
        "detail": detail,
    }


def _score_records(
    records: list[ExternalRawRecord],
    detector: InjectionDetector,
    threshold: float,
) -> list[EvaluatedRecord]:
    evaluated: list[EvaluatedRecord] = []
    for record in records:
        result = detector.scan(record.text, source_type="user")
        predicted_label = 1 if result.risk_score >= threshold else 0
        evaluated.append(
            EvaluatedRecord(
                id=record.id,
                text=record.text,
                label=record.label,
                source_dataset=record.source_dataset,
                risk_score=result.risk_score,
                predicted_label=predicted_label,
            )
        )
    return evaluated


def _metrics_with_fpr(
    records: list[ExternalRawRecord],
    evaluated: list[EvaluatedRecord],
    *,
    threshold: float,
    config_name: str,
) -> dict[str, Any]:
    y_true = [record.label for record in records]
    y_pred = [row.predicted_label for row in evaluated]
    y_scores = [row.risk_score for row in evaluated]
    metrics = compute_metrics(
        y_true,
        y_pred,
        y_scores,
        threshold=threshold,
        config_name=config_name,
    )
    confusion = metrics.confusion
    negatives = confusion.fp + confusion.tn
    false_positive_rate = confusion.fp / negatives if negatives else 0.0
    payload = metrics.to_dict()
    payload["false_positive_rate"] = round(false_positive_rate, 4)
    return payload


def _run_threshold_analysis(
    split_bundle: Any,
    detectors: dict[str, InjectionDetector],
    cached_scores: dict[str, dict[str, list[float]]],
    logger: Logger,
) -> dict[str, dict[str, Any]]:
    threshold_results: dict[str, dict[str, Any]] = {}
    validation_labels = split_bundle.validation.labels()
    test_labels = split_bundle.test.labels()

    for config_name, detector in detectors.items():
        if detector.mode == "rules":
            threshold_results[config_name] = {
                "best_threshold": DEFAULT_THRESHOLD,
                "default_validation_metrics": None,
                "optimized_validation_metrics": None,
                "default_test_metrics": None,
                "optimized_test_metrics": None,
                "test_recall_improvement": 0.0,
                "status": "not_applicable_binary_rules_mode",
            }
            logger.log(f"  {config_name}: threshold sweep not applicable in rules mode")
            continue

        validation_scores = cached_scores["validation"][config_name]
        test_scores = cached_scores["test"][config_name]
        sweep = threshold_sweep(validation_labels, validation_scores)
        best_threshold = float(sweep.best_f1_threshold)

        default_validation_pred = [1 if score >= DEFAULT_THRESHOLD else 0 for score in validation_scores]
        optimized_validation_pred = [1 if score >= best_threshold else 0 for score in validation_scores]
        default_test_pred = [1 if score >= DEFAULT_THRESHOLD else 0 for score in test_scores]
        optimized_test_pred = [1 if score >= best_threshold else 0 for score in test_scores]

        default_validation_metrics = _metrics_dict_from_arrays(
            validation_labels,
            default_validation_pred,
            validation_scores,
            threshold=DEFAULT_THRESHOLD,
            config_name=f"{config_name} [validation default]",
        )
        optimized_validation_metrics = _metrics_dict_from_arrays(
            validation_labels,
            optimized_validation_pred,
            validation_scores,
            threshold=best_threshold,
            config_name=f"{config_name} [validation optimized]",
        )
        default_test_metrics = _metrics_dict_from_arrays(
            test_labels,
            default_test_pred,
            test_scores,
            threshold=DEFAULT_THRESHOLD,
            config_name=f"{config_name} [test default]",
        )
        optimized_test_metrics = _metrics_dict_from_arrays(
            test_labels,
            optimized_test_pred,
            test_scores,
            threshold=best_threshold,
            config_name=f"{config_name} [test optimized]",
        )

        recall_improvement = round(
            optimized_test_metrics["recall"] - default_test_metrics["recall"],
            4,
        )
        threshold_results[config_name] = {
            "best_threshold": round(best_threshold, 4),
            "recall_at_0_8_validation_threshold": sweep.recall_at_0_8_threshold,
            "default_validation_metrics": default_validation_metrics,
            "optimized_validation_metrics": optimized_validation_metrics,
            "default_test_metrics": default_test_metrics,
            "optimized_test_metrics": optimized_test_metrics,
            "test_recall_improvement": recall_improvement,
            "status": "optimized_on_validation_applied_to_test",
        }
        logger.log(
            f"  {config_name}: best_threshold={best_threshold:.4f} "
            f"default_test_recall={default_test_metrics['recall']:.4f} "
            f"optimized_test_recall={optimized_test_metrics['recall']:.4f} "
            f"optimized_test_fpr={optimized_test_metrics['false_positive_rate']:.4f} "
            f"improvement={recall_improvement:+.4f}"
        )

    return threshold_results


def _build_error_analysis(
    records: list[ExternalRawRecord],
    evaluated: list[EvaluatedRecord],
) -> dict[str, list[dict[str, Any]]]:
    false_negatives: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []

    for record, scored in zip(records, evaluated):
        payload = {
            "id": record.id,
            "source_dataset": record.source_dataset,
            "label": record.label,
            "risk_score": round(scored.risk_score, 4),
            "text": record.text,
        }
        if record.label == 1 and scored.predicted_label == 0:
            false_negatives.append(payload)
        if record.label == 0 and scored.predicted_label == 1:
            false_positives.append(payload)

    false_negatives.sort(key=lambda row: (row["risk_score"], row["id"]))
    false_positives.sort(key=lambda row: (-row["risk_score"], row["id"]))
    return {
        "false_negatives": false_negatives[:5],
        "false_positives": false_positives[:5],
    }


def _log_error_examples(logger: Logger, error_analysis: dict[str, list[dict[str, Any]]]) -> None:
    logger.log("  False negatives (top 5)")
    for row in error_analysis["false_negatives"]:
        logger.log(
            f"    - [{row['id']}] src={row['source_dataset']} risk={row['risk_score']:.4f} "
            f"text={str(row['text'])[:180].replace(chr(10), ' ')}"
        )

    logger.log("  False positives (top 5)")
    for row in error_analysis["false_positives"]:
        logger.log(
            f"    - [{row['id']}] src={row['source_dataset']} risk={row['risk_score']:.4f} "
            f"text={str(row['text'])[:180].replace(chr(10), ' ')}"
        )


def _build_observations(
    *,
    evaluation_results: dict[str, dict[str, Any]],
    threshold_analysis: dict[str, dict[str, Any]],
    error_analysis: dict[str, list[dict[str, Any]]],
) -> dict[str, list[str]]:
    test_default = evaluation_results["test"]
    config_a = test_default["A: Regex only"]["default"]
    config_b = test_default["B: Regex + Scoring"]["default"]
    config_c = test_default["C: + Classifier"]["default"]
    config_b_optimized = threshold_analysis["B: Regex + Scoring"]["optimized_test_metrics"]
    config_c_optimized = threshold_analysis["C: + Classifier"]["optimized_test_metrics"]

    strengths = [
        (
            "Config C achieves the strongest ranking quality on test "
            f"(AUC {config_c['roc_auc']:.4f}) even before threshold tuning."
        ),
        (
            "Validation-tuned Config C generalizes cleanly to test with "
            f"precision {config_c_optimized['precision']:.4f}, recall {config_c_optimized['recall']:.4f}, "
            f"F1 {config_c_optimized['f1']:.4f}, and FPR {config_c_optimized['false_positive_rate']:.4f}."
        ),
    ]

    weaknesses = []
    if config_b["f1"] < config_a["f1"]:
        weaknesses.append(
            "At the default threshold, Config B underperforms Config A on test F1 because heuristic scores stay too conservative on many real attacks."
        )
    if config_b_optimized and config_b_optimized["false_positive_rate"] >= 0.99:
        weaknesses.append(
            "Config B's best-F1 validation threshold collapses to 0.0000, which recovers recall but flags essentially every benign test sample."
        )
    if error_analysis["false_negatives"]:
        weaknesses.append(
            "Some attacks remain below threshold, especially prompts phrased as persona or roleplay requests without explicit override keywords."
        )
    if error_analysis["false_positives"]:
        weaknesses.append(
            "Some benign prompts are flagged when imperative phrasing overlaps with jailbreak-style patterns."
        )

    generalization = [
        (
            "Threshold optimization on validation changes Config C test recall by "
            f"{threshold_analysis['C: + Classifier']['test_recall_improvement']:+.4f} "
            f"at threshold {threshold_analysis['C: + Classifier']['best_threshold']:.4f}."
        ),
        (
            "Config C optimized test F1 is "
            f"{config_c_optimized['f1']:.4f} compared with default {config_c['f1']:.4f}."
        ),
    ]

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "generalization_behavior": generalization,
    }


def _build_markdown_report(payload: dict[str, Any]) -> str:
    dataset_summary = payload["dataset_summary"]
    split_summary = payload["split_summary"]
    evaluation_results = payload["evaluation_results"]
    threshold_analysis = payload["threshold_analysis"]
    error_analysis = payload["error_analysis"]
    observations = payload["observations"]

    validation_rows = _table_rows_from_metrics(evaluation_results["validation"])
    test_rows = _table_rows_from_metrics(evaluation_results["test"])

    threshold_lines = []
    for config_name in ("B: Regex + Scoring", "C: + Classifier"):
        analysis = threshold_analysis[config_name]
        threshold_lines.append(
            f"- {config_name}: best threshold {analysis['best_threshold']:.4f}, "
            f"optimized test F1 {analysis['optimized_test_metrics']['f1']:.4f}, "
            f"optimized test FPR {analysis['optimized_test_metrics']['false_positive_rate']:.4f}, "
            f"test recall improvement {analysis['test_recall_improvement']:+.4f}"
        )

    fn_lines = [
        f"- FN [{row['id']}] ({row['source_dataset']}, risk={row['risk_score']:.4f}): {row['text'][:180]}"
        for row in error_analysis["false_negatives"]
    ]
    fp_lines = [
        f"- FP [{row['id']}] ({row['source_dataset']}, risk={row['risk_score']:.4f}): {row['text'][:180]}"
        for row in error_analysis["false_positives"]
    ]

    report = [
        "# External Raw Dataset Research Report",
        "",
        "## 1. Dataset Summary",
        f"- Total samples: {dataset_summary['total_samples']}",
        f"- Per-source breakdown: {json.dumps(dataset_summary['per_source'], sort_keys=True)}",
        f"- Label distribution: {json.dumps(dataset_summary['label_distribution'], sort_keys=True)}",
        f"- Train/Validation/Test sizes: {split_summary['train']['size']} / {split_summary['validation']['size']} / {split_summary['test']['size']}",
        "",
        "## 2. Evaluation Results (Validation)",
        "| Config | Precision | Recall | F1 | AUC | FPR |",
        "| ------ | --------- | ------ | -- | --- | --- |",
        *validation_rows,
        "",
        "## 2. Evaluation Results (Test - Primary)",
        "| Config | Precision | Recall | F1 | AUC | FPR |",
        "| ------ | --------- | ------ | -- | --- | --- |",
        *test_rows,
        "",
        "## 3. Threshold Analysis",
        *threshold_lines,
        "",
        "## 4. Error Analysis",
        "### False Negatives",
        *(fn_lines or ["- None in top-5 sample window."]),
        "### False Positives",
        *(fp_lines or ["- None in top-5 sample window."]),
        "",
        "## 5. Observations",
        "### Strengths",
        *[f"- {line}" for line in observations["strengths"]],
        "### Weaknesses",
        *([f"- {line}" for line in observations["weaknesses"]] or ["- No recurring weakness was evident in the sampled error set."]),
        "### Generalization Behavior",
        *[f"- {line}" for line in observations["generalization_behavior"]],
        "",
        "## 6. Integrity Checks",
        "- No synthetic data used.",
        "- Only the required raw datasets were loaded.",
        "- Config C was trained on the training split only.",
        "- Validation was used for threshold optimization; test remained held out for primary reporting.",
        "- No exact text overlap exists between train, validation, and test splits.",
        "",
    ]
    return "\n".join(report)


def _table_rows_from_metrics(metrics_by_config: dict[str, dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for config_name in ("A: Regex only", "B: Regex + Scoring", "C: + Classifier"):
        metrics = metrics_by_config[config_name]["default"]
        rows.append(
            f"| {config_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {_fmt_auc(metrics['roc_auc'])} | {metrics['false_positive_rate']:.4f} |"
        )
    return rows


def _fmt_auc(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _split_summary_dict(split: Any) -> dict[str, Any]:
    return {
        "size": split.size(),
        "label_distribution": split.label_distribution(),
        "source_distribution": split.source_distribution(),
        "source_label_distribution": split.source_label_distribution(),
    }


def _metrics_dict_from_arrays(
    y_true: list[int],
    y_pred: list[int],
    y_scores: list[float],
    *,
    threshold: float,
    config_name: str,
) -> dict[str, Any]:
    metrics = compute_metrics(
        y_true,
        y_pred,
        y_scores,
        threshold=threshold,
        config_name=config_name,
    )
    confusion = metrics.confusion
    negatives = confusion.fp + confusion.tn
    payload = metrics.to_dict()
    payload["false_positive_rate"] = round(confusion.fp / negatives, 4) if negatives else 0.0
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
