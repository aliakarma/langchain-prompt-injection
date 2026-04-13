"""
policy.py
─────────
Policy engine that maps a ``DetectionResult`` to one of four decisions:

    allow     – pass the request through unchanged.
    annotate  – pass the request through but attach alert metadata to state.
    redact    – replace suspicious content with a placeholder and continue.
    block     – raise ``PromptInjectionError`` (or a subclass).

The engine also differentiates between moderate and high-risk events,
raising ``HighRiskInjectionError`` when the risk score exceeds the
``high_risk_threshold``.

Usage
-----
    from prompt_injection.policy import PolicyEngine, PolicyDecision

    engine   = PolicyEngine(strategy="block", threshold=0.50)
    decision = engine.decide(detection_result, message)

    if decision.action == PolicyDecision.BLOCK:
        raise decision.exception
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from prompt_injection.detector import DetectionResult
from prompt_injection.exceptions import (
    EvasionAttemptError,
    HighRiskInjectionError,
    PromptInjectionError,
    UntrustedSourceError,
)


# ---------------------------------------------------------------------------
# Enums and decision dataclass
# ---------------------------------------------------------------------------


class PolicyDecision(str, Enum):
    """Four possible policy outcomes."""

    ALLOW = "allow"
    ANNOTATE = "annotate"
    REDACT = "redact"
    BLOCK = "block"


@dataclass
class PolicyResult:
    """
    Output of ``PolicyEngine.decide()``.

    Attributes
    ----------
    action : PolicyDecision
        The chosen action.
    result : DetectionResult
        The detection result that triggered this decision.
    annotation : dict | None
        Metadata to attach to the agent state when action is ANNOTATE.
    redacted_text : str | None
        The sanitised text when action is REDACT.
    exception : PromptInjectionError | None
        The exception to raise when action is BLOCK.
    """

    action: PolicyDecision
    result: DetectionResult
    annotation: dict[str, Any] | None = None
    redacted_text: str | None = None
    exception: PromptInjectionError | None = None


# ---------------------------------------------------------------------------
# Redaction helper
# ---------------------------------------------------------------------------

_REDACT_PLACEHOLDER = "[CONTENT REDACTED BY INJECTION FILTER]"


def _redact(text: str, result: DetectionResult) -> str:
    """
    Replace matched spans with placeholder text.
    If no spans are recorded, replace the whole text.
    """
    if not result.hits or all(h.matched_span is None for h in result.hits):
        return _REDACT_PLACEHOLDER

    # Build non-overlapping span list sorted by start.
    spans = sorted(
        (h.matched_span for h in result.hits if h.matched_span),
        key=lambda s: s[0],
    )
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            parts.append(text[cursor:start])
        parts.append(_REDACT_PLACEHOLDER)
        cursor = max(cursor, end)
    parts.append(text[cursor:])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------


class PolicyEngine:
    """
    Maps detection results to policy actions.

    Parameters
    ----------
    strategy : str
        Default strategy: ``"allow"``, ``"annotate"``, ``"redact"``,
        or ``"block"``.
    threshold : float
        Minimum risk score to trigger any action.  Below this the engine
        always returns ALLOW.
    high_risk_threshold : float
        Above this score the engine raises ``HighRiskInjectionError``
        instead of the base ``PromptInjectionError``.
    untrusted_sources : list[str]
        Source types that trigger ``UntrustedSourceError`` on block.
    evasion_categories : list[str]
        Detection categories that trigger ``EvasionAttemptError`` on block.
    annotate_key : str
        State key used when injecting alert metadata (ANNOTATE action).
    """

    def __init__(
        self,
        strategy: str = "block",
        threshold: float = 0.50,
        high_risk_threshold: float = 0.85,
        untrusted_sources: list[str] | None = None,
        evasion_categories: list[str] | None = None,
        annotate_key: str = "prompt_injection_alerts",
    ) -> None:
        valid = {"allow", "annotate", "redact", "block"}
        if strategy not in valid:
            raise ValueError(f"strategy must be one of {valid!r}, got {strategy!r}.")

        self.strategy = strategy
        self.threshold = threshold
        self.high_risk_threshold = high_risk_threshold
        self.untrusted_sources = set(
            untrusted_sources or ["rag", "tool", "file", "web"]
        )
        self.evasion_categories = set(
            evasion_categories or ["evasion"]
        )
        self.annotate_key = annotate_key

    # ------------------------------------------------------------------
    # Main decision method
    # ------------------------------------------------------------------

    def decide(
        self,
        detection: DetectionResult,
        original_text: str | None = None,
        *,
        override_strategy: str | None = None,
    ) -> PolicyResult:
        """
        Apply the policy to a single ``DetectionResult``.

        Parameters
        ----------
        detection : DetectionResult
            Output from ``InjectionDetector.scan()``.
        original_text : str | None
            The original text (needed for REDACT action).
        override_strategy : str | None
            Temporarily override the engine's strategy for this call.
        """
        strategy = override_strategy or self.strategy

        # Below threshold → always allow.
        if not detection.is_injection or detection.risk_score < self.threshold:
            return PolicyResult(action=PolicyDecision.ALLOW, result=detection)

        action = PolicyDecision(strategy)

        if action == PolicyDecision.ALLOW:
            return PolicyResult(action=PolicyDecision.ALLOW, result=detection)

        if action == PolicyDecision.ANNOTATE:
            return self._make_annotate(detection)

        if action == PolicyDecision.REDACT:
            text = original_text or detection.text_preview
            redacted = _redact(text, detection)
            return PolicyResult(
                action=PolicyDecision.REDACT,
                result=detection,
                redacted_text=redacted,
            )

        # BLOCK
        exc = self._make_exception(detection)
        return PolicyResult(
            action=PolicyDecision.BLOCK,
            result=detection,
            exception=exc,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_annotate(self, detection: DetectionResult) -> PolicyResult:
        annotation = {
            "risk_score": detection.risk_score,
            "hit_categories": detection.hit_categories,
            "source_type": detection.source_type,
            "hit_count": len(detection.hits),
            "has_high_severity": detection.has_high_severity_hit,
            "classifier_score": detection.classifier_score,
            "latency_ms": detection.latency_ms,
        }
        return PolicyResult(
            action=PolicyDecision.ANNOTATE,
            result=detection,
            annotation={self.annotate_key: annotation},
        )

    def _make_exception(self, detection: DetectionResult) -> PromptInjectionError:
        hit_categories = set(detection.hit_categories)
        detections_payload = [
            {
                "source_type": detection.source_type,
                "hits": [
                    {
                        "pattern_id": h.pattern_id,
                        "category": h.category,
                        "severity": h.severity,
                    }
                    for h in detection.hits
                ],
                "risk_score": detection.risk_score,
                "content_preview": detection.text_preview,
            }
        ]
        common_kwargs = {
            "detections": detections_payload,
            "risk_score": detection.risk_score,
        }

        # Evasion detected → EvasionAttemptError
        if hit_categories & self.evasion_categories:
            return EvasionAttemptError(
                f"Evasion attempt detected (risk={detection.risk_score:.3f}).",
                **common_kwargs,
            )

        # Untrusted source → UntrustedSourceError
        if detection.source_type in self.untrusted_sources:
            return UntrustedSourceError(
                f"Injection in untrusted source '{detection.source_type}' "
                f"(risk={detection.risk_score:.3f}).",
                source_type=detection.source_type,
                **common_kwargs,
            )

        # High risk → HighRiskInjectionError
        if detection.risk_score >= self.high_risk_threshold:
            return HighRiskInjectionError(
                f"High-risk prompt injection (risk={detection.risk_score:.3f}).",
                **common_kwargs,
            )

        # Default block
        return PromptInjectionError(
            f"Prompt injection detected (risk={detection.risk_score:.3f}).",
            **common_kwargs,
        )

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def decide_many(
        self,
        detections: list[DetectionResult],
        original_texts: list[str] | None = None,
    ) -> list[PolicyResult]:
        """Apply policy to a list of ``DetectionResult`` objects."""
        texts = original_texts or [None] * len(detections)  # type: ignore[list-item]
        return [
            self.decide(det, txt)
            for det, txt in zip(detections, texts)
        ]

    def first_block(
        self, results: list["PolicyResult"]
    ) -> "PolicyResult | None":
        """Return the first BLOCK result in a list, or None."""
        return next(
            (r for r in results if r.action == PolicyDecision.BLOCK), None
        )
