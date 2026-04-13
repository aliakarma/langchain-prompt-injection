"""
exceptions.py
─────────────
Custom exception hierarchy for the prompt-injection detection package.

All public exceptions inherit from PromptInjectionError so callers can
catch the entire family with a single ``except PromptInjectionError``.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class PromptInjectionError(ValueError):
    """
    Raised when the policy engine decides to *block* a request due to
    detected prompt-injection content.

    Attributes
    ----------
    detections : list[dict]
        Structured list of detection hits that triggered the block.
        Each entry contains ``role``, ``hits``, ``risk_score``, and
        ``content_preview`` keys.
    risk_score : float
        Aggregated risk score across all flagged messages (0.0 – 1.0).
    """

    def __init__(
        self,
        message: str = "Prompt injection detected.",
        *,
        detections: list[dict[str, Any]] | None = None,
        risk_score: float = 1.0,
    ) -> None:
        super().__init__(message)
        self.detections: list[dict[str, Any]] = detections or []
        self.risk_score: float = risk_score

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"risk_score={self.risk_score:.3f}, "
            f"detections={len(self.detections)})"
        )


# ---------------------------------------------------------------------------
# Specialisations
# ---------------------------------------------------------------------------


class HighRiskInjectionError(PromptInjectionError):
    """
    Raised when the risk score exceeds the *high-risk* threshold
    (default ≥ 0.85).  Allows callers to distinguish severity levels::

        try:
            agent.invoke(...)
        except HighRiskInjectionError as exc:
            # Critical – alert security team
            ...
        except PromptInjectionError as exc:
            # Moderate – log and return safe error to user
            ...
    """


class EvasionAttemptError(PromptInjectionError):
    """
    Raised when the hybrid scorer detects indicators of *active evasion*
    (e.g. homoglyph substitution, excessive whitespace splitting, or
    base64-encoded instructions) in addition to pattern matches.
    """


class UntrustedSourceError(PromptInjectionError):
    """
    Raised when injection is detected specifically inside content that
    arrived through an *untrusted* channel (RAG retrieval, tool output,
    uploaded file) rather than the direct user turn.

    Carries ``source_type`` so the caller can identify the poisoned
    retrieval channel.

    Attributes
    ----------
    source_type : str
        One of ``"rag"``, ``"tool"``, ``"file"``, ``"web"``.
    """

    def __init__(
        self,
        message: str = "Injection detected in untrusted source.",
        *,
        detections: list[dict[str, Any]] | None = None,
        risk_score: float = 1.0,
        source_type: str = "unknown",
    ) -> None:
        super().__init__(message, detections=detections, risk_score=risk_score)
        self.source_type: str = source_type

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"source_type={self.source_type!r}, "
            f"risk_score={self.risk_score:.3f})"
        )


# ---------------------------------------------------------------------------
# Configuration / setup errors (not detection errors)
# ---------------------------------------------------------------------------


class DetectorConfigurationError(RuntimeError):
    """Raised when the detector or middleware is misconfigured at setup time."""


class ClassifierNotAvailableError(DetectorConfigurationError):
    """
    Raised when ``mode='full'`` is requested but no classifier model
    has been fitted or provided.
    """
