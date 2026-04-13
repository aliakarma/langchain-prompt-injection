"""
detector.py
───────────
Three-configuration detection engine.

Configuration A — rules only
    Pattern scanner only.  Binary hit / no-hit.  < 1 ms / request.

Configuration B — rules + scoring
    Pattern scanner + keyword density heuristic + continuous risk score
    in [0.0, 1.0].  Tunable threshold.  < 2 ms / request.

Configuration C — rules + scoring + classifier
    A + B + a pluggable secondary scorer (logistic regression by default).
    Requires a fitted model.  3–8 ms / request.

Public API
----------
    from prompt_injection.detector import InjectionDetector, DetectionResult

    detector = InjectionDetector(mode="hybrid")     # B
    result   = detector.scan(text, source_type="user")
    if result.is_injection:
        print(result.risk_score, result.hits)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from prompt_injection.exceptions import ClassifierNotAvailableError, DetectorConfigurationError
from prompt_injection.patterns import (
    CATEGORY_WEIGHTS,
    PATTERN_REGISTRY,
    SEVERITY_WEIGHTS,
    PatternEntry,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class HitRecord:
    """A single pattern match within a scanned text."""

    pattern_id: str
    category: str
    severity: str
    description: str
    matched_span: tuple[int, int] | None = None


@dataclass
class DetectionResult:
    """
    Full output of a single ``InjectionDetector.scan()`` call.

    Attributes
    ----------
    text_preview : str
        First 200 characters of the scanned text (for logging / alerts).
    source_type : str
        Origin channel: ``"user"``, ``"tool"``, ``"rag"``, ``"file"``,
        ``"web"``, or ``"unknown"``.
    is_injection : bool
        True when ``risk_score >= threshold`` or any high-severity pattern
        fired and mode is ``"rules"``.
    risk_score : float
        Continuous risk score in [0.0, 1.0].  Always 0.0 or 1.0 in
        ``mode="rules"`` (binary).
    hits : list[HitRecord]
        All pattern matches that fired.
    hit_categories : list[str]
        Unique categories among the hits.
    classifier_score : float | None
        Raw secondary classifier probability, if available.
    latency_ms : float
        Wall-clock time taken for this scan in milliseconds.
    """

    text_preview: str
    source_type: str
    is_injection: bool
    risk_score: float
    hits: list[HitRecord]
    hit_categories: list[str]
    classifier_score: float | None = None
    latency_ms: float = 0.0

    @property
    def has_high_severity_hit(self) -> bool:
        return any(h.severity == "high" for h in self.hits)

    def to_dict(self) -> dict:
        return {
            "text_preview": self.text_preview,
            "source_type": self.source_type,
            "is_injection": self.is_injection,
            "risk_score": round(self.risk_score, 4),
            "hit_count": len(self.hits),
            "hit_categories": self.hit_categories,
            "has_high_severity": self.has_high_severity_hit,
            "classifier_score": (
                round(self.classifier_score, 4)
                if self.classifier_score is not None
                else None
            ),
            "latency_ms": round(self.latency_ms, 3),
        }


# ---------------------------------------------------------------------------
# Classifier protocol (pluggable secondary scorer)
# ---------------------------------------------------------------------------


@runtime_checkable
class ScorerProtocol(Protocol):
    """
    Any object implementing ``score(text) -> float`` can act as the
    secondary classifier in Configuration C.

    Built-in implementation: ``LogisticRegressionScorer`` (see below).
    External examples: a fine-tuned transformer, a remote API classifier.
    """

    def score(self, text: str) -> float:
        """Return a probability in [0.0, 1.0] that *text* is an injection."""
        ...


# ---------------------------------------------------------------------------
# Built-in heuristic scorer (Config B layer)
# ---------------------------------------------------------------------------

# Supplementary keyword sets that boost risk score without matching
# the full structural patterns above.
_BOOST_KEYWORDS: list[tuple[float, re.Pattern[str]]] = [
    # (weight, pattern)
    (0.10, re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE)),
    (0.10, re.compile(r"\binstructions?\b.*\bignore\b", re.IGNORECASE | re.DOTALL)),
    (0.08, re.compile(r"\bjailbreak\b", re.IGNORECASE)),
    (0.08, re.compile(r"\buncensored\b", re.IGNORECASE)),
    (0.06, re.compile(r"\bbypass\b", re.IGNORECASE)),
    (0.06, re.compile(r"\boverride\b", re.IGNORECASE)),
    (0.05, re.compile(r"\brestrictions?\b", re.IGNORECASE)),
    (0.05, re.compile(r"\bunrestricted\b", re.IGNORECASE)),
    (0.04, re.compile(r"\bhidden\b", re.IGNORECASE)),
    (0.04, re.compile(r"\bconfidential\b", re.IGNORECASE)),
    (0.03, re.compile(r"\bsudo\b", re.IGNORECASE)),
    (0.03, re.compile(r"\broot\s+access\b", re.IGNORECASE)),
]

# Penalty if text contains evasion signals beyond the pattern registry.
_EVASION_SIGNALS: list[re.Pattern[str]] = [
    re.compile(r"[^\x00-\x7F]{3,}"),          # unusual Unicode runs
    re.compile(r"(\w)\1{4,}"),                 # excessive character repetition
    re.compile(r"(\S\s){5,}"),                 # excessive space splitting (l e t t e r)
]


def _heuristic_score(text: str, hits: list[HitRecord]) -> float:
    """
    Compute a continuous risk score in [0.0, 1.0] given pattern hits
    and supplementary keyword density.

    Score components
    ----------------
    1. Pattern hit contribution – weighted sum of severity × category weight.
    2. Keyword density boost – additive from _BOOST_KEYWORDS.
    3. Evasion signal penalty – additive when evasion indicators present.
    4. Multi-category bonus – increases score when hits span 2+ categories.
    """
    if not hits:
        # Only keyword density matters when no patterns fired.
        base = 0.0
    else:
        hit_score = 0.0
        for h in hits:
            sev_w = SEVERITY_WEIGHTS.get(h.severity, 0.5)
            cat_w = CATEGORY_WEIGHTS.get(h.category, 0.5)
            hit_score += sev_w * cat_w
        # Normalise: one high-severity/high-weight hit ≈ 0.90.
        base = min(hit_score / max(len(hits), 1) + 0.10 * min(len(hits), 5), 0.95)

    # Evasion override: active evasion is itself a high-confidence signal.
    # Any evasion-category hit guarantees the score clears the default 0.50 threshold.
    evasion_hits = [h for h in hits if h.category == "evasion"]
    if evasion_hits:
        base = max(base, 0.60)

    # Keyword density boost
    boost = 0.0
    for weight, kw_pattern in _BOOST_KEYWORDS:
        if kw_pattern.search(text):
            boost += weight
    boost = min(boost, 0.30)

    # Evasion signal
    evasion = 0.0
    for sig in _EVASION_SIGNALS:
        if sig.search(text):
            evasion += 0.05
    evasion = min(evasion, 0.15)

    # Multi-category bonus
    categories = {h.category for h in hits}
    multi_bonus = 0.05 * max(0, len(categories) - 1)

    raw = base + boost + evasion + multi_bonus
    return round(min(raw, 1.0), 4)


# ---------------------------------------------------------------------------
# Built-in logistic regression scorer (Config C, no external dependencies)
# ---------------------------------------------------------------------------


class LogisticRegressionScorer:
    """
    Lightweight TF-IDF + logistic regression classifier.

    Call ``fit(texts, labels)`` before use, or load a pre-fitted
    instance.  Implements ``ScorerProtocol``.

    This is the default secondary scorer for Configuration C.
    For production, replace with a fine-tuned transformer or API call
    by passing any object that satisfies ``ScorerProtocol``.
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._fitted = False

    def fit(self, texts: list[str], labels: list[int]) -> "LogisticRegressionScorer":
        """
        Fit the classifier on labelled (text, label) pairs.

        Parameters
        ----------
        texts : list[str]
            Training texts.
        labels : list[int]
            Binary labels (1 = injection, 0 = benign).
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
        except ImportError as exc:  # pragma: no cover
            raise DetectorConfigurationError(
                "scikit-learn is required for Configuration C. "
                "Install it with: pip install scikit-learn"
            ) from exc

        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=8000,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])
        self._pipeline.fit(texts, labels)
        self._fitted = True
        return self

    def score(self, text: str) -> float:
        """Return injection probability for *text*."""
        if not self._fitted or self._pipeline is None:
            raise ClassifierNotAvailableError(
                "LogisticRegressionScorer has not been fitted. "
                "Call .fit(texts, labels) first."
            )
        prob: float = self._pipeline.predict_proba([text])[0][1]
        return float(prob)

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------


class InjectionDetector:
    """
    Three-configuration prompt-injection detector.

    Parameters
    ----------
    mode : str
        Detection configuration:

        * ``"rules"``  — Configuration A: regex only, binary output.
        * ``"hybrid"`` — Configuration B: regex + heuristic scoring.
        * ``"full"``   — Configuration C: regex + scoring + classifier.

    threshold : float
        Risk score above which ``is_injection`` is set to True.
        Applies in ``"hybrid"`` and ``"full"`` modes.
        In ``"rules"`` mode, any pattern hit triggers ``is_injection``.

    classifier : ScorerProtocol | None
        Secondary scorer for ``"full"`` mode.  Defaults to
        ``LogisticRegressionScorer`` (must be fitted before use).

    classifier_weight : float
        Weight given to classifier score when blending with heuristic
        score in ``"full"`` mode.  Range [0.0, 1.0].

    trusted_roles : list[str]
        Message roles whose content is *not* scanned.  Defaults to
        ``["system", "developer"]``.

    scan_categories : list[str] | None
        Restrict scanning to specific attack categories.
        ``None`` means scan all categories.
    """

    def __init__(
        self,
        mode: str = "hybrid",
        threshold: float = 0.50,
        classifier: ScorerProtocol | None = None,
        classifier_weight: float = 0.40,
        trusted_roles: list[str] | None = None,
        scan_categories: list[str] | None = None,
    ) -> None:
        valid_modes = {"rules", "hybrid", "full"}
        if mode not in valid_modes:
            raise DetectorConfigurationError(
                f"mode must be one of {valid_modes!r}, got {mode!r}."
            )
        self.mode = mode
        self.threshold = threshold
        self.classifier_weight = classifier_weight
        self.trusted_roles = set(trusted_roles or ["system", "developer"])
        self.scan_categories = set(scan_categories) if scan_categories else None

        # Build active pattern list (pre-filter by category if requested).
        self._patterns: list[PatternEntry] = [
            e for e in PATTERN_REGISTRY
            if (self.scan_categories is None or e["category"] in self.scan_categories)
        ]

        # Classifier setup
        if mode == "full":
            self._classifier: ScorerProtocol | None = (
                classifier if classifier is not None else LogisticRegressionScorer()
            )
        else:
            self._classifier = classifier  # stored but not used

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def scan(self, text: str, source_type: str = "unknown") -> DetectionResult:
        """
        Scan a single text string and return a ``DetectionResult``.

        Parameters
        ----------
        text : str
            The text to scan (user message, retrieved chunk, tool output…).
        source_type : str
            Origin channel for logging and exception enrichment.
        """
        t0 = time.perf_counter()

        hits = self._run_patterns(text)
        categories = sorted({h.category for h in hits})

        if self.mode == "rules":
            risk_score = 1.0 if hits else 0.0
            is_injection = bool(hits)
            clf_score = None
        else:
            heuristic = _heuristic_score(text, hits)
            clf_score = None

            if self.mode == "full":
                clf_score = self._run_classifier(text)
                if clf_score is not None:
                    risk_score = (
                        (1 - self.classifier_weight) * heuristic
                        + self.classifier_weight * clf_score
                    )
                else:
                    risk_score = heuristic
            else:
                risk_score = heuristic

            risk_score = round(min(risk_score, 1.0), 4)
            is_injection = risk_score >= self.threshold

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return DetectionResult(
            text_preview=text[:200],
            source_type=source_type,
            is_injection=is_injection,
            risk_score=risk_score,
            hits=hits,
            hit_categories=categories,
            classifier_score=clf_score,
            latency_ms=round(latency_ms, 3),
        )

    def scan_messages(
        self,
        messages: list[dict],
        *,
        content_key: str = "content",
        role_key: str = "role",
        source_type: str = "user",
    ) -> list[tuple[dict, DetectionResult]]:
        """
        Scan a list of LangChain-style message dicts.

        Returns a list of ``(message, result)`` tuples for every
        *untrusted* message (trusted roles are skipped).
        """
        results = []
        for msg in messages:
            role = msg.get(role_key, "")
            if role in self.trusted_roles:
                continue
            content = msg.get(content_key, "")
            if not isinstance(content, str) or not content.strip():
                continue
            result = self.scan(content, source_type=source_type)
            results.append((msg, result))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_patterns(self, text: str) -> list[HitRecord]:
        hits: list[HitRecord] = []
        for entry in self._patterns:
            match = entry["pattern"].search(text)
            if match:
                hits.append(HitRecord(
                    pattern_id=entry["id"],
                    category=entry["category"],
                    severity=entry["severity"],
                    description=entry["description"],
                    matched_span=(match.start(), match.end()),
                ))
        return hits

    def _run_classifier(self, text: str) -> float | None:
        if self._classifier is None:
            return None
        try:
            return float(self._classifier.score(text))
        except ClassifierNotAvailableError:
            return None
        except Exception:  # pragma: no cover
            return None
