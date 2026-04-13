"""
prompt_injection
────────────────
Prompt-injection detection middleware for LangChain.

Quick start
-----------
    from prompt_injection import PromptInjectionMiddleware

    # Drop into any create_agent(...) call:
    agent = create_agent(
        model=model,
        tools=tools,
        middleware=[PromptInjectionMiddleware(mode="hybrid", strategy="block")],
    )

    # Or use standalone:
    guard  = PromptInjectionMiddleware(mode="full")
    result = guard.inspect_text(user_input, source_type="user")

Public API
----------
Core:
    PromptInjectionMiddleware   Main middleware class (all four hooks).
    InjectionDetector           Low-level scanner (rules / hybrid / full).
    PolicyEngine                Maps DetectionResult → policy action.
    DetectionResult             Output of a single scan call.
    PolicyResult                Output of a policy decision.

Exceptions:
    PromptInjectionError        Base block exception.
    HighRiskInjectionError      Raised when risk score ≥ high_risk_threshold.
    EvasionAttemptError         Raised when evasion patterns detected.
    UntrustedSourceError        Raised when injection in RAG/tool content.

Evaluation:
    SyntheticDataset            Labelled injection dataset loader/generator.
    MetricsReport               Precision/recall/F1/AUC output dataclass.
    BenchmarkRunner             Three-config ablation benchmark.
    PerformanceProfiler         Latency / throughput measurement.
"""

from prompt_injection.middleware import PromptInjectionMiddleware, InspectionResult
from prompt_injection.detector import InjectionDetector, DetectionResult, LogisticRegressionScorer
from prompt_injection.policy import PolicyEngine, PolicyResult, PolicyDecision
from prompt_injection.exceptions import (
    PromptInjectionError,
    HighRiskInjectionError,
    EvasionAttemptError,
    UntrustedSourceError,
    DetectorConfigurationError,
    ClassifierNotAvailableError,
)

__version__ = "0.1.0"
__author__ = "Ali Akarma"
__all__ = [
    # Core
    "PromptInjectionMiddleware",
    "InspectionResult",
    "InjectionDetector",
    "DetectionResult",
    "LogisticRegressionScorer",
    "PolicyEngine",
    "PolicyResult",
    "PolicyDecision",
    # Exceptions
    "PromptInjectionError",
    "HighRiskInjectionError",
    "EvasionAttemptError",
    "UntrustedSourceError",
    "DetectorConfigurationError",
    "ClassifierNotAvailableError",
]
