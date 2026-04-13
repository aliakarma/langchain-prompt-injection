"""
middleware.py
─────────────
LangChain AgentMiddleware implementation that plugs the detector and
policy engine into all four available hooks:

    before_model      – scan assembled messages before the LLM sees them.
    wrap_model_call   – inspect full request / response envelope.
    after_model       – validate LLM output for exfiltration attempts.
    wrap_tool_call    – scan tool inputs and tool outputs.

The middleware respects the trust boundary defined in THREAT_MODEL.md:
  * TRUSTED  – "system" and "developer" roles are never scanned.
  * UNTRUSTED – user messages, tool outputs, RAG content, file I/O.

Usage
-----
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from prompt_injection.middleware import PromptInjectionMiddleware

    agent = create_agent(
        model=ChatOpenAI(model="gpt-4o-mini"),
        tools=[],
        middleware=[
            PromptInjectionMiddleware(
                mode="hybrid",
                strategy="block",
                threshold=0.50,
            )
        ],
    )

Standalone use (no LangChain agent)
------------------------------------
    from prompt_injection.middleware import PromptInjectionMiddleware

    guard = PromptInjectionMiddleware(mode="hybrid", strategy="annotate")

    # Inspect a list of message dicts directly.
    result = guard.inspect_messages(messages, source_type="user")
    if result.action == "block":
        raise result.exception
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from prompt_injection.detector import DetectionResult, InjectionDetector, ScorerProtocol
from prompt_injection.exceptions import (
    ClassifierNotAvailableError,
    DetectorConfigurationError,
    PromptInjectionError,
)
from prompt_injection.policy import PolicyDecision, PolicyEngine, PolicyResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangChain base-class import (graceful degradation when not installed)
# ---------------------------------------------------------------------------

try:
    from langchain.agents.middleware import AgentMiddleware  # type: ignore[import]

    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover
    # When running outside a LangChain environment (e.g. in notebooks or
    # unit tests that do not install LangChain) we provide a no-op base so
    # the rest of the module still imports cleanly.
    class AgentMiddleware:  # type: ignore[no-redef]
        """Fallback stub when langchain is not installed."""

        def before_model(self, state, runtime):  # pragma: no cover
            return None

        def after_model(self, state, runtime):  # pragma: no cover
            return None

        def wrap_model_call(self, call, state, runtime):  # pragma: no cover
            return call(state, runtime)

        def wrap_tool_call(self, call, tool_input, state, runtime):  # pragma: no cover
            return call(tool_input, state, runtime)

    _LANGCHAIN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Standalone inspection result (used without LangChain)
# ---------------------------------------------------------------------------


@dataclass
class InspectionResult:
    """
    Returned by ``PromptInjectionMiddleware.inspect_messages()`` for
    use outside a LangChain agent context.
    """

    action: str                           # "allow" | "annotate" | "redact" | "block"
    detections: list[DetectionResult]
    policy_results: list[PolicyResult]
    state_patch: dict[str, Any]           # merged annotation updates (ANNOTATE action)
    exception: PromptInjectionError | None


# ---------------------------------------------------------------------------
# Main middleware class
# ---------------------------------------------------------------------------


class PromptInjectionMiddleware(AgentMiddleware):
    """
    LangChain prompt-injection detection middleware.

    Parameters
    ----------
    mode : str
        Detector mode: ``"rules"`` | ``"hybrid"`` | ``"full"``.
    strategy : str
        Policy strategy: ``"allow"`` | ``"annotate"`` | ``"redact"`` |
        ``"block"``.
    threshold : float
        Risk score threshold for flagging (default 0.50).
    high_risk_threshold : float
        Above this score ``HighRiskInjectionError`` is raised (default 0.85).
    classifier : ScorerProtocol | None
        Secondary scorer for ``mode="full"``.  Defaults to the built-in
        ``LogisticRegressionScorer`` (must be fitted before use).
    classifier_weight : float
        Blending weight for the classifier score (0.0–1.0).
    trusted_roles : list[str]
        Message roles treated as trusted (default: ["system", "developer"]).
    scan_categories : list[str] | None
        Restrict scanning to specific attack categories.
    scan_tool_outputs : bool
        Whether to scan text returned by tools (default: True).
    scan_after_model : bool
        Whether to scan the LLM's own output for exfiltration (default: True).
    log_detections : bool
        Whether to log detection events at WARNING level (default: True).
    annotate_key : str
        State key used to inject alert metadata in ANNOTATE mode.
    """

    def __init__(
        self,
        *,
        mode: str = "hybrid",
        strategy: str = "block",
        threshold: float = 0.50,
        high_risk_threshold: float = 0.85,
        classifier: ScorerProtocol | None = None,
        classifier_weight: float = 0.40,
        trusted_roles: list[str] | None = None,
        scan_categories: list[str] | None = None,
        scan_tool_outputs: bool = True,
        scan_after_model: bool = True,
        log_detections: bool = True,
        annotate_key: str = "prompt_injection_alerts",
    ) -> None:
        super().__init__()

        self._detector = InjectionDetector(
            mode=mode,
            threshold=threshold,
            classifier=classifier,
            classifier_weight=classifier_weight,
            trusted_roles=trusted_roles,
            scan_categories=scan_categories,
        )
        self._policy = PolicyEngine(
            strategy=strategy,
            threshold=threshold,
            high_risk_threshold=high_risk_threshold,
            annotate_key=annotate_key,
        )
        self.scan_tool_outputs = scan_tool_outputs
        self.scan_after_model = scan_after_model
        self.log_detections = log_detections
        self.annotate_key = annotate_key
        self.mode = mode
        self.strategy = strategy

    # ------------------------------------------------------------------
    # LangChain hooks
    # ------------------------------------------------------------------

    def before_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """
        Scan all messages in ``state["messages"]`` before the LLM call.
        Trusted roles (system, developer) are skipped.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        patch: dict[str, Any] = {}
        for msg in messages:
            role = getattr(msg, "type", None) or (
                msg.get("role", "") if isinstance(msg, dict) else ""
            )
            # Skip trusted roles — never scan system / developer messages.
            if role in self._detector.trusted_roles:
                continue
            content = getattr(msg, "content", None) or (
                msg.get("content", "") if isinstance(msg, dict) else ""
            )
            if not isinstance(content, str) or not content.strip():
                continue

            detection = self._detector.scan(content, source_type="user")
            if not detection.is_injection:
                continue

            self._log(detection, hook="before_model")
            policy = self._policy.decide(detection, content)

            if policy.action == PolicyDecision.BLOCK:
                raise policy.exception

            if policy.action == PolicyDecision.ANNOTATE and policy.annotation:
                existing = patch.get(self.annotate_key, [])
                existing.append(policy.annotation[self.annotate_key])
                patch[self.annotate_key] = existing

        return patch or None

    def wrap_model_call(self, call: Any, state: dict[str, Any], runtime: Any) -> Any:
        """
        Wrap the actual LLM API call.  Runs before_model scan again on
        the final assembled state (in case state was mutated between
        hooks), then delegates to *call*, then runs after_model scan.
        """
        # Pre-call scan (belt-and-suspenders on the final state).
        self.before_model(state, runtime)

        response = call(state, runtime)

        # Post-call output scan.
        if self.scan_after_model:
            self._scan_model_output(response)

        return response

    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """
        Scan the most recent assistant message for exfiltration attempts
        (e.g. the model repeating a system prompt it was tricked into revealing).
        """
        if not self.scan_after_model:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        # Find the last assistant/ai message.
        last_ai = None
        for msg in reversed(messages):
            role = getattr(msg, "type", None) or (
                msg.get("role", "") if isinstance(msg, dict) else ""
            )
            if role in {"ai", "assistant"}:
                last_ai = msg
                break

        if last_ai is None:
            return None

        content = getattr(last_ai, "content", None) or (
            last_ai.get("content", "") if isinstance(last_ai, dict) else ""
        )
        if not isinstance(content, str) or not content.strip():
            return None

        detection = self._detector.scan(content, source_type="model_output")
        if not detection.is_injection:
            return None

        self._log(detection, hook="after_model")
        policy = self._policy.decide(detection, content)

        if policy.action == PolicyDecision.BLOCK:
            raise policy.exception

        if policy.action == PolicyDecision.ANNOTATE and policy.annotation:
            return policy.annotation

        return None

    def wrap_tool_call(
        self,
        call: Any,
        tool_input: Any,
        state: dict[str, Any],
        runtime: Any,
    ) -> Any:
        """
        Scan tool input and tool output for injected content.

        Tool outputs are the primary RAG / indirect injection vector.
        """
        # Scan tool input.
        if isinstance(tool_input, str) and tool_input.strip():
            det = self._detector.scan(tool_input, source_type="tool_input")
            if det.is_injection:
                self._log(det, hook="wrap_tool_call[input]")
                policy = self._policy.decide(det, tool_input)
                if policy.action == PolicyDecision.BLOCK:
                    raise policy.exception

        # Execute the tool.
        tool_output = call(tool_input, state, runtime)

        # Scan tool output (primary indirect injection vector).
        if self.scan_tool_outputs:
            output_text = self._extract_text(tool_output)
            if output_text:
                det = self._detector.scan(output_text, source_type="tool")
                if det.is_injection:
                    self._log(det, hook="wrap_tool_call[output]")
                    policy = self._policy.decide(det, output_text)
                    if policy.action == PolicyDecision.BLOCK:
                        raise policy.exception

        return tool_output

    # ------------------------------------------------------------------
    # Standalone inspection (no LangChain required)
    # ------------------------------------------------------------------

    def inspect_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        source_type: str = "user",
    ) -> InspectionResult:
        """
        Inspect a list of message dicts outside a LangChain agent.

        Returns an ``InspectionResult`` rather than raising directly,
        giving the caller control over the exception.

        Example::

            result = guard.inspect_messages(messages)
            if result.action == "block" and result.exception:
                raise result.exception
        """
        detections: list[DetectionResult] = []
        policy_results: list[PolicyResult] = []
        state_patch: dict[str, Any] = {}
        first_exception: PromptInjectionError | None = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in self._detector.trusted_roles:
                continue
            if not isinstance(content, str) or not content.strip():
                continue

            det = self._detector.scan(content, source_type=source_type)
            detections.append(det)
            pol = self._policy.decide(det, content)
            policy_results.append(pol)

            if pol.action == PolicyDecision.ANNOTATE and pol.annotation:
                for k, v in pol.annotation.items():
                    existing = state_patch.get(k, [])
                    existing.append(v)
                    state_patch[k] = existing

            if pol.action == PolicyDecision.BLOCK and first_exception is None:
                first_exception = pol.exception

        # Determine aggregate action.
        if first_exception is not None:
            agg_action = "block"
        elif any(p.action == PolicyDecision.REDACT for p in policy_results):
            agg_action = "redact"
        elif any(p.action == PolicyDecision.ANNOTATE for p in policy_results):
            agg_action = "annotate"
        else:
            agg_action = "allow"

        return InspectionResult(
            action=agg_action,
            detections=detections,
            policy_results=policy_results,
            state_patch=state_patch,
            exception=first_exception,
        )

    def inspect_text(
        self,
        text: str,
        *,
        source_type: str = "unknown",
    ) -> PolicyResult:
        """
        Inspect a single text string and return the ``PolicyResult``.

        Convenience method for RAG chunk scanning, tool output scanning,
        and unit testing.
        """
        detection = self._detector.scan(text, source_type=source_type)
        return self._policy.decide(detection, text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_model_output(self, response: Any) -> None:
        """Extract text from a raw LLM response and scan it."""
        text = self._extract_text(response)
        if not text:
            return
        det = self._detector.scan(text, source_type="model_output")
        if det.is_injection:
            self._log(det, hook="model_output_scan")
            policy = self._policy.decide(det, text)
            if policy.action == PolicyDecision.BLOCK:
                raise policy.exception

    @staticmethod
    def _extract_text(obj: Any) -> str:
        """Best-effort text extraction from various response shapes."""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            for key in ("content", "text", "output", "result", "message"):
                val = obj.get(key)
                if isinstance(val, str):
                    return val
        if hasattr(obj, "content") and isinstance(obj.content, str):
            return obj.content
        if hasattr(obj, "text") and isinstance(obj.text, str):
            return obj.text
        return ""

    def _log(self, detection: DetectionResult, hook: str) -> None:
        if not self.log_detections:
            return
        logger.warning(
            "[PromptInjectionMiddleware] hook=%s source=%s risk=%.3f "
            "categories=%s hits=%d",
            hook,
            detection.source_type,
            detection.risk_score,
            detection.hit_categories,
            len(detection.hits),
        )
