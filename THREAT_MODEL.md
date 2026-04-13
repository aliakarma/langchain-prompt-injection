# Threat Model

**Package:** `langchain-prompt-injection`  
**Version:** 0.1.0  
**Date:** 2026-04-13

---

## 1. Overview

This document defines the threat model for the prompt-injection detection
middleware.  It specifies the adversary's assumed capabilities, the attack
surface by entry point, the trust boundaries enforced by the middleware,
and the residual risks that are out of scope.

---

## 2. Adversary Capabilities

| Level | Description | In Scope |
|---|---|---|
| **Black-box** | Adversary supplies only user-turn text. Has no access to the system prompt, model weights, or pattern list. | ✅ Primary target |
| **Grey-box** | Adversary controls content in retrieved documents (RAG), uploaded files, tool API responses, or web-fetched content. | ✅ Primary target |
| **White-box** | Adversary has full knowledge of the regex pattern list and threshold values, and crafts evasion strings accordingly. | ⚠ Partially addressed (see §6) |
| **Insider** | Adversary has write access to the system prompt or developer instructions. | ❌ Out of scope |
| **Model-weight** | Adversary has modified or fine-tuned the underlying LLM. | ❌ Out of scope |

---

## 3. Injection Entry Points and Trust Boundaries

```
┌─────────────────────────────────────────────────────────────┐
│                     TRUST BOUNDARY                          │
│                                                             │
│   TRUSTED (never scanned)      UNTRUSTED (always scanned)  │
│   ────────────────────────     ───────────────────────────  │
│   • system prompt              • user chat input            │
│   • developer messages         • retrieved RAG chunks       │
│   • hard-coded tool schemas    • web-fetched content        │
│                                • uploaded files (OCR text)  │
│                                • tool / API responses       │
│                                • email / calendar content   │
└─────────────────────────────────────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │  PromptInjectionMiddleware  │
              │  before_model()       │  ← user messages
              │  wrap_tool_call()     │  ← tool inputs + outputs
              │  after_model()        │  ← LLM output exfiltration
              │  wrap_model_call()    │  ← full request envelope
              └───────────┬───────────┘
                          │
                   LLM API call
```

### Trust-Level Assignment (configurable)

The `trusted_roles` parameter on `PromptInjectionMiddleware` defines which
message roles are excluded from scanning.  Default: `["system", "developer"]`.

**Never lower trust on content that arrived through an external channel.**
Attackers frequently attempt to inject content into RAG documents or tool
outputs that impersonates a trusted role (e.g. `[SYSTEM]: …`).  The middleware
detects role-impersonation attempts via the `indirect_injection` pattern category
(patterns IND-001 through IND-005).

---

## 4. Attack Surface by Hook

| Hook | Entry Points Covered | Attack Categories Targeted |
|---|---|---|
| `before_model` | User messages assembled into the prompt | All 8 categories |
| `wrap_tool_call` | Tool inputs; tool / retrieval outputs | `indirect_injection`, `instruction_override` |
| `after_model` | LLM's own output (exfiltration defence) | `system_prompt_extraction`, `boundary_escape` |
| `wrap_model_call` | Full request/response envelope | All categories (belt-and-suspenders) |

---

## 5. Detection Configurations and Their Security Properties

| Config | Mode | Adversary Resistance | Latency |
|---|---|---|---|
| **A** | Regex only | Vulnerable to white-box evasion; strong against unsophisticated attacks | < 1 ms |
| **B** | Regex + Scoring | Tolerates moderate obfuscation; continuous score enables threshold tuning | < 2 ms |
| **C** | Regex + Scoring + Classifier | Best against novel phrasing; vulnerable to adversarial ML attacks on the classifier | 3–8 ms |

---

## 6. Residual Risks and Known Limitations

### 6.1 White-box Pattern Evasion
An adversary who inspects the `patterns.py` source can craft text that avoids
all regex patterns while still conveying the injected instruction semantically.

**Mitigation:** Config C's classifier is trained on semantic features (TF-IDF
n-grams) and is harder to evade purely lexically.  For production, replace the
logistic regression with a fine-tuned embedding-based classifier.

### 6.2 Classifier Adversarial Examples
The scikit-learn logistic regression in Config C is vulnerable to adversarial
perturbations that preserve human-readable meaning but shift the classifier's
decision boundary.

**Mitigation:** Retrain the classifier periodically on newly observed attack
variants.  Monitor `classifier_score` in annotate mode to detect drift.

### 6.3 Semantic Paraphrase Attacks
A sufficiently sophisticated adversary can paraphrase injection instructions
in a way that no known pattern matches and the classifier assigns a low score.

**Mitigation:** This is a fundamental limitation of detection-based approaches.
Complement with output monitoring (after_model hook) and rate-limiting.

### 6.4 Multi-Turn Slow Poisoning
An adversary may spread an injection across multiple turns, where no single
turn triggers the detector.

**Mitigation:** Currently the middleware scans each message independently.
Cross-turn context accumulation is a planned future enhancement.

### 6.5 Insider / Compromised System Prompt
If an attacker can modify the system prompt (e.g. through a compromised
deployment pipeline), the trust boundary is violated before the middleware
runs.

**Mitigation:** Out of scope for this package.  Use infrastructure-level
controls (access management, GitOps signing) to protect the system prompt.

---

## 7. Security Recommendations for Production Deployment

1. **Use Config B or C** for production; Config A alone is insufficient
   against motivated adversaries.

2. **Start in `annotate` mode**, collect logs for one week, then switch
   to `block` once the false-positive rate is acceptable.

3. **Treat all RAG content as untrusted.**  Do not set `scan_tool_outputs=False`
   in any deployment that fetches external content.

4. **Monitor `risk_score` distributions** over time.  A sudden increase in
   borderline scores (0.40–0.55) may indicate an active adversary probing
   the threshold.

5. **Retrain the Config C classifier** monthly on logged borderline cases
   with human-reviewed labels.

6. **Set a latency budget** using `PerformanceProfiler.profile(budget_ms=...)`
   and alert when mean latency exceeds the budget, as this can indicate
   regex-complexity DoS attempts via pathological input.

7. **Rotate pattern versions** periodically.  Publish `patterns.py` with a
   version string so blue teams can track which pattern set is in production.

---

## 8. Non-Goals

This middleware is **not** a complete security solution.  It does not:

- Prevent jailbreaks that exploit the model's own in-weights knowledge.
- Detect attacks that succeed without any textual injection signal.
- Replace application-level authentication and authorisation.
- Protect against model-weight tampering or supply-chain attacks on the LLM.

---

## 9. References

- Perez & Ribeiro (2022). *Ignore Previous Prompt: Attack Techniques For Language Models.*
- Yi et al. (2023). *Benchmarking and Defending Against Indirect Prompt Injection in Large Language Models.*  
- Schulhoff et al. (2023). *Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs.*  
- Greshake et al. (2023). *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injections.*
- Wallace et al. (2021). *Universal Adversarial Triggers for Attacking and Analyzing NLP.*
