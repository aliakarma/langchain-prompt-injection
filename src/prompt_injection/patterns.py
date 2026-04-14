"""
patterns.py
───────────
Curated, compiled regex pattern library for prompt-injection detection.

Patterns are grouped by *attack category* so the detector can report
which category fired, enabling per-category ablation and hit-rate
analysis in the evaluation notebooks.

Each entry in PATTERN_REGISTRY is a dict::

    {
        "id":       str,          # stable unique identifier
        "category": str,          # attack category label
        "severity": str,          # "low" | "medium" | "high"
        "pattern":  re.Pattern,   # pre-compiled, IGNORECASE | DOTALL
        "description": str,       # human-readable explanation
    }

Usage
-----
    from prompt_injection.patterns import PATTERN_REGISTRY, get_patterns_by_category

    for entry in PATTERN_REGISTRY:
        if entry["pattern"].search(text):
            print(entry["category"], entry["id"])
"""

from __future__ import annotations

import re
from typing import TypedDict


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------


class PatternEntry(TypedDict):
    id: str
    category: str
    severity: str
    pattern: re.Pattern[str]
    description: str


# ---------------------------------------------------------------------------
# Compilation helper
# ---------------------------------------------------------------------------

_FLAGS = re.IGNORECASE | re.DOTALL


def _p(pattern: str) -> re.Pattern[str]:
    """Compile with standard flags."""
    return re.compile(pattern, _FLAGS)


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

PATTERN_REGISTRY: list[PatternEntry] = [
    # ── Category 1: Instruction Override ──────────────────────────────────
    {
        "id": "IO-001",
        "category": "instruction_override",
        "severity": "high",
        "pattern": _p(r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?"),
        "description": "Classic instruction override: 'ignore previous instructions'",
    },
    {
        "id": "IO-002",
        "category": "instruction_override",
        "severity": "high",
        "pattern": _p(r"disregard\s+(all\s+)?(previous|prior|above|the\s+)?(instructions?|prompts?|rules?|constraints?)"),
        "description": "Disregard variant of instruction override",
    },
    {
        "id": "IO-003",
        "category": "instruction_override",
        "severity": "high",
        "pattern": _p(r"do\s+not\s+follow\s+(the\s+)?(above|previous|prior|original)\s+(instructions?|rules?|guidelines?)"),
        "description": "Negation-based instruction override",
    },
    {
        "id": "IO-004",
        "category": "instruction_override",
        "severity": "medium",
        "pattern": _p(r"forget\s+(everything|all)\s+(you\s+)?(were\s+)?(told|instructed|trained|taught)"),
        "description": "'Forget everything you were told' override",
    },
    {
        "id": "IO-005",
        "category": "instruction_override",
        "severity": "medium",
        "pattern": _p(r"override\s+(your\s+)?(instructions?|programming|directives?|system\s+prompt)"),
        "description": "Explicit override keyword targeting instructions",
    },
    {
        "id": "IO-006",
        "category": "instruction_override",
        "severity": "high",
        "pattern": _p(r"new\s+(instructions?|rules?|directives?)\s*:\s*(from\s+(now|this\s+point)|you\s+(must|should|will))"),
        "description": "Injected new instructions claim",
    },
    {
        "id": "IO-007",
        "category": "instruction_override",
        "severity": "medium",
        "pattern": _p(r"your\s+(real|true|actual|original)\s+(instructions?|purpose|goal|task)\s+is"),
        "description": "Redefinition of assistant's purpose",
    },
    {
        "id": "IO-008",
        "category": "instruction_override",
        "severity": "high",
        "pattern": _p(r"ignore\s+(your\s+)?(system\s+prompt|developer\s+(message|instructions?)|previous\s+context)"),
        "description": "Instruction override targeting system prompt directly",
    },

    # ── Category 2: System Prompt Extraction ──────────────────────────────
    {
        "id": "SPE-001",
        "category": "system_prompt_extraction",
        "severity": "high",
        "pattern": _p(r"(reveal|print|show|output|display|repeat|tell\s+me)\s+(the\s+)?(your\s+)?(full\s+|complete\s+|entire\s+|hidden\s+|original\s+)?(system\s+prompt|system\s+message|developer\s+(message|instructions?)|initial\s+(instructions?|prompt))"),
        "description": "Direct system prompt extraction request",
    },
    {
        "id": "SPE-002",
        "category": "system_prompt_extraction",
        "severity": "high",
        "pattern": _p(r"what\s+(are|were|is)\s+(your\s+)?(system\s+prompt|initial\s+instructions?|hidden\s+instructions?|secret\s+instructions?)"),
        "description": "Question-form system prompt extraction",
    },
    {
        "id": "SPE-003",
        "category": "system_prompt_extraction",
        "severity": "medium",
        "pattern": _p(r"(output|print|repeat|echo)\s+(everything|all)\s+(above|before|prior|from\s+the\s+beginning)"),
        "description": "Echo-everything prompt dump",
    },
    {
        "id": "SPE-004",
        "category": "system_prompt_extraction",
        "severity": "high",
        "pattern": _p(r"(translate|summarize|paraphrase|rewrite)\s+(your\s+)?(system\s+prompt|instructions?|context)\s+(into|to|as)"),
        "description": "Indirect extraction via translation/summarisation",
    },
    {
        "id": "SPE-005",
        "category": "system_prompt_extraction",
        "severity": "medium",
        "pattern": _p(r"(what|which)\s+(instructions?|rules?|constraints?|guidelines?)\s+(have\s+you\s+been\s+given|are\s+you\s+following|were\s+you\s+given)"),
        "description": "Indirect extraction via rule inquiry",
    },
    {
        "id": "SPE-006",
        "category": "system_prompt_extraction",
        "severity": "high",
        "pattern": _p(r"(reveal|expose|show|print|output)\s+(it|them|everything)\s*(now|immediately|please|below)?"),
        "description": "Reveal with pronoun — indirect system prompt extraction",
    },

    # ── Category 3: Role Hijacking ─────────────────────────────────────────
    {
        "id": "RH-001",
        "category": "role_hijacking",
        "severity": "high",
        "pattern": _p(r"you\s+are\s+now\s+(a|an|the)?\s*(unrestricted|uncensored|free|unfiltered|jailbroken|different|new|alternative)\s+(ai|assistant|model|bot|llm)"),
        "description": "Role redefinition to unrestricted persona",
    },
    {
        "id": "RH-002",
        "category": "role_hijacking",
        "severity": "high",
        "pattern": _p(r"(act|pretend|behave|respond)\s+as\s+(if\s+you\s+are\s+|like\s+)?(a|an|the)?\s*(evil|malicious|hacked|uncensored|unrestricted)"),
        "description": "Evil/uncensored persona instruction",
    },
    {
        "id": "RH-003",
        "category": "role_hijacking",
        "severity": "medium",
        "pattern": _p(r"from\s+(now\s+on|this\s+point|here)\s+(you\s+are|act\s+as|pretend\s+to\s+be|behave\s+as)"),
        "description": "Persistent role override from this point forward",
    },
    {
        "id": "RH-004",
        "category": "role_hijacking",
        "severity": "medium",
        "pattern": _p(r"switch\s+(to|into)\s+(a\s+)?(different|new|alternative|another)\s+(mode|role|persona|character|personality)"),
        "description": "Mode/role switch instruction",
    },
    {
        "id": "RH-005",
        "category": "role_hijacking",
        "severity": "high",
        "pattern": _p(r"your\s+(new\s+)?(name|identity|role|persona)\s+is\s+\w+"),
        "description": "Identity/name reassignment",
    },
    {
        "id": "RH-006",
        "category": "role_hijacking",
        "severity": "medium",
        "pattern": _p(r"pretend\s+you\s+are\s+(a|an|the)?\s*\w+"),
        "description": "Explicit 'pretend you are' role hijack",
    },
    {
        "id": "RH-007",
        "category": "role_hijacking",
        "severity": "medium",
        "pattern": _p(r"\bact\s+as\b\s+(a|an|the)?\s*\w+"),
        "description": "Explicit 'act as' role hijack",
    },
    {
        "id": "RH-008",
        "category": "role_hijacking",
        "severity": "medium",
        "pattern": _p(r"imagine\s+you\s+are\s+(a|an|the)?\s*\w+"),
        "description": "Imagined persona role hijack",
    },
    {
        "id": "RH-009",
        "category": "role_hijacking",
        "severity": "high",
        "pattern": _p(r"from\s+now\s+on\s+you\s+are\s+(a|an|the)?\s*\w+"),
        "description": "Persistent role assignment from now on",
    },
    {
        "id": "RH-010",
        "category": "role_hijacking",
        "severity": "high",
        "pattern": _p(r"you\s+have\s+no\s+restrictions?"),
        "description": "No-restrictions persona framing",
    },
    {
        "id": "RH-011",
        "category": "role_hijacking",
        "severity": "medium",
        "pattern": _p(r"your\s+true\s+self"),
        "description": "'your true self' persona invocation",
    },

    # ── Category 4: DAN / Jailbreak ────────────────────────────────────────
    {
        "id": "DAN-001",
        "category": "dan_jailbreak",
        "severity": "high",
        "pattern": _p(r"\bDAN\b"),
        "description": "DAN (Do Anything Now) keyword",
    },
    {
        "id": "DAN-002",
        "category": "dan_jailbreak",
        "severity": "high",
        "pattern": _p(r"do\s+anything\s+now"),
        "description": "DAN phrase in full",
    },
    {
        "id": "DAN-003",
        "category": "dan_jailbreak",
        "severity": "high",
        "pattern": _p(r"(developer|jailbreak|god|sudo|admin|root|privileged)\s+mode"),
        "description": "Privileged/developer mode activation attempt",
    },
    {
        "id": "DAN-004",
        "category": "dan_jailbreak",
        "severity": "high",
        "pattern": _p(r"you\s+(have\s+been\s+|are\s+now\s+)?(freed?|liberated?|unlocked?|jailbroken?)\s+from\s+(your\s+)?(restrictions?|constraints?|limitations?|rules?|guidelines?)"),
        "description": "Liberation/freedom from restrictions framing",
    },
    {
        "id": "DAN-005",
        "category": "dan_jailbreak",
        "severity": "medium",
        "pattern": _p(r"(disable|bypass|remove|turn\s+off|deactivate)\s+(your\s+)?(safety|content\s+filter|guardrail|restriction|censorship|moderation)"),
        "description": "Safety system bypass instruction",
    },
    {
        "id": "DAN-006",
        "category": "dan_jailbreak",
        "severity": "medium",
        "pattern": _p(r"(no\s+restrictions?|without\s+restrictions?|unrestricted(ly)?)\s+(respond|answer|reply|generate|output|provide)"),
        "description": "Unrestricted response demand",
    },

    # ── Category 5: Boundary Escape ────────────────────────────────────────
    {
        "id": "BE-001",
        "category": "boundary_escape",
        "severity": "high",
        "pattern": _p(r"disregard\s+(safety|ethical|moral)\s+(guidelines?|rules?|constraints?|instructions?)"),
        "description": "Safety guideline disregard",
    },
    {
        "id": "BE-002",
        "category": "boundary_escape",
        "severity": "medium",
        "pattern": _p(r"(print|output|show|reveal|expose)\s+(hidden|secret|confidential|internal|private)\s+(context|data|information|memory|state)"),
        "description": "Hidden context/memory extraction attempt",
    },
    {
        "id": "BE-003",
        "category": "boundary_escape",
        "severity": "high",
        "pattern": _p(r"(ignore|bypass|circumvent|skip)\s+(the\s+)?(safety\s+check|content\s+policy|usage\s+policy|terms\s+of\s+service|ethical\s+guidelines?)"),
        "description": "Policy bypass instruction",
    },
    {
        "id": "BE-004",
        "category": "boundary_escape",
        "severity": "medium",
        "pattern": _p(r"this\s+is\s+(a\s+)?(test|simulation|hypothetical|fictional|training)\s+(scenario|environment|mode).*?(ignore|bypass|disregard)"),
        "description": "Fictional framing used to justify bypass",
    },
    {
        "id": "BE-005",
        "category": "boundary_escape",
        "severity": "medium",
        "pattern": _p(r"(for\s+)?(research|educational|academic|scientific)\s+purposes?\s*(only)?\s*[,:]?\s*(ignore|bypass|provide|reveal|tell)"),
        "description": "Research framing as bypass justification",
    },

    # ── Category 6: Social Engineering ────────────────────────────────────
    {
        "id": "SE-001",
        "category": "social_engineering",
        "severity": "medium",
        "pattern": _p(r"(your|the)\s+(developer|creator|maker|trainer|anthropic|openai)\s+(told|instructed|said|wants?|requires?)\s+(you|me|us)\s+to"),
        "description": "False authority claim from developer/creator",
    },
    {
        "id": "SE-002",
        "category": "social_engineering",
        "severity": "medium",
        "pattern": _p(r"i\s+(have|had)\s+(permission|authorization|clearance|access)\s+to\s+(see|view|access|receive|obtain)"),
        "description": "False permission/authorization claim",
    },
    {
        "id": "SE-003",
        "category": "social_engineering",
        "severity": "medium",
        "pattern": _p(r"(the\s+)?(admin|administrator|superuser|operator|owner)\s+(has\s+)?(granted|given|approved|authorized)\s+(me|us|this)"),
        "description": "False admin approval claim",
    },
    {
        "id": "SE-004",
        "category": "social_engineering",
        "severity": "low",
        "pattern": _p(r"(this\s+is\s+)?(an?\s+)?(emergency|urgent|critical)\s+(situation|case|request)\s+(that\s+)?(requires?|needs?)\s+(you\s+to\s+)?(bypass|ignore|override|skip)"),
        "description": "Urgency/emergency used to justify bypass",
    },

    # ── Category 7: Indirect Injection (RAG / Tool) ────────────────────────
    {
        "id": "IND-001",
        "category": "indirect_injection",
        "severity": "high",
        "pattern": _p(r"\[\s*(system|assistant|human|ai|instruction)\s*\]\s*:"),
        "description": "Injected role tag in retrieved content",
    },
    {
        "id": "IND-002",
        "category": "indirect_injection",
        "severity": "high",
        "pattern": _p(r"<\s*(system|instruction|prompt|command)\s*>"),
        "description": "XML-style injected tag in retrieved content",
    },
    {
        "id": "IND-003",
        "category": "indirect_injection",
        "severity": "high",
        "pattern": _p(r"---\s*(system|instruction|hidden\s+prompt|new\s+instructions?)\s*---"),
        "description": "Markdown separator injected instruction block",
    },
    {
        "id": "IND-004",
        "category": "indirect_injection",
        "severity": "medium",
        "pattern": _p(r"(when\s+you\s+read\s+this|upon\s+reading\s+this|attention\s+ai|note\s+to\s+(ai|assistant|model|llm))\s*:"),
        "description": "Attention-grabbing indirect injection opener",
    },
    {
        "id": "IND-005",
        "category": "indirect_injection",
        "severity": "high",
        "pattern": _p(r"(execute|run|perform|carry\s+out)\s+(the\s+)?(following\s+)?(instructions?|commands?|actions?)\s*:"),
        "description": "Embedded command execution instruction",
    },

    # ── Category 8: Evasion Indicators ────────────────────────────────────
    {
        "id": "EVA-001",
        "category": "evasion",
        "severity": "medium",
        "pattern": _p(r"(?:\bi\W*g\W*n\W*o\W*r\W*e\b|\bd\W*i\W*s\W*r\W*e\W*g\W*a\W*r\W*d\b).{0,30}(?:\bp\W*r\W*e\W*v\W*i\W*o\W*u\W*s\b|\binstructions?\b)"),
        "description": "Letter-spaced or punctuated evasion near override intent",
    },
    {
        "id": "EVA-002",
        "category": "evasion",
        "severity": "medium",
        "pattern": _p(r"base64\s*:\s*[A-Za-z0-9+/]{20,}={0,2}"),
        "description": "Base64-encoded content in user/tool message",
    },
    {
        "id": "EVA-003",
        "category": "evasion",
        "severity": "low",
        "pattern": _p(r"(\bingnore\b|\bignroe\b|\bdisreguard\b|\bignor3\b|\b1gnore\b|\bi9nore\b|\big[nm]0re\b)"),
        "description": "Common typo and partial-obfuscation variants of override keywords",
    },
]


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def get_patterns_by_category(category: str) -> list[PatternEntry]:
    """Return all patterns belonging to *category*."""
    return [e for e in PATTERN_REGISTRY if e["category"] == category]


def get_patterns_by_severity(severity: str) -> list[PatternEntry]:
    """Return all patterns with a given *severity* (low/medium/high)."""
    return [e for e in PATTERN_REGISTRY if e["severity"] == severity]


def list_categories() -> list[str]:
    """Return sorted unique list of all category names."""
    return sorted({e["category"] for e in PATTERN_REGISTRY})


# Category-level severity weights used by the scoring layer.
CATEGORY_WEIGHTS: dict[str, float] = {
    "instruction_override": 0.90,
    "system_prompt_extraction": 0.85,
    "role_hijacking": 0.80,
    "dan_jailbreak": 0.95,
    "boundary_escape": 0.75,
    "social_engineering": 0.60,
    "indirect_injection": 0.85,
    "evasion": 0.50,
}

# Severity multipliers for per-hit risk contribution.
SEVERITY_WEIGHTS: dict[str, float] = {
    "high": 1.0,
    "medium": 0.65,
    "low": 0.35,
}
