"""
Query intent classification for CodeSage.

Provides lightweight, deterministic heuristic classification of developer questions
to route retrieval and context sizing appropriately:
- SINGLE_SYMBOL: Specific functions, classes, methods, endpoints, or constants.
- ARCHITECTURAL: Cross-file workflows, pipelines, communication, and system design.
- DEBUGGING: Exceptions, stack traces, crash descriptions, and bug inquiries.
"""

import re

SINGLE_SYMBOL = "SINGLE_SYMBOL"
ARCHITECTURAL = "ARCHITECTURAL"
DEBUGGING = "DEBUGGING"

# Patterns indicating error reports, stack traces, or exception debugging
_DEBUGGING_PATTERNS = [
    r"\btraceback\b",
    r"file [\'\"].*?[\'\"], line \d+",
    r"\b(typeerror|valueerror|attributeerror|keyerror|indexerror|importerror|filenotfounderror|syntaxerror|runtimeerror|connectionerror|timeouterror|zerodivisionerror|assertionerror)\b",
    r"\b(500|404|400|401|403)\s*(error|status|code)?\b",
    r"\bwhy is .* (failing|broken|crashing|raising|timing out)\b",
    r"\b(fix this (error|bug|issue|exception)|how to fix (this|the) (error|bug|exception))\b",
]

# Patterns indicating cross-file workflows, architectures, pipelines, or interactions
_ARCHITECTURAL_PATTERNS = [
    r"\b(architecture|architectural|system design|high-level|overall structure|component diagram)\b",
    r"\b(pipeline|pipelines|workflow|workflows|lifecycle)\b",
    r"\b(end-to-end|end to end|e2e)\b",
    r"\b(data\s*flow|dataflow|control\s*flow)\b",
    r"\b(communicate|communicates|communication)\b",
    r"\b(interact|interacts|interaction|interactions)\b",
    r"\btrace .* (flow|path|pipeline|execution|request)\b",
    r"\bhow do .* and .* (interact|work|communicate|connect)\b",
    r"\bhow does .* (communicate with|connect to|interact with|flow into|call into|integrate with)\b",
    r"\bhow is .* passed (to|from|between)\b",
    r"\bacross (multiple|different|all) files\b",
]

# Patterns indicating a specific single-symbol or definition query
_SINGLE_SYMBOL_PATTERNS = [
    r"^(where is|where are|which function|which class|which method|what function|what class|what method)\b",
    r"\b(defined|definition|declared|declaration|initialized|implemented|implementation)\b",
    r"\b(function|method|class|variable|constant|endpoint|route|symbol)\b.*(where|which|name|signature)",
    r"\b(signature of|parameters of|arguments of|return value of|returns of)\b",
]

# Explicit test query pattern (preserved from vector_store)
_TEST_QUERY_PATTERN = r"\b(test|tests|testing|testcase|testcases|spec|specs|assert|assertion|assertions|unit\s*test|integration\s*test|mock|mocking)\b"


def is_explicit_test_query(query: str) -> bool:
    """Detects if user is explicitly asking about tests, testing, specs, or assertions."""
    return bool(re.search(_TEST_QUERY_PATTERN, query.lower()))


def classify_query_intent(query: str) -> str:
    """
    Classifies a developer query into SINGLE_SYMBOL, ARCHITECTURAL, or DEBUGGING.

    Evaluation precedence:
    1. DEBUGGING: High priority for explicit exceptions and stack traces.
    2. ARCHITECTURAL: Detects cross-file, workflow, pipeline, or system interactions.
    3. SINGLE_SYMBOL: Matches definition, location, and symbol lookup patterns.
    4. Heuristic Fallback: Identifiers (snake_case, camelCase, /endpoints) default to SINGLE_SYMBOL.
    """
    q = query.strip().lower()
    if not q:
        return SINGLE_SYMBOL

    # 1. Debugging check (high priority for crashes, exceptions, traces)
    for pat in _DEBUGGING_PATTERNS:
        if re.search(pat, q):
            return DEBUGGING

    # 2. Architectural check (cross-file flows, pipeline interactions)
    for pat in _ARCHITECTURAL_PATTERNS:
        if re.search(pat, q):
            return ARCHITECTURAL

    # 3. Single symbol definition patterns
    for pat in _SINGLE_SYMBOL_PATTERNS:
        if re.search(pat, q):
            return SINGLE_SYMBOL

    # 4. Check for code identifiers (functions like ask_llm, endpoints like /health)
    if re.search(r"[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+|[a-z]+_[a-z0-9_]+|/[a-z0-9_-]+", query):
        return SINGLE_SYMBOL

    return SINGLE_SYMBOL
