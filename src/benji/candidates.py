"""Loose detection of calls that *might* cost money — the coverage denominator.

The shipped tool needs its own denominator so it can say "analysed 34 of 51",
which is the line that turns a miss from silent into visible. The scoring
harness under ``corpus/`` keeps a separate copy of this logic on purpose: a
measuring instrument that shares code with the system it measures agrees with
its own bugs, so the two are allowed to drift.

Deliberately over-inclusive. A candidate we cannot price is an admitted gap in
the comment; a metered call that never became a candidate is invisible.
"""

from __future__ import annotations

import ast
import warnings

from benji.extract import attr_path

STRONG_VERBS = frozenset(
    {
        "acompletion",
        "achat",
        "chat",
        "chat_completion",
        "completion",
        "completions",
        "count_tokens",
        "embeddings",
        "generate_content",
        "messages",
        "invoke",
        "ainvoke",
    }
)

WEAK_VERBS = frozenset(
    {
        "acreate",
        "astream",
        "call",
        "complete",
        "create",
        "embed",
        "generate",
        "predict",
        "run",
        "send",
        "stream",
    }
)

NOUNS = frozenset(
    {
        "anthropic",
        "azure",
        "bedrock",
        "chat",
        "claude",
        "cohere",
        "completions",
        "embeddings",
        "genai",
        "gpt",
        "groq",
        "litellm",
        "llm",
        "messages",
        "model",
        "ollama",
        "openai",
        "responses",
        "vertex",
    }
)

ANTI_NOUNS = frozenset(
    {
        "bucket",
        "cache",
        "conn",
        "connection",
        "cursor",
        "database",
        "db",
        "engine",
        "log",
        "logger",
        "objects",
        "parser",
        "pool",
        "queryset",
        "redis",
        "s3",
        "session",
        "socket",
        "table",
        "thread",
    }
)

BARE_NAMES = frozenset({"acompletion", "chat_completion", "completion", "text_completion"})

# Receivers that are metered when called directly: `self.llm(messages)`. Narrow
# on purpose — `self.model(x)` is a forward pass in every piece of PyTorch ever
# written, so `model` is excluded.
CALLABLE_RECEIVERS = frozenset(
    {"llm", "_llm", "chat_llm", "llm_client", "language_model", "chat_model"}
)

HTTP_VERBS = frozenset({"post", "request", "stream"})

ENDPOINT_HINTS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/messages",
    "/v1/embeddings",
    "/v1/responses",
    "/api/chat",
    "/api/generate",
    "/api/embed",
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
)


def _strings(node: ast.Call) -> list[str]:
    found = []
    for child in list(node.args) + [kw.value for kw in node.keywords]:
        for sub in ast.walk(child):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                found.append(sub.value)
    return found


def is_candidate(path: tuple[str, ...], node: ast.Call) -> bool:
    if not path:
        return False
    lowered = [segment.lower() for segment in path]
    tail, context = lowered[-1], lowered[:-1]

    if tail in CALLABLE_RECEIVERS:
        return True

    if len(path) == 1:
        return path[0] in BARE_NAMES  # case-sensitive: a capitalised bare name is a class

    if tail in STRONG_VERBS:
        return True
    if tail in WEAK_VERBS:
        if not any(segment in ANTI_NOUNS for segment in context) and any(
            segment in NOUNS for segment in context
        ):
            return True
    if tail in HTTP_VERBS:
        return any(hint in text for text in _strings(node) for hint in ENDPOINT_HINTS)
    return False


def count_candidates(source: str, filename: str = "<unknown>") -> int:
    """How many calls in this file might cost money."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=filename)
    except (SyntaxError, ValueError, RecursionError):
        return 0
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and is_candidate(attr_path(node.func), node)
    )
