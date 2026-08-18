"""Loose detection of *candidate* metered call sites.

This produces the **denominator** of the coverage number. It is deliberately
over-inclusive: a candidate is any call that a reasonable person reading the
code might suspect costs money. The analyser's job is to convert candidates into
priced call sites, and ``resolved / candidates`` is only an honest measurement if
the denominator was chosen without knowing what the analyser can do.

Over-counting is the conservative direction. A candidate we cannot price shows up
as an admitted gap in the PR comment; a metered call that never became a
candidate is invisible, which is the failure this whole file exists to prevent.
"""

from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass

# Tail attributes that are metered on their own. `messages` and `completions`
# appear here as tails for the namespace-only forms some SDKs expose.
STRONG_VERBS = frozenset(
    {
        "acompletion",
        "achat",
        "chat",
        "chat_completion",
        "chat_completions",
        "completion",
        "completions",
        "count_tokens",
        "embeddings",
        "generate_content",
        "messages",
    }
)

# Tail attributes that are metered *in the right company* only. `create` alone is
# hopeless — every ORM in the world has one.
#
# `acreate` sits here rather than with the strong verbs because Django's async
# ORM uses it: 26 of khoj's 36 candidates were `Model.objects.acreate(...)`,
# which is a database write. The old OpenAI shape `openai.ChatCompletion.acreate`
# still matches, because `openai` is a provider noun in its path.
WEAK_VERBS = frozenset(
    {
        "acreate",
        "ainvoke",
        "astream",
        "call",
        "complete",
        "create",
        "embed",
        "generate",
        "invoke",
        "predict",
        "run",
        "send",
        "stream",
    }
)

# Any segment of the attribute path that suggests a model provider or client.
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

# Segments that mean "this is a database, a cache or a logger, not a model".
# These suppress weak matches; a strong verb still counts.
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

# Bare function calls that are metered — the `from litellm import completion`
# shape. Kept short on purpose; a bare name carries no receiver evidence at all.
#
# Matched CASE-SENSITIVELY, unlike everything else here. The corpus found
# prompt_toolkit's `Completion(...)` class in aider being counted as a metered
# call: a capitalised bare name is a constructor by convention, and lowercasing
# threw away the one piece of evidence that distinguished them.
BARE_NAMES = frozenset(
    {
        "acompletion",
        "chat_completion",
        "completion",
        "text_completion",
    }
)

# Providers reached over raw HTTP rather than through an SDK. open-webui calls
# `session.post(f"{url}/api/chat", ...)` and never imports a provider SDK at all,
# so an attribute-path detector sees nothing. Those calls cost money, and a
# denominator that silently omits them would make coverage look good by ignoring
# the cases we cannot handle.
#
# Matched against string constants in the call's own arguments, including the
# literal parts of f-strings. Endpoint paths are specific enough to stay quiet.
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


@dataclass(frozen=True)
class Candidate:
    """One call that might cost money.

    Carries no identity. Identity is ``file : qualified_function : ordinal`` and
    it belongs to the analyser (build step 1), not to the measuring instrument.
    ``lineno`` here is for a human going to look at the code, nothing more.
    """

    file: str
    lineno: int
    path: tuple[str, ...]
    rules: tuple[str, ...]

    @property
    def dotted(self) -> str:
        """Render the path the way it appeared in source.

        ``("get_client", "()", "chat")`` -> ``get_client().chat``. The
        placeholders are joined onto their receiver rather than separated by a
        dot, because these strings are read by a human going to find the call.
        """
        return ".".join(self.path).replace(".()", "()").replace(".[]", "[]")

    @property
    def receiver(self) -> str | None:
        return self.path[-2] if len(self.path) >= 2 else None


def attr_path(node: ast.AST) -> tuple[str, ...]:
    """Flatten a call target into its dotted segments.

    ``self.llm.chat`` -> ``("self", "llm", "chat")``. Non-name links are kept as
    placeholders rather than dropped, so ``get_client().chat`` becomes
    ``("get_client", "()", "chat")`` and never collapses into a bare ``chat``.
    Keeping the factory's own name matters: step 2 resolves receivers, and
    ``get_client`` is exactly the evidence it will need.
    """
    parts: list[str] = []
    current = node
    while True:
        if isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        elif isinstance(current, ast.Name):
            parts.append(current.id)
            break
        elif isinstance(current, ast.Call):
            parts.append("()")
            current = current.func
        elif isinstance(current, ast.Subscript):
            parts.append("[]")
            current = current.value
        else:
            parts.append("?")
            break
    return tuple(reversed(parts))


def string_constants(node: ast.Call) -> list[str]:
    """Every string literal appearing in the call's own arguments.

    Includes the constant segments of f-strings, which is where the endpoint
    usually lives: ``session.post(f"{base}/api/chat")``.
    """
    found: list[str] = []
    for child in list(node.args) + [kw.value for kw in node.keywords]:
        for sub in ast.walk(child):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                found.append(sub.value)
    return found


def classify(path: tuple[str, ...], call: ast.Call | None = None) -> tuple[str, ...]:
    """Return the rules that fire for this call, or ``()`` for none.

    ``call`` is optional so the attribute-path rules stay unit-testable on their
    own; the HTTP rule needs the arguments and is skipped without it.
    """
    if not path:
        return ()

    lowered = [segment.lower() for segment in path]
    tail = lowered[-1]
    context = lowered[:-1]

    # The model object called directly, rather than through a method.
    if tail in CALLABLE_RECEIVERS:
        return ("callable",)

    if len(path) == 1:
        # Case-sensitive on purpose — see BARE_NAMES.
        return ("bare",) if path[0] in BARE_NAMES else ()

    rules: list[str] = []
    if tail in STRONG_VERBS:
        rules.append("strong")

    if tail in WEAK_VERBS:
        suppressed = any(segment in ANTI_NOUNS for segment in context)
        if not suppressed and any(segment in NOUNS for segment in context):
            rules.append("weak+noun")

    # Deliberately not gated on ANTI_NOUNS: the canonical shape is
    # `session.post(...)`, and `session` is an anti-noun.
    if call is not None and tail in HTTP_VERBS:
        literals = string_constants(call)
        if any(hint in literal for literal in literals for hint in ENDPOINT_HINTS):
            rules.append("http+endpoint")

    return tuple(rules)


def find_candidates(source: str, filename: str) -> list[Candidate]:
    """Parse ``source`` and return every candidate metered call in it.

    Uses ``ast.walk``, which discards the enclosing scope. That is fine and
    intentional: counting does not need identity.

    Raises ``SyntaxError`` on unparseable input; the caller decides whether an
    unparseable file is a skipped file or a failure.
    """
    # Third-party code emits its own SyntaxWarnings (stray `\d` in a non-raw
    # string, and so on). Those belong to the corpus, not to us, and they would
    # bury the scoring output.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source, filename=filename)

    candidates: list[Candidate] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        path = attr_path(node.func)
        rules = classify(path, node)
        if rules:
            candidates.append(Candidate(file=filename, lineno=node.lineno, path=path, rules=rules))

    candidates.sort(key=lambda c: (c.file, c.lineno))
    return candidates
