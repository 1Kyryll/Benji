"""One file in, ``list[CallSite]`` out.

This is the first of the three layers. It answers *which calls cost money and
where are they*, and nothing else — not how often they run, not how much they
cost. Keeping those apart is why the bot can later say which of them moved.

Two call shapes are recognised, and they are the same thing wearing different
syntax:

* **SDK** — ``client.chat.completions.create(model="gpt-4o", ...)``. The model is
  a keyword argument.
* **HTTP** — ``session.post(f"{base}/api/chat", json={"model": "llama3", ...})``.
  The model is a key in the request payload.

A quarter of real application code uses the second shape and never imports a
provider SDK at all. Treating it as a separate subsystem would have meant two of
everything downstream; treating it as another way to spell the same call site
means the rest of the pipeline never learns the difference.
"""

from __future__ import annotations

import ast
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

Literal = str | int | float | bool | None

MODULE_SCOPE = "<module>"

# Attribute-path tails that are known metered SDK calls, longest tail first.
# The tail is what matters, so any receiver name matches: `client`, `self._oai`
# and `get_client()` all resolve the same way.
SDK_DETECTORS: dict[tuple[str, ...], tuple[str, str]] = {
    ("chat", "completions", "create"): ("openai", "chat"),
    ("chat", "completions", "parse"): ("openai", "chat"),
    ("beta", "messages", "create"): ("anthropic", "messages"),
    ("completions", "create"): ("openai", "completions"),
    ("embeddings", "create"): ("openai", "embeddings"),
    ("responses", "create"): ("openai", "responses"),
    ("messages", "create"): ("anthropic", "messages"),
    ("messages", "stream"): ("anthropic", "messages"),
}

# litellm is called as a plain function, either bare or off the module.
LITELLM_NAMES = frozenset({"completion", "acompletion", "text_completion"})

# Framework calls: LangChain and friends. `llm.invoke(messages)` is a metered
# call, but the receiver's type lives in a third-party package we refuse to parse,
# so the wrapper index stops at the repository boundary and can never prove it.
#
# The corpus says this is the dominant unresolved shape in real applications —
# people wrap the SDK in someone else's abstraction far more often than in one of
# their own. Refusing to detect it would mean a bot that reports nothing on the
# most common way Python code calls a model.
#
# The evidence is a naming convention rather than a type, so these sites carry
# reduced confidence and are tiered accordingly rather than reported as certain.
FRAMEWORK_VERBS = frozenset(
    {"invoke", "ainvoke", "stream", "astream", "batch", "abatch", "chat", "achat"}
)

FRAMEWORK_RECEIVERS = frozenset(
    {"llm", "_llm", "model", "_model", "chat_model", "chat_llm", "llm_client", "language_model"}
)

# Receivers that are metered when *called directly*: `self.llm(messages)` rather
# than `self.llm.invoke(messages)`. LangChain models are callable, and older code
# uses this form everywhere.
#
# Narrower than FRAMEWORK_RECEIVERS on purpose. `model` and `_model` are excluded
# because `self.model(x)` is a forward pass in every piece of PyTorch ever
# written, and a false positive there would price a tensor multiply.
FRAMEWORK_CALLABLE_RECEIVERS = frozenset(
    {"llm", "_llm", "chat_llm", "llm_client", "language_model", "chat_model"}
)

FRAMEWORK_CONFIDENCE = 0.7

HTTP_VERBS = frozenset({"post", "request", "stream"})

# Endpoint -> provider. Deliberately duplicated from ``corpus/detect.py`` rather
# than shared: the harness measures this module, so the denominator must not
# shift every time this table improves.
HTTP_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("api.openai.com", "openai"),
    ("api.anthropic.com", "anthropic"),
    ("generativelanguage.googleapis.com", "google"),
    ("/v1/chat/completions", "openai"),
    ("/v1/completions", "openai"),
    ("/v1/embeddings", "openai"),
    ("/v1/responses", "openai"),
    ("/v1/messages", "anthropic"),
    ("/api/chat", "ollama"),
    ("/api/generate", "ollama"),
    ("/api/embed", "ollama"),
)

# Payload keywords that carry a JSON request body.
PAYLOAD_KWARGS = ("json", "data", "json_data")


@dataclass(frozen=True)
class WrapperHint:
    """What the wrapper index learned about a call that is not literally an SDK call.

    ``self.llm.chat()`` costs money, but only because something further down does.
    The index works that out; this carries the answer back so ordinals are
    assigned in one place instead of two.
    """

    provider: str
    api: str
    depth: int
    confidence: float


@dataclass(frozen=True)
class CallSite:
    """One metered call, identified by position rather than by line.

    ``id`` is ``file : qualified_function : ordinal``. It survives inserted
    imports, reformatting and unrelated edits, and it breaks on function rename,
    file move, or a new metered call inserted above this one. A line-number key
    would break on all of those *and* on a whitespace change, orphaning whatever
    telemetry was filed under it.

    ``lineno`` is carried for display and for the telemetry join, and is
    deliberately not part of ``id``.
    """

    file: str
    qualname: str
    ordinal: int
    provider: str
    api: str
    shape: str  # "sdk" | "http" | "framework" | "wrapper"
    model: str | None
    lineno: int
    literals: tuple[tuple[str, Literal], ...] = ()
    confidence: float = 1.0
    depth: int = 0
    # The call as written, normalised by `ast.unparse`. Survives reformatting;
    # changes the moment someone edits the call, which is exactly when the
    # content-hash matcher should stop claiming it is the same site.
    content: str = ""

    @property
    def id(self) -> str:
        return f"{self.file}:{self.qualname}:{self.ordinal}"


def attr_path(node: ast.AST) -> tuple[str, ...]:
    """Flatten a call target into its dotted segments.

    ``self.llm.chat`` -> ``("self", "llm", "chat")``. Links that are not plain
    names become placeholders rather than vanishing, so ``get_client().chat``
    keeps evidence that a call produced the receiver.
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


def kwarg_literal(node: ast.Call, name: str) -> Literal:
    """The literal value of keyword ``name``, or ``None`` if it is not knowable.

    ``model=config.DEFAULT_MODEL`` yields ``None``, and that is correct output
    rather than a gap: downstream stages have to be able to see that the model is
    unresolved. Absent and unresolvable both return ``None`` on purpose — there
    is no sentinel to mistake for a real value.
    """
    for keyword in node.keywords:
        if keyword.arg == name:
            if isinstance(keyword.value, ast.Constant):
                return keyword.value.value
            return None
    return None


def dict_literal(node: ast.Dict, key: str) -> Literal:
    """The literal value stored under ``key`` in a dict display, else ``None``."""
    for key_node, value_node in zip(node.keys, node.values, strict=False):
        if isinstance(key_node, ast.Constant) and key_node.value == key:
            if isinstance(value_node, ast.Constant):
                return value_node.value
            return None
    return None


def string_constants(node: ast.Call) -> list[str]:
    """Every string literal in the call's own arguments, f-string parts included."""
    found: list[str] = []
    for child in list(node.args) + [kw.value for kw in node.keywords]:
        for sub in ast.walk(child):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                found.append(sub.value)
    return found


def match_sdk(path: tuple[str, ...]) -> tuple[str, str] | None:
    """Resolve an attribute path to ``(provider, api)`` if it is a known SDK call."""
    for width in (3, 2):
        if len(path) >= width:
            hit = SDK_DETECTORS.get(path[-width:])
            if hit:
                return hit

    tail = path[-1]
    if tail in LITELLM_NAMES and (len(path) == 1 or path[-2].lower() == "litellm"):
        return ("litellm", "completion")

    return None


def match_framework(path: tuple[str, ...]) -> tuple[str, str] | None:
    """Resolve a framework-shaped call from its receiver name.

    Three forms, all LangChain-shaped: ``llm.invoke(...)``,
    ``self._llm.chat(...)``, and ``self.llm(messages)`` — the model object called
    directly, which is what older code does and what a method-name detector
    cannot see at all.

    Weaker evidence than a type, which is why these sites carry
    ``FRAMEWORK_CONFIDENCE``.
    """
    if not path:
        return None

    # The callable-object form. The receiver *is* the call target.
    if path[-1].lower() in FRAMEWORK_CALLABLE_RECEIVERS:
        return ("framework", "call")

    if len(path) < 2 or path[-1] not in FRAMEWORK_VERBS:
        return None
    if path[-2].lower() not in FRAMEWORK_RECEIVERS:
        return None
    return ("framework", path[-1])


def match_http(path: tuple[str, ...], node: ast.Call) -> tuple[str, str] | None:
    """Resolve a raw HTTP call to ``(provider, endpoint)`` from its arguments."""
    if path[-1].lower() not in HTTP_VERBS:
        return None
    for literal in string_constants(node):
        for hint, provider in HTTP_ENDPOINTS:
            if hint in literal:
                return (provider, hint)
    return None


def http_model(node: ast.Call) -> str | None:
    """The model named in an HTTP request payload, if it is a literal."""
    for name in PAYLOAD_KWARGS:
        for keyword in node.keywords:
            if keyword.arg == name and isinstance(keyword.value, ast.Dict):
                value = dict_literal(keyword.value, "model")
                return value if isinstance(value, str) else None
    return None


class CallSiteVisitor(ast.NodeVisitor):
    """Walks a module, tracking the enclosing scope and a per-scope ordinal.

    ``ast.walk`` finds every call but discards where it sat, and the enclosing
    function is exactly what the identifier needs — hence a real visitor with a
    scope stack. The ordinal counts **metered calls only**, so adding an ordinary
    function call between two metered ones does not renumber anything, and each
    scope counts independently so a nested helper gets its own sequence.
    """

    def __init__(
        self,
        file: str,
        wrappers: Mapping[int, WrapperHint] | None = None,
        include_framework: bool = True,
    ) -> None:
        self.file = file
        self.scope: list[str] = []
        self.counters: dict[tuple[str, ...], int] = {}
        self.sites: list[CallSite] = []
        # Calls the wrapper index resolved. They are metered too, so they take
        # ordinals alongside direct calls — which is precisely why ordinals are
        # assigned here and only here.
        self.wrappers: Mapping[int, WrapperHint] = wrappers or {}
        self.include_framework = include_framework

    # A class body has its own counter: a metered call at class scope is rare
    # but it is not inside any method, and it needs somewhere to live.
    def _enter(self, node: ast.AST) -> None:
        self.scope.append(node.name)  # type: ignore[attr-defined]
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = _enter
    visit_AsyncFunctionDef = _enter
    visit_ClassDef = _enter

    def visit_Call(self, node: ast.Call) -> None:
        path = attr_path(node.func)
        site = self._build(node, path)
        if site is not None:
            self.sites.append(site)
        # Keep descending: a metered call can appear inside another call's
        # arguments, and a lambda or comprehension can hold one too.
        self.generic_visit(node)

    def _build(self, node: ast.Call, path: tuple[str, ...]) -> CallSite | None:
        if not path:
            return None

        literals: tuple[tuple[str, Literal], ...] = ()
        confidence, depth = 1.0, 0

        sdk = match_sdk(path)
        if sdk is not None:
            provider, api = sdk
            shape = "sdk"
            model = kwarg_literal(node, "model")
            model = model if isinstance(model, str) else None
            literals = (("max_tokens", kwarg_literal(node, "max_tokens")),)
        elif (http := match_http(path, node)) is not None:
            provider, endpoint = http
            api, shape = "http", "http"
            model = http_model(node)
            literals = (("endpoint", endpoint),)
        elif (hint := self.wrappers.get(node.lineno)) is not None:
            provider, api, shape = hint.provider, hint.api, "wrapper"
            # The model belongs to whatever the wrapper eventually calls, and
            # that call may take it as an argument. Unknown here is honest.
            model = kwarg_literal(node, "model")
            model = model if isinstance(model, str) else None
            confidence, depth = hint.confidence, hint.depth
            literals = (("wrapped", ".".join(path)),)
        elif self.include_framework and (framework := match_framework(path)) is not None:
            # Last, on purpose. If the wrapper index proved where this call goes,
            # that answer is better than a guess from the receiver's name.
            provider, api = framework
            shape = "framework"
            model = kwarg_literal(node, "model")
            model = model if isinstance(model, str) else None
            confidence = FRAMEWORK_CONFIDENCE
            literals = (("receiver", path[-2] if len(path) > 1 else path[-1]),)
        else:
            return None

        key = tuple(self.scope)
        ordinal = self.counters.get(key, 0)
        self.counters[key] = ordinal + 1

        return CallSite(
            file=self.file,
            qualname=".".join(self.scope) or MODULE_SCOPE,
            ordinal=ordinal,
            provider=provider,
            api=api,
            shape=shape,
            model=model,
            lineno=node.lineno,
            literals=literals,
            confidence=confidence,
            depth=depth,
            content=ast.unparse(node),
        )


def extract(
    source: str,
    filename: str,
    wrappers: Mapping[int, WrapperHint] | None = None,
    include_framework: bool = True,
) -> list[CallSite]:
    """Every metered call site in ``source``, in source order.

    Raises ``SyntaxError`` on unparseable input. The caller decides whether that
    is a skipped file or a failure; silently returning an empty list would make a
    broken parse look like a file with no cost in it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source, filename=filename)

    visitor = CallSiteVisitor(filename, wrappers, include_framework)
    visitor.visit(tree)
    return sorted(visitor.sites, key=lambda s: s.lineno)
