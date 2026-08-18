"""How many times a call site fires per invocation of the function holding it.

The middle layer of the equation. One `client.chat.completions.create(...)` costs
a fraction of a cent; the same line inside `for org in orgs:` costs whatever
`len(orgs)` happens to be, and that is usually where an expensive pull request
hides.

Multiplicity is a product of the constructs enclosing the call:

    for x in [a, b, c]        exactly 3
    for x in range(5)         exactly 5
    for x in orgs             unknown, and `orgs` is named so it can be declared
    while ...                 unknown
    if ...:                   0 to 1
    @retry(max=3)             1 to 3
    nested                    multiply

**Unknown bounds are named, not guessed.** `for org in orgs` yields no range and
records `orgs` instead. That name is what lets the comment say the estimate is
driven by `len(orgs)` rather than printing an invented total, and it is the key
the user declares in config to make the number real.

**A nested function resets the count.** Defining a function inside a loop does
not run its body there, and multiplicity is per invocation of the function that
holds the call. Inheriting the outer loop would multiply a cost that is never
paid.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass

from benji.estimate import Range
from benji.extract import extract

# A guarded call happens or it does not, and nothing in the source says which.
# The expected value is a maximum-entropy coin flip, not a measurement, and it
# is the first thing telemetry should replace.
GUARD = Range(0.0, 0.5, 1.0)

# Retries usually do not happen. The high is the declared attempt limit; the
# expected is a little above one, which is an assumption, not an observation.
RETRY_EXPECTED_MULTIPLIER = 1.1

# Wrappers that pass their iterable through, so the name underneath is the one
# worth reporting: `for i, org in enumerate(orgs)` is driven by `orgs`.
TRANSPARENT_ITERABLES = frozenset({"enumerate", "list", "sorted", "reversed", "tuple", "set"})

RETRY_DECORATORS = frozenset({"retry", "on_exception", "retry_with_backoff", "backoff"})

RETRY_LIMIT_KWARGS = ("max_tries", "max_attempts", "stop_after_attempt", "attempts", "tries")


@dataclass(frozen=True)
class Factor:
    """One construct multiplying a call site, resolved or not."""

    kind: str  # "loop" | "comprehension" | "guard" | "retry" | "while"
    name: str  # what to show a human: "orgs", "if ticket.is_urgent", "@retry"
    range: Range | None  # None when the bound is not knowable statically
    key: str = ""  # scoped config key, e.g. "handle_ticket:orgs"


@dataclass(frozen=True)
class Multiplicity:
    """The product of every factor enclosing a call site.

    ``range`` is ``None`` when any factor is unresolved. That is correct output,
    not a gap: a bounded-looking number invented from an unbounded loop is the
    failure this whole layer exists to avoid.
    """

    factors: tuple[Factor, ...] = ()

    @property
    def unknowns(self) -> tuple[Factor, ...]:
        return tuple(f for f in self.factors if f.range is None)

    @property
    def range(self) -> Range | None:
        if any(f.range is None for f in self.factors):
            return None
        total = Range.exact(1.0)
        for factor in self.factors:
            total = total * factor.range  # type: ignore[operator]
        return total

    @property
    def resolved(self) -> bool:
        return self.range is not None

    def describe(self) -> str:
        if not self.factors:
            return "1"
        return " x ".join(f.name if f.range is None else str(f.range) for f in self.factors)


def _literal_len(node: ast.AST) -> int | None:
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return len(node.elts)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return len(node.value)
    return None


def _range_call_len(node: ast.AST) -> int | None:
    """The length of `range(...)` when every argument is a literal."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
        return None
    if node.func.id != "range" or node.keywords:
        return None
    values = []
    for argument in node.args:
        if not (isinstance(argument, ast.Constant) and isinstance(argument.value, int)):
            return None
        values.append(argument.value)
    try:
        return len(range(*values))
    except (TypeError, ValueError):
        return None


def _unwrap(node: ast.AST) -> ast.AST:
    """See through `enumerate(orgs)` to `orgs`."""
    while (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in TRANSPARENT_ITERABLES
        and node.args
    ):
        node = node.args[0]
    return node


def loop_factor(iterable: ast.AST, scope: str, kind: str = "loop") -> Factor:
    """Resolve a loop's iterable to a count, or name it for declaration."""
    inner = _unwrap(iterable)

    size = _literal_len(inner)
    if size is None:
        size = _range_call_len(inner)
    if size is not None:
        return Factor(kind, f"{size}x", Range.exact(float(size)))

    name = ast.unparse(inner)
    return Factor(kind, name, None, key=f"{scope}:{name}" if scope else name)


def retry_factor(decorator: ast.AST) -> Factor | None:
    """A retry decorator's attempt limit, when it declares one."""
    if isinstance(decorator, ast.Call):
        path = decorator.func
        name = path.attr if isinstance(path, ast.Attribute) else getattr(path, "id", "")
        if name not in RETRY_DECORATORS:
            return None

        limit: int | None = None
        for keyword in decorator.keywords:
            if keyword.arg in RETRY_LIMIT_KWARGS:
                if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, int):
                    limit = keyword.value.value
                # tenacity: stop=stop_after_attempt(3)
                elif isinstance(keyword.value, ast.Call):
                    for argument in keyword.value.args:
                        if isinstance(argument, ast.Constant) and isinstance(argument.value, int):
                            limit = argument.value
            elif keyword.arg == "stop" and isinstance(keyword.value, ast.Call):
                for argument in keyword.value.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, int):
                        limit = argument.value

        if limit is None:
            return Factor("retry", f"@{name}", None)
        return Factor(
            "retry", f"@{name}(<={limit})", Range(1.0, RETRY_EXPECTED_MULTIPLIER, float(limit))
        )

    name = decorator.attr if isinstance(decorator, ast.Attribute) else getattr(decorator, "id", "")
    if name in RETRY_DECORATORS:
        return Factor("retry", f"@{name}", None)
    return None


class MultiplicityVisitor(ast.NodeVisitor):
    """Walks a module keeping a stack of the constructs enclosing each call."""

    def __init__(self, metered: set[int]) -> None:
        self.metered = metered
        self.stack: list[Factor] = []
        self.scope: list[str] = []
        self.results: dict[int, Multiplicity] = {}

    # --- scopes -----------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Decorators are evaluated in the enclosing scope, not inside the body.
        for decorator in node.decorator_list:
            if retry_factor(decorator) is None:
                self.visit(decorator)

        outer = self.stack
        retries = [f for f in (retry_factor(d) for d in node.decorator_list) if f is not None]
        # A new function resets the count: defining it inside a loop does not run
        # its body there, and multiplicity is per invocation of this function.
        self.stack = retries
        self.scope.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.scope.pop()
        self.stack = outer

    def visit_Lambda(self, node: ast.Lambda) -> None:
        outer = self.stack
        self.stack = []
        self.visit(node.body)
        self.stack = outer

    # --- loops ------------------------------------------------------------

    def visit_For(self, node: ast.For) -> None:
        self._loop(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._loop(node)

    def _loop(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)  # evaluated once, outside the loop
        self.stack.append(loop_factor(node.iter, ".".join(self.scope)))
        for statement in node.body:
            self.visit(statement)
        self.stack.pop()
        for statement in node.orelse:  # runs at most once
            self.visit(statement)

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        self.stack.append(Factor("while", "while loop", None))
        for statement in node.body:
            self.visit(statement)
        self.stack.pop()
        for statement in node.orelse:
            self.visit(statement)

    def _comprehension(self, node: ast.AST) -> None:
        added = 0
        for generator in node.generators:  # type: ignore[attr-defined]
            self.visit(generator.iter)
            self.stack.append(loop_factor(generator.iter, ".".join(self.scope), "comprehension"))
            added += 1
            for condition in generator.ifs:
                self.stack.append(Factor("guard", f"if {ast.unparse(condition)}", GUARD))
                added += 1
        for field_name in ("elt", "key", "value"):
            child = getattr(node, field_name, None)
            if child is not None:
                self.visit(child)
        for _ in range(added):
            self.stack.pop()

    visit_ListComp = _comprehension
    visit_SetComp = _comprehension
    visit_GeneratorExp = _comprehension
    visit_DictComp = _comprehension

    # --- guards -----------------------------------------------------------

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)  # always evaluated
        self.stack.append(Factor("guard", f"if {ast.unparse(node.test)}", GUARD))
        for statement in node.body + node.orelse:
            self.visit(statement)
        self.stack.pop()

    def visit_Try(self, node: ast.Try) -> None:
        for statement in node.body + node.finalbody:
            self.visit(statement)
        self.stack.append(Factor("guard", "except handler", GUARD))
        for handler in node.handlers:
            for statement in handler.body:
                self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)
        self.stack.pop()

    # --- the calls themselves ---------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        if node.lineno in self.metered and node.lineno not in self.results:
            self.results[node.lineno] = Multiplicity(tuple(self.stack))
        self.generic_visit(node)


def multiplicities(source: str, filename: str) -> dict[int, Multiplicity]:
    """Multiplicity for every metered call site in ``source``, keyed by line.

    Which calls are metered comes from the extractor, so there is one definition
    of "this costs money" rather than one per stage.
    """
    metered = {site.lineno for site in extract(source, filename)}
    if not metered:
        return {}
    visitor = MultiplicityVisitor(metered)
    visitor.visit(ast.parse(source, filename=filename))
    return visitor.results


def apply_declared(multiplicity: Multiplicity, declared: Mapping[str, Range]) -> Multiplicity:
    """Fill unknown loop bounds from declared sizes, by key then by name.

    This is how `for org in orgs` becomes a number: the user says how big `orgs`
    is, in the same config that declares frequency. Nothing is invented — an
    undeclared loop stays unresolved.
    """
    filled = []
    for factor in multiplicity.factors:
        if factor.range is None and (factor.key in declared or factor.name in declared):
            found = declared.get(factor.key) or declared[factor.name]
            filled.append(Factor(factor.kind, factor.name, found, factor.key))
        else:
            filled.append(factor)
    return Multiplicity(tuple(filled))
