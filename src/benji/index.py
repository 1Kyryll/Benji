"""Wrapper index, call graph, and reverse call graph for one repository.

Real applications do not call the SDK inline. They wrap it once — in a client
class, a service layer, a dependency-injection container — and then call the
wrapper from everywhere. Two consequences follow, and this module exists for
both of them.

**Coverage.** ``self.llm.chat()`` is a metered call and looks like nothing. The
index follows it down to whatever eventually spends money, and the confidence of
that answer decays with every hop, because a depth-1 wrapper is near-certain and
a depth-4 chain through a name lookup is a guess.

**Blast radius.** Editing ``LLMClient.chat`` changes the cost of every call site
that reaches it, across files the diff never touches. Analysing only changed
files would report zero impact on the most expensive pull request of the quarter.
The reverse graph is what prevents that — and the same structure later carries
declared frequency from entry points down to individual call sites.

Resolution walks toward callees and stops on four conditions, per the design:
a metered call (success), an already-visited node (cycles), the depth limit, or
leaving the repository — we never parse ``site-packages``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from benji.extract import WrapperHint, attr_path, extract

MAX_DEPTH = 5

# How much each resolution strategy is trusted, applied once per hop. These
# started as invented numbers; the corpus is what turns them into measurements,
# which is the entire reason it was built first.
STRATEGY_CONFIDENCE: dict[str, float] = {
    "direct": 1.00,
    "annotation": 0.95,
    "import": 0.95,
    "same-module": 0.95,
    "init-assign": 0.90,
    "local-assign": 0.85,
    "module-singleton": 0.85,
    "self-method": 0.95,
    "unique-name": 0.60,
}


@dataclass(frozen=True)
class Resolution:
    """What a non-obvious call turned out to cost, and how sure we are."""

    provider: str
    api: str
    depth: int
    confidence: float
    via: tuple[str, ...]
    strategies: tuple[str, ...]

    def hint(self) -> WrapperHint:
        return WrapperHint(
            provider=self.provider,
            api=self.api,
            depth=self.depth,
            confidence=self.confidence,
        )


@dataclass(frozen=True)
class CallRef:
    lineno: int
    path: tuple[str, ...]
    metered: bool
    provider: str = ""
    api: str = ""


@dataclass
class FuncInfo:
    key: str
    module: str
    qualname: str
    file: str
    lineno: int
    class_name: str | None = None
    calls: list[CallRef] = field(default_factory=list)
    local_types: dict[str, str] = field(default_factory=dict)


@dataclass
class ClassInfo:
    key: str
    module: str
    name: str
    # attribute -> (class simple name, strategy that told us)
    attributes: dict[str, tuple[str, str]] = field(default_factory=dict)


@dataclass
class ModuleInfo:
    dotted: str
    file: str
    imports: dict[str, str] = field(default_factory=dict)
    singletons: dict[str, tuple[str, str]] = field(default_factory=dict)


def module_path(root: Path, file: Path) -> str:
    relative = file.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def constructed_class(node: ast.AST) -> str | None:
    """The class simple name for ``X()`` or ``module.X()``, else ``None``."""
    if isinstance(node, ast.Call):
        path = attr_path(node.func)
        if path and path[-1][:1].isupper():
            return path[-1]
    return None


class _ModuleScanner(ast.NodeVisitor):
    """One pass over one module, recording everything the resolver will need."""

    def __init__(self, module: ModuleInfo) -> None:
        self.module = module
        self.classes: dict[str, ClassInfo] = {}
        self.functions: dict[str, FuncInfo] = {}
        self._scope: list[str] = []
        self._class_stack: list[ClassInfo] = []
        self._func_stack: list[FuncInfo] = []

    # --- imports ---------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.module.imports[alias.asname or alias.name.split(".")[0]] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # Relative imports are not resolved: without a package root the dots are
        # ambiguous, and a wrong module is worse than an unresolved one.
        if node.level:
            return
        for alias in node.names:
            target = f"{node.module}.{alias.name}" if node.module else alias.name
            self.module.imports[alias.asname or alias.name] = target

    # --- definitions -----------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        info = ClassInfo(
            key=f"{self.module.dotted}:{node.name}",
            module=self.module.dotted,
            name=node.name,
        )
        self.classes[info.key] = info

        # Class-level annotations: `llm: LLMClient`.
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                annotation = attr_path(stmt.annotation)
                if annotation:
                    info.attributes[stmt.target.id] = (annotation[-1], "annotation")

        self._scope.append(node.name)
        self._class_stack.append(info)
        self.generic_visit(node)
        self._class_stack.pop()
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._scope.append(node.name)
        qualname = ".".join(self._scope)
        info = FuncInfo(
            key=f"{self.module.dotted}:{qualname}",
            module=self.module.dotted,
            qualname=qualname,
            file=self.module.file,
            lineno=node.lineno,
            class_name=self._class_stack[-1].name if self._class_stack else None,
        )
        self.functions[info.key] = info

        if node.name == "__init__" and self._class_stack:
            self._scan_init(node, self._class_stack[-1])

        # Parameter annotations give local types for free.
        for arg in list(node.args.args) + list(node.args.kwonlyargs):
            if arg.annotation is not None:
                annotation = attr_path(arg.annotation)
                if annotation:
                    info.local_types[arg.arg] = annotation[-1]

        self._func_stack.append(info)
        self.generic_visit(node)
        self._func_stack.pop()
        self._scope.pop()

    def _scan_init(self, node: ast.AST, klass: ClassInfo) -> None:
        """Learn attribute types from ``__init__``.

        Two shapes carry evidence: an annotated parameter passed straight through
        (``def __init__(self, llm: LLMClient)`` then ``self.llm = llm``), and a
        construction in place (``self.llm = LLMClient()``).
        """
        annotated: dict[str, str] = {}
        for arg in list(node.args.args) + list(node.args.kwonlyargs):  # type: ignore[attr-defined]
            if arg.annotation is not None:
                annotation = attr_path(arg.annotation)
                if annotation:
                    annotated[arg.arg] = annotation[-1]

        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if not (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    continue
                built = constructed_class(stmt.value)
                if built:
                    klass.attributes[target.attr] = (built, "init-assign")
                elif isinstance(stmt.value, ast.Name) and stmt.value.id in annotated:
                    klass.attributes[target.attr] = (annotated[stmt.value.id], "annotation")

    # --- assignments and calls -------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        built = constructed_class(node.value)
        if built:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if self._func_stack:
                        self._func_stack[-1].local_types[target.id] = built
                    elif not self._class_stack:
                        self.module.singletons[target.id] = (built, "module-singleton")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            annotation = attr_path(node.annotation)
            if annotation:
                if self._func_stack:
                    self._func_stack[-1].local_types[node.target.id] = annotation[-1]
                elif not self._class_stack:
                    self.module.singletons[node.target.id] = (annotation[-1], "annotation")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._func_stack:
            path = attr_path(node.func)
            if path:
                self._func_stack[-1].calls.append(CallRef(node.lineno, path, metered=False))
        self.generic_visit(node)


class RepoIndex:
    """Everything known about one repository at one commit.

    Built per run rather than cached per commit on main. The design calls for a
    persistent index, but a GitHub Action has nowhere to persist it, and parsing
    a repository with stdlib ``ast`` costs seconds. Caching is an optimisation to
    add with the service, not a prerequisite.
    """

    def __init__(self) -> None:
        self.modules: dict[str, ModuleInfo] = {}
        self.classes: dict[str, ClassInfo] = {}
        self.functions: dict[str, FuncInfo] = {}
        self.sources: dict[str, str] = {}
        self.classes_by_name: dict[str, list[str]] = {}
        self.functions_by_method: dict[str, list[str]] = {}
        self.reverse: dict[str, set[str]] = {}
        self._resolved: dict[str, Resolution | None] = {}

    # --- construction ----------------------------------------------------

    @classmethod
    def build(cls, root: Path, files: list[Path]) -> RepoIndex:
        index = cls()
        for file in files:
            try:
                source = file.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=str(file))
            except (SyntaxError, ValueError, OSError, RecursionError):
                continue

            relative = str(file.relative_to(root))
            module = ModuleInfo(dotted=module_path(root, file), file=relative)
            index.modules[module.dotted] = module
            index.sources[relative] = source

            scanner = _ModuleScanner(module)
            scanner.visit(tree)
            index.classes.update(scanner.classes)
            index.functions.update(scanner.functions)

            # Direct metered calls come from the extractor, so there is exactly
            # one definition of "this call costs money" in the system.
            #
            # Framework-shaped calls are excluded here. `self.llm.chat()` looks
            # like one, and if it were treated as metered the walk would stop at
            # the guess instead of following the wrapper to the certainty
            # underneath it. The heuristic is a fallback for calls the index
            # cannot resolve, so it is applied last, in `call_sites`.
            try:
                metered = {
                    site.lineno: site for site in extract(source, relative, include_framework=False)
                }
            except (SyntaxError, ValueError, RecursionError):
                metered = {}
            for info in scanner.functions.values():
                info.calls = [
                    CallRef(
                        call.lineno,
                        call.path,
                        metered=call.lineno in metered,
                        provider=metered[call.lineno].provider if call.lineno in metered else "",
                        api=metered[call.lineno].api if call.lineno in metered else "",
                    )
                    for call in info.calls
                ]

        index._build_lookups()
        index._build_reverse_graph()
        return index

    def _build_lookups(self) -> None:
        for key, klass in self.classes.items():
            self.classes_by_name.setdefault(klass.name, []).append(key)
        for key, func in self.functions.items():
            self.functions_by_method.setdefault(func.qualname.split(".")[-1], []).append(key)

    def _build_reverse_graph(self) -> None:
        for key, func in self.functions.items():
            for call in func.calls:
                target = self._target_of(func, call)
                if target is not None:
                    self.reverse.setdefault(target[0], set()).add(key)

    # --- name resolution --------------------------------------------------

    def _class_key(self, module: str, simple_name: str) -> tuple[str | None, str]:
        info = self.modules.get(module)
        if info is not None:
            imported = info.imports.get(simple_name)
            if imported:
                owner, _, name = imported.rpartition(".")
                candidate = f"{owner}:{name}"
                if candidate in self.classes:
                    return candidate, "import"
            same = f"{module}:{simple_name}"
            if same in self.classes:
                return same, "same-module"

        matches = self.classes_by_name.get(simple_name, [])
        if len(matches) == 1:
            return matches[0], "unique-name"
        return None, ""

    def _method_key(self, class_key: str, method: str) -> str | None:
        klass = self.classes[class_key]
        candidate = f"{klass.module}:{klass.name}.{method}"
        return candidate if candidate in self.functions else None

    def _target_of(self, func: FuncInfo, call: CallRef) -> tuple[str, str] | None:
        """Resolve a call to ``(function key, strategy)``, or ``None``."""
        path = call.path
        if not path:
            return None
        method = path[-1]

        # self.method(...)
        if len(path) == 2 and path[0] == "self" and func.class_name:
            class_key, _ = self._class_key(func.module, func.class_name)
            if class_key:
                target = self._method_key(class_key, method)
                if target:
                    return target, "self-method"

        # self.<attr>.method(...)
        if len(path) == 3 and path[0] == "self" and func.class_name:
            owner_key, _ = self._class_key(func.module, func.class_name)
            if owner_key:
                attribute = self.classes[owner_key].attributes.get(path[1])
                if attribute:
                    simple, strategy = attribute
                    class_key, _ = self._class_key(func.module, simple)
                    if class_key:
                        target = self._method_key(class_key, method)
                        if target:
                            return target, strategy

        # <local or singleton>.method(...)
        if len(path) == 2:
            receiver = path[0]
            found = func.local_types.get(receiver)
            strategy = "local-assign"
            if found is None:
                module = self.modules.get(func.module)
                if module and receiver in module.singletons:
                    found, strategy = module.singletons[receiver]
            if found:
                class_key, _ = self._class_key(func.module, found)
                if class_key:
                    target = self._method_key(class_key, method)
                    if target:
                        return target, strategy

        # bare_function(...)
        if len(path) == 1:
            same = f"{func.module}:{method}"
            if same in self.functions:
                return same, "same-module"
            module = self.modules.get(func.module)
            imported = module.imports.get(method) if module else None
            if imported:
                owner, _, name = imported.rpartition(".")
                candidate = f"{owner}:{name}"
                if candidate in self.functions:
                    return candidate, "import"

        # Last resort. A method name unique across the whole repository is
        # evidence, but weak evidence, and it is priced accordingly.
        matches = self.functions_by_method.get(method, [])
        if len(matches) == 1:
            return matches[0], "unique-name"

        return None

    # --- resolution -------------------------------------------------------

    def resolve_function(self, key: str, depth: int = 0, stack: frozenset[str] = frozenset()):
        """The cheapest metered call this function reaches, or ``None``.

        Stops on a metered call, a cycle, the depth limit, or an unresolvable
        target. Confidence multiplies down the chain.
        """
        if key in stack or depth > MAX_DEPTH:
            return None
        if depth == 0 and key in self._resolved:
            return self._resolved[key]

        func = self.functions.get(key)
        if func is None:
            return None

        best: Resolution | None = None
        for call in func.calls:
            found = self._resolve_call(func, call, depth, stack | {key})
            if found is not None and (
                best is None
                or found.depth < best.depth
                or (found.depth == best.depth and found.confidence > best.confidence)
            ):
                best = found

        if depth == 0:
            self._resolved[key] = best
        return best

    def _resolve_call(
        self, func: FuncInfo, call: CallRef, depth: int, stack: frozenset[str]
    ) -> Resolution | None:
        if call.metered:
            return Resolution(
                provider=call.provider,
                api=call.api,
                depth=depth,
                confidence=1.0,
                via=(func.key,),
                strategies=("direct",),
            )

        target = self._target_of(func, call)
        if target is None:
            return None

        target_key, strategy = target
        deeper = self.resolve_function(target_key, depth + 1, stack)
        if deeper is None:
            return None

        factor = STRATEGY_CONFIDENCE.get(strategy, 0.5)
        # `deeper` was resolved at depth+1, so the hop is already counted. Adding
        # to it again would double-count every link in the chain.
        return Resolution(
            provider=deeper.provider,
            api=deeper.api,
            depth=deeper.depth,
            confidence=deeper.confidence * factor,
            via=(func.key,) + deeper.via,
            strategies=(strategy,) + deeper.strategies,
        )

    def wrapper_hints(self, file: str) -> dict[int, WrapperHint]:
        """Lines in ``file`` that are metered only because of what they call."""
        hints: dict[int, WrapperHint] = {}
        for func in self.functions.values():
            if func.file != file:
                continue
            for call in func.calls:
                if call.metered:
                    continue
                target = self._target_of(func, call)
                if target is None:
                    continue
                deeper = self.resolve_function(target[0], 1, frozenset({func.key}))
                if deeper is None:
                    continue
                factor = STRATEGY_CONFIDENCE.get(target[1], 0.5)
                hints[call.lineno] = WrapperHint(
                    provider=deeper.provider,
                    api=deeper.api,
                    depth=deeper.depth,
                    confidence=deeper.confidence * factor,
                )
        return hints

    def call_sites(self, file: str):
        """Every metered call site in ``file``, direct and wrapped, with ordinals.

        Ordinals come from the extractor so there is one implementation of
        identity. Note the consequence: an ordinal counts the calls *this version
        of Benji recognises*, so improving the wrapper index can renumber
        sites. That is a versioning concern for the telemetry index, not a bug —
        and it is why the AST index is stamped per commit.
        """
        source = self.sources.get(file)
        if source is None:
            return []
        return extract(source, file, self.wrapper_hints(file))

    def blast_radius(self, key: str) -> set[str]:
        """Every function that transitively reaches ``key``.

        This is what turns "four lines changed" into "47 call sites across 12
        files". Editing a wrapper is invisible to a diff and expensive in
        production, and those are the pull requests worth commenting on.
        """
        seen: set[str] = set()
        queue = [key]
        while queue:
            current = queue.pop()
            for caller in self.reverse.get(current, ()):  # noqa: B007
                if caller not in seen:
                    seen.add(caller)
                    queue.append(caller)
        return seen
