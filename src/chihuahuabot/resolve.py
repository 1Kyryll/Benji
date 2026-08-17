"""Adapter between the analyser and the coverage harness.

The harness owns the denominator and must not know how the analyser works, so
this is the entire contract between them: given a candidate the harness found,
say whether we can turn it into a priced call site.

It lives here rather than in ``corpus/`` because it grows as the analyser does.
At build step 1 it answered from strict detectors alone. It now builds a whole
repository index first, so a call like ``self.llm.chat()`` resolves through the
wrapper that owns it — and the number moves without the denominator shifting
underneath it.
"""

from __future__ import annotations

from pathlib import Path

from chihuahuabot.extract import extract
from chihuahuabot.index import RepoIndex


class CorpusResolver:
    name = "chihuahuabot.index (build step 2: wrapper index + reverse call graph)"

    def __init__(self) -> None:
        self._index: RepoIndex | None = None
        self._root: Path | None = None
        self._lines: dict[str, set[int]] = {}

    def begin_repo(self, root: Path, files: list[Path]) -> None:
        """Build the index once per repository.

        The harness calls this before scoring a repository's candidates. Without
        it the resolver would rebuild the index per candidate, which for a repo
        the size of onyx means parsing 1,885 files forty times over.
        """
        self._root = root
        self._lines = {}
        try:
            self._index = RepoIndex.build(root, files)
        except (RecursionError, MemoryError):
            # A pathological repository must degrade to step-1 behaviour rather
            # than take the whole scoring run down with it.
            self._index = None

    def _resolved_lines(self, file: str, source: str) -> set[int]:
        cached = self._lines.get(file)
        if cached is not None:
            return cached

        try:
            if self._index is not None and file in self._index.sources:
                sites = self._index.call_sites(file)
            else:
                sites = extract(source, file)
            cached = {site.lineno for site in sites}
        except (SyntaxError, ValueError, RecursionError):
            cached = set()

        self._lines[file] = cached
        return cached

    def resolve(self, candidate, source: str) -> str | None:
        """Return a truthy marker when this candidate becomes a real call site.

        Matching is by line number, which is exactly the thing call-site identity
        refuses to use. That is fine and deliberate: harness and analyser read the
        same bytes at the same instant, so there is no version skew for a line
        number to be wrong about. Identity has to survive across commits; this
        comparison does not.
        """
        return (
            candidate.file
            if candidate.lineno in self._resolved_lines(candidate.file, source)
            else None
        )
