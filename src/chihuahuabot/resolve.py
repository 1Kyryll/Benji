"""Adapter between the analyser and the coverage harness.

The harness owns the denominator and must not know how the analyser works, so
this is the entire contract between them: given a candidate the harness found,
say whether we can turn it into a priced call site.

It lives here rather than in ``corpus/`` because it grows as the analyser does.
At build step 1 it answers from strict detectors alone; at step 2 the wrapper
index will let it follow ``self.llm.chat()`` down to an SDK call, and the number
it reports will move without the denominator shifting underneath it.
"""

from __future__ import annotations

from chihuahuabot.extract import extract


class CorpusResolver:
    name = "chihuahuabot.extract (build step 1: strict detectors, no wrapper index)"

    def __init__(self) -> None:
        # Extraction is per-file, the harness asks per-candidate. Cache on the
        # source itself so two repos sharing a relative path cannot collide.
        self._cache: dict[tuple[str, int], set[int]] = {}

    def _lines(self, file: str, source: str) -> set[int]:
        key = (file, hash(source))
        cached = self._cache.get(key)
        if cached is None:
            try:
                cached = {site.lineno for site in extract(source, file)}
            except (SyntaxError, ValueError, RecursionError):
                cached = set()
            self._cache[key] = cached
        return cached

    def resolve(self, candidate, source: str) -> str | None:
        """Return a truthy marker when this candidate becomes a real call site.

        Matching is by line number, which is exactly the thing call-site identity
        refuses to use. That is fine and deliberate: harness and analyser read the
        same bytes at the same instant, so there is no version skew for a line
        number to be wrong about. Identity has to survive across commits; this
        comparison does not.
        """
        return candidate.file if candidate.lineno in self._lines(candidate.file, source) else None
