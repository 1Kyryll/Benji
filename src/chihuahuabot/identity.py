"""Matching call sites across two versions of a codebase.

The bot is only useful if it can say *this* call became more expensive, which
means recognising the same call site in the base commit and the head commit. Two
schemes are available and both are wrong on their own:

* **Positional** — ``file : qualified_function : ordinal``. Survives
  reformatting and unrelated edits. Broken by a rename, a file move, or a new
  metered call inserted above an existing one.
* **Content hash** — survives renames and moves. Broken precisely when someone
  *edits the call*, which is the single most valuable thing the bot reports. It
  also gives no uniqueness: two identical calls in different functions hash the
  same.

They fail on disjoint sets, so the matcher uses both, in a fixed order, with
confidence decaying down the chain.

**When in doubt, do not guess.** The system genuinely cannot distinguish "this
was renamed" from "this was deleted and something unrelated was added" — the
evidence is identical. Guessing *renamed* wrongly drags foreign telemetry into a
confident dollar figure, silently. Guessing *new* wrongly produces a spurious
removed/added pair and lowers confidence: noisy, but visible. Always take the
visible failure. Ambiguity therefore matches nothing at all.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from chihuahuabot.extract import CallSite

# Ordered strongest first. The order is the design, not an optimisation.
MATCH_CONFIDENCE: dict[str, float] = {
    "id": 1.00,
    "content-same-file": 0.90,
    "id-edited": 0.85,
    "content-moved": 0.75,
}


def content_hash(site: CallSite) -> str:
    """A hash of the call as written, normalised for formatting.

    Includes the arguments, so a model swap changes it. That is deliberate: the
    hash is a fallback for finding a site whose *position* moved, and a call
    whose text changed is not a call that merely moved.
    """
    return hashlib.sha1(site.content.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Match:
    base: CallSite
    head: CallSite
    method: str
    confidence: float

    @property
    def edited(self) -> bool:
        return content_hash(self.base) != content_hash(self.head)

    @property
    def model_changed(self) -> bool:
        return self.base.model != self.head.model


@dataclass(frozen=True)
class SiteDiff:
    matched: tuple[Match, ...]
    added: tuple[CallSite, ...]
    removed: tuple[CallSite, ...]

    @property
    def edited(self) -> tuple[Match, ...]:
        """Matched sites whose text changed — the interesting half of a diff."""
        return tuple(m for m in self.matched if m.edited)

    @property
    def model_changes(self) -> tuple[Match, ...]:
        return tuple(m for m in self.matched if m.model_changed)

    def summary(self) -> str:
        return (
            f"{len(self.matched)} matched "
            f"({len(self.edited)} edited), "
            f"{len(self.added)} added, {len(self.removed)} removed"
        )


def _unique(candidates: list[int], still_open: set[int]) -> int | None:
    """The one open candidate, or ``None`` when there is a choice to be made.

    Two identical calls hash identically, so a content lookup can return several.
    Picking one would file real telemetry under the wrong site, and nothing
    downstream would ever show that it happened.
    """
    open_candidates = [index for index in candidates if index in still_open]
    return open_candidates[0] if len(open_candidates) == 1 else None


def match(base: Sequence[CallSite], head: Sequence[CallSite]) -> SiteDiff:
    """Pair up call sites between two versions.

    Four passes, strongest evidence first. Each pass only considers sites no
    earlier pass has claimed.
    """
    base_open = set(range(len(base)))
    head_open = set(range(len(head)))
    matches: list[Match] = []

    def claim(base_index: int, head_index: int, method: str) -> None:
        base_open.discard(base_index)
        head_open.discard(head_index)
        matches.append(
            Match(
                base=base[base_index],
                head=head[head_index],
                method=method,
                confidence=MATCH_CONFIDENCE[method],
            )
        )

    def group(indices: set[int], key) -> dict:
        buckets: dict = {}
        for index in indices:
            buckets.setdefault(key(base[index]), []).append(index)
        return buckets

    # Pass 1 — same identifier, same text. Nothing to argue about.
    by_id = group(base_open, lambda s: s.id)
    for head_index in sorted(head_open):
        candidate = _unique(by_id.get(head[head_index].id, []), base_open)
        if candidate is not None and content_hash(base[candidate]) == content_hash(
            head[head_index]
        ):
            claim(candidate, head_index, "id")

    # Pass 2 — same text, same file. This is what catches a renamed function, and
    # a new metered call inserted above an existing one: the ordinals all shift,
    # so the identifiers lie, but the text of the untouched call did not move.
    by_file_content = group(base_open, lambda s: (s.file, content_hash(s)))
    for head_index in sorted(head_open):
        key = (head[head_index].file, content_hash(head[head_index]))
        candidate = _unique(by_file_content.get(key, []), base_open)
        if candidate is not None:
            claim(candidate, head_index, "content-same-file")

    # Pass 3 — same identifier, different text. Somebody edited this call, which
    # is the single most valuable thing the bot reports.
    by_id = group(base_open, lambda s: s.id)
    for head_index in sorted(head_open):
        candidate = _unique(by_id.get(head[head_index].id, []), base_open)
        if candidate is not None:
            claim(candidate, head_index, "id-edited")

    # Pass 4 — same text, different file. A move, and the weakest claim here.
    by_content = group(base_open, content_hash)
    for head_index in sorted(head_open):
        candidate = _unique(by_content.get(content_hash(head[head_index]), []), base_open)
        if candidate is not None:
            claim(candidate, head_index, "content-moved")

    return SiteDiff(
        matched=tuple(matches),
        added=tuple(head[i] for i in sorted(head_open)),
        removed=tuple(base[i] for i in sorted(base_open)),
    )
