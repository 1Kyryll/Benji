"""How often a call site runs in production. The third layer, and the moat.

Static analysis knows what a diff changed; observability knows what you already
paid. Neither knows what a change *about to merge* will cost, and that gap is the
product. Bridging it means multiplying the first two layers by a number that only
production can tell you.

**This layer is adapter-shaped, not config-shaped.** The MVP reads declared
numbers from a file because install friction is the binding constraint for a
community tool: requiring a Langfuse account and a `commit_sha` instrumentation
change before the bot says anything useful means nobody installs it. But declared
numbers are a stopgap, and the interface exists so that swapping in real
telemetry is an added implementation rather than a rewrite of everything
downstream.

**Frequency is declared at entry points, not at call sites.** Nobody knows how
often `LLMClient.chat` runs; everybody knows roughly how many tickets arrive a
day. The reverse call graph carries the declaration from the HTTP handler or the
cron job down to every call site it reaches — the same structure built in step 2
to answer what a diff cannot.

Known limitation, stated rather than hidden: propagation does not multiply by the
multiplicities along the path. If a declared entry point calls a helper inside a
loop, the call sites in that helper run more often than this reports. The result
carries a note saying so. Fixing it needs per-edge multiplicity, which is a
larger change than this step.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from benji.estimate import Range

DEFAULT_CONFIG_NAMES = ("benji.toml", ".benji.toml")


def parse_range(value: object, where: str) -> Range:
    """Read a declared quantity.

    A bare number is a certainty; a table is a range. Both are accepted because
    `5000` is what someone writes first and `{low, expected, high}` is what they
    write once they realise it varies by fifteen times across their customers.
    """
    if isinstance(value, int | float) and not isinstance(value, bool):
        return Range.exact(float(value))
    if isinstance(value, dict):
        try:
            low = float(value["low"])
            high = float(value["high"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{where}: a range needs numeric 'low' and 'high'") from exc
        expected = float(value.get("expected", (low + high) / 2))
        try:
            return Range(low, expected, high)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
    raise ValueError(f"{where}: expected a number or a table with low/expected/high")


@dataclass(frozen=True)
class Config:
    """Everything the user declares about their own system.

    Frequency and iterable sizes live in one file because they are the same kind
    of statement: a fact about production that no amount of reading the source
    can recover.
    """

    frequency: Mapping[str, Range] = field(default_factory=dict)
    iterables: Mapping[str, Range] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, path: Path) -> Config:
        data = tomllib.loads(path.read_text())
        return cls(
            frequency={
                key: parse_range(value, f"frequency.{key}")
                for key, value in data.get("frequency", {}).items()
            },
            iterables={
                key: parse_range(value, f"iterables.{key}")
                for key, value in data.get("iterables", {}).items()
            },
            path=path,
        )

    @classmethod
    def discover(cls, root: Path) -> Config:
        """Load the repository's config, or an empty one.

        A missing config is not an error. The bot still reports cost per call and
        multiplicity; it simply cannot reach dollars per day, and says so.
        """
        for name in DEFAULT_CONFIG_NAMES:
            candidate = root / name
            if candidate.is_file():
                return cls.load(candidate)
        return cls()

    @property
    def declared(self) -> bool:
        return bool(self.frequency)


@runtime_checkable
class FrequencySource(Protocol):
    """Where invocations-per-day comes from.

    One method, because everything else about a telemetry backend is its own
    business. Runtime-checkable, so a source supplied from outside is validated
    at the boundary rather than failing somewhere downstream.

    `ConfigFrequencySource` is implementation one. A Langfuse source joining
    spans to call sites through the per-commit AST index is where this is going,
    and it plugs in here without touching a caller.
    """

    name: str

    def invocations_per_day(self, function_key: str) -> Range | None: ...


@dataclass
class ConfigFrequencySource:
    """Declared numbers. Honest about being assumptions, not observations."""

    config: Config
    name: str = "declared (benji.toml)"

    def invocations_per_day(self, function_key: str) -> Range | None:
        return self.config.frequency.get(function_key)


@dataclass(frozen=True)
class FrequencyResult:
    """What a call site's traffic is, and where the claim came from."""

    range: Range | None
    entry_points: tuple[str, ...] = ()
    note: str = ""

    @property
    def resolved(self) -> bool:
        return self.range is not None


UNDECLARED = FrequencyResult(
    None,
    note="no declared entry point reaches this call site",
)

PATH_MULTIPLICITY_NOTE = (
    "does not account for multiplicity between the entry point and this call site"
)


def propagate(index, source: FrequencySource, function_key: str) -> FrequencyResult:
    """Traffic reaching ``function_key``, summed over declared entry points.

    A function is fed by every declared caller that can reach it, plus its own
    declaration if it has one. Nothing is assumed for a function no declaration
    reaches: an undeclared call site returns ``None`` and the comment says the
    frequency is unknown rather than inventing a plausible number.
    """
    contributors: list[tuple[str, Range]] = []

    own = source.invocations_per_day(function_key)
    if own is not None:
        contributors.append((function_key, own))

    for caller in sorted(index.blast_radius(function_key)):
        declared = source.invocations_per_day(caller)
        if declared is not None:
            contributors.append((caller, declared))

    if not contributors:
        return UNDECLARED

    total = contributors[0][1]
    for _, declared in contributors[1:]:
        total = total + declared

    indirect = any(key != function_key for key, _ in contributors)
    return FrequencyResult(
        range=total,
        entry_points=tuple(key for key, _ in contributors),
        note=PATH_MULTIPLICITY_NOTE if indirect else "",
    )
