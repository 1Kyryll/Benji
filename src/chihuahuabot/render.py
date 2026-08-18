"""The PR comment. The only part of this system anyone actually reads.

Every design decision in the layers underneath exists to make three statements
possible here, and the rendering is what decides whether they land:

1. **A range, never a point.** A confident `+$127.40/mo` that is wrong by ten
   times is the failure that kills a tool like this.
2. **The dominant uncertainty, named.** `$40 to $900` is noise. "The range is
   driven by `len(orgs)`, which is undeclared" is an instruction.
3. **Coverage, always.** "Analysed 34 of 51 candidate call sites" turns a miss
   from silent into visible, which was the whole point of tiering the output.

The comment is found and updated in place by a hidden marker, so a branch with
twenty pushes gets one comment rather than twenty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chihuahuabot.estimate import Factor, Range, SiteEstimate

MARKER = "<!-- chihuahuabot-comment -->"
TITLE = "## 💸 ChihuahuaBot — cost impact"

# Below this, a range is tight enough that naming its dominant factor is noise.
NOTEWORTHY_SPREAD = 2.0


def money(value: float) -> str:
    """Dollars at a precision that does not overstate what we know."""
    magnitude = abs(value)
    if magnitude == 0:
        return "$0"
    if magnitude < 0.01:
        return f"${value:.5f}".rstrip("0")
    if magnitude < 10:
        return f"${value:,.2f}"
    return f"${value:,.0f}"


def signed(value: float) -> str:
    return ("+" if value > 0 else "−" if value < 0 else "") + money(abs(value))


def money_range(span: Range, sign: bool = False) -> str:
    """A range rendered as a range. There is no branch here that prints one number."""
    render = signed if sign else money
    if span.is_point:
        return render(span.expected)
    return f"{render(span.low)} to {render(span.high)}"


def count(value: float) -> str:
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.1f}"


def multiplier(span: Range | None, label: str | None) -> str:
    """How a multiplicity reads in a table cell."""
    if span is None:
        return f"×`{label}`" if label else "×?"
    if span.is_point and span.expected == 1:
        return "×1"
    if span.is_point:
        return f"×{count(span.expected)}"
    return f"×{count(span.low)}–{count(span.high)}"


@dataclass(frozen=True)
class SiteChange:
    """One call site's story in this diff."""

    kind: str  # "added" | "removed" | "edited" | "affected"
    site_id: str
    detail: str = ""
    estimate: SiteEstimate | None = None
    multiplicity_label: str | None = None
    frequency: Range | None = None
    confidence: float = 1.0

    @property
    def label(self) -> str:
        return {"added": "**new**", "removed": "**removed**", "edited": "", "affected": ""}.get(
            self.kind, ""
        )


@dataclass(frozen=True)
class Coverage:
    """Decision four, in one line. Never omitted, even when it flatters nobody."""

    analysed: int
    candidates: int
    reasons: tuple[tuple[str, int], ...] = ()

    @property
    def unresolved(self) -> int:
        return max(0, self.candidates - self.analysed)

    def line(self) -> str:
        return f"Coverage: {self.analysed} of {self.candidates} candidate call sites analysed"


@dataclass(frozen=True)
class BlastRadius:
    """The headline nobody else can produce: a diff that is not local.

    ``sites`` counts **metered call sites** whose cost this change moves, not
    the functions that reach it. Those are different numbers and conflating them
    produced a headline claiming twenty-four call sites in a repository that
    contains one. ``callers`` keeps the function count, which is real and
    interesting, under a name that says what it is.
    """

    function: str
    sites: int
    files: int
    callers: int = 0
    changed_lines: int = 0

    @property
    def worth_saying(self) -> bool:
        """Only when the change genuinely reaches beyond its own call site."""
        return self.sites > 1


@dataclass(frozen=True)
class Report:
    """Everything the comment needs. Assembled by the caller; rendered here."""

    delta: Range | None = None
    changes: tuple[SiteChange, ...] = ()
    dominant: Factor | None = None
    # The spread of the cost model behind the estimate, not of the delta. A
    # delta that crosses zero — a change that might save money or might cost far
    # more — has no meaningful high-over-low ratio, and asking for one produces
    # a negative number that means nothing.
    total_spread: float = 1.0
    coverage: Coverage | None = None
    blast: BlastRadius | None = None
    prices_as_of: str = "unknown"
    prices_stale: bool = False
    frequency_source: str = ""
    config_present: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def empty(self) -> bool:
        """Nothing metered changed. The caller decides whether to stay silent."""
        return not self.changes


def headline(report: Report) -> str:
    if report.delta is None:
        return (
            "**Cost impact unknown.** Call sites changed, but at least one layer "
            "could not be resolved — see the table for which."
        )
    span = report.delta
    if span.is_point and span.expected == 0:
        return "**No change in predicted cost.**"
    return f"**{money_range(span, sign=True)} / month** · expected **{signed(span.expected)}**"


def table(report: Report) -> str:
    if not report.changes:
        return ""
    lines = ["| call site | change |", "|---|---|"]
    for change in report.changes:
        parts = [part for part in (change.label, change.detail) if part]
        estimate = change.estimate
        if estimate is not None and estimate.cost_per_call is not None:
            parts.append(f"{money(estimate.cost_per_call.expected)}/call")
            parts.append(multiplier(estimate.multiplicity, change.multiplicity_label))
            if estimate.frequency is not None:
                parts.append(f"{count(estimate.frequency.expected)}/day")
            else:
                parts.append("frequency undeclared")
        elif estimate is not None:
            parts.append("_unpriced: " + ", ".join(estimate.missing) + "_")
        if change.confidence < 1.0:
            parts.append(f"conf {change.confidence:.0%}")
        site = change.site_id.replace("|", "\\|")
        lines.append(f"| `{site}` | {' · '.join(parts)} |")
    return "\n".join(lines)


def uncertainty(report: Report) -> str:
    """The sentence the tool exists for."""
    factor = report.dominant
    if factor is None:
        return ""
    spread = report.total_spread
    # A non-finite or non-positive spread is not a wide range, it is a spread
    # that was never meaningful. Saying nothing beats saying nonsense.
    if not (NOTEWORTHY_SPREAD <= spread < float("inf")):
        return ""

    narrowed = report.total_spread / factor.range.spread if factor.range.spread > 0 else None
    lead = f"> **The range is driven almost entirely by `{factor.name}`**"
    if factor.source == "prior":
        lead += ", which is undeclared."
    else:
        lead += f", declared as {count(factor.range.low)}–{count(factor.range.high)}."

    if narrowed is not None and narrowed >= 1.0:
        lead += (
            f"\n> Pinning it narrows this from {report.total_spread:.0f}× to about {narrowed:.1f}×."
        )
    if factor.source == "prior":
        lead += "\n> Declare it under `[iterables]` or `[frequency]` in `chihuahuabot.toml`."
    return lead


def footer(report: Report) -> str:
    bits = [f"prices as of {report.prices_as_of}"]
    if report.frequency_source:
        bits.append(f"frequency: {report.frequency_source}")
    line = f"<sub>{' · '.join(bits)} · {MARKER}</sub>"
    if report.prices_stale:
        line = (
            f"> ⚠️ The price table was last checked on {report.prices_as_of} "
            "and may be out of date.\n\n" + line
        )
    return line


def render(report: Report) -> str:
    """The whole comment.

    Ordered by what a reader needs first: the number, then what it applies to,
    then why it is uncertain, then how far the change reaches, then the coverage
    that keeps all of it honest.
    """
    blocks: list[str] = [TITLE, headline(report)]

    body = table(report)
    if body:
        blocks.append(body)

    driver = uncertainty(report)
    if driver:
        blocks.append(driver)

    if not report.config_present:
        blocks.append(
            "> No `chihuahuabot.toml` found, so nothing declares how often this code runs. "
            "Cost per call and multiplicity are still exact; dollars per day are not available."
        )

    if report.blast is not None and report.blast.worth_saying:
        blast = report.blast
        change = f"This {blast.changed_lines}-line change" if blast.changed_lines else "This change"
        blocks.append(
            f"⚠️ {change} to `{blast.function}` affects "
            f"**{blast.sites} call sites across {blast.files} files**."
        )

    for note in report.notes:
        blocks.append(f"> {note}")

    if report.coverage is not None:
        detail = ""
        if report.coverage.reasons:
            listed = ", ".join(f"{n} {reason}" for reason, n in report.coverage.reasons)
            detail = f"\n{report.coverage.unresolved} unresolved — {listed}.\n"
        blocks.append(f"<details><summary>{report.coverage.line()}</summary>\n{detail}</details>")

    blocks.append(footer(report))
    return "\n\n".join(blocks) + "\n"
