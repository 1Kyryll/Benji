"""Ranges, and the arithmetic that carries uncertainty through the pipeline.

Every number in this system is a triple. The bot never prints a point estimate,
because fake precision is what kills a tool like this: "+$127.40/mo", wrong by a
factor of ten, is worse than saying nothing. A range that admits what it does
not know survives being checked.

Multiplication is deliberately naive — lows by lows, highs by highs, no
statistics. Correlations between factors are unknowable here and pretending
otherwise would dress a guess as a calculation.

Build step 5 needs the type. Propagation across the three layers and the
dominant-uncertainty calculation arrive in step 7.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Range:
    """A low, expected and high value for one quantity."""

    low: float
    expected: float
    high: float

    def __post_init__(self) -> None:
        if not (self.low <= self.expected <= self.high):
            raise ValueError(f"range out of order: {self.low}, {self.expected}, {self.high}")

    @classmethod
    def exact(cls, value: float) -> Range:
        """A quantity we know. Still a range, so nothing downstream special-cases it."""
        return cls(value, value, value)

    def __mul__(self, other: Range) -> Range:
        return Range(
            self.low * other.low,
            self.expected * other.expected,
            self.high * other.high,
        )

    def __add__(self, other: Range) -> Range:
        """Combine two independent contributions to the same quantity.

        Used where several declared entry points reach one call site: the traffic
        is the sum of theirs. Adding lows to lows is as naive as multiplying them
        and for the same reason — the alternative is inventing a correlation.
        """
        return Range(
            self.low + other.low,
            self.expected + other.expected,
            self.high + other.high,
        )

    def scale(self, factor: float) -> Range:
        """Multiply by a certainty. Negative factors would invert the range."""
        if factor < 0:
            raise ValueError("ranges scale by non-negative factors only")
        return Range(self.low * factor, self.expected * factor, self.high * factor)

    @property
    def is_point(self) -> bool:
        return self.low == self.high

    @property
    def spread(self) -> float:
        """How many times wider the high is than the low.

        This is the number that decides which factor gets blamed for an
        unhelpful range. A zero low means unbounded: a guard can make a call
        never happen, and no multiplier rescues that.
        """
        if self.low == 0:
            return float("inf") if self.high > 0 else 1.0
        return self.high / self.low

    def __str__(self) -> str:
        if self.is_point:
            return f"{self.expected:g}"
        return f"{self.low:g}..{self.high:g} (exp {self.expected:g})"


# ---------------------------------------------------------------------------
# Propagation and dominant uncertainty
#
# This is the highest-value output in the product. "$40 to $900" is useless and
# slightly insulting. "The range is driven almost entirely by tickets-per-org,
# which varies fifteen times across your customers" is a sentence someone
# installs a bot to read.
#
# Everything below is pure arithmetic on ranges, with no dependency on the
# tokenizer, the price table or the AST. The layers hand it numbers; it decides
# what those numbers mean and which one to blame.
# ---------------------------------------------------------------------------

# What an interpolated value costs when nobody has told us and no telemetry has
# been observed. Deliberately wide: a prior pretending to be narrow is how a
# tool starts lying. Declaring the real distribution is the fix, and naming this
# as the dominant uncertainty is how the user finds out they should.
DEFAULT_SLOT_PRIOR = Range(20, 200, 4000)

# Output length follows the instruction, not the data, so a sibling call site
# tells you nothing about it. Best available evidence is observed telemetry,
# then sampling the call in CI, then these priors by task shape.
#
# max_tokens is a ceiling, not a prediction — a classifier told to answer in one
# word against max_tokens=50 overshoots by twenty times — so the fractions below
# stay well under it.
OUTPUT_FRACTION_OF_CEILING: dict[str, Range] = {
    "classifier": Range(0.01, 0.05, 0.30),
    "extractor": Range(0.02, 0.10, 0.50),
    "summariser": Range(0.10, 0.40, 0.90),
    "chat": Range(0.10, 0.60, 1.00),
}

# Used when the call declares no ceiling at all.
OUTPUT_ABSOLUTE: dict[str, Range] = {
    "classifier": Range(1, 3, 10),
    "extractor": Range(10, 80, 500),
    "summariser": Range(40, 300, 1200),
    "chat": Range(50, 400, 2000),
}

DEFAULT_SHAPE = "chat"


def input_range(
    static: int,
    unknown_slots: int,
    slot_prior: Range = DEFAULT_SLOT_PRIOR,
) -> Range:
    """Input tokens: what we counted, plus a prior for each hole.

    The static half is exact and contributes no width at all. Every unit of
    uncertainty here comes from an interpolated value, which is exactly the
    story the comment should tell.
    """
    total = Range.exact(float(static))
    for _ in range(unknown_slots):
        total = total + slot_prior
    return total


def output_range(max_tokens: int | None = None, shape: str = DEFAULT_SHAPE) -> Range:
    """Output tokens, from a prior by task shape.

    Deterministic code picks the prior; an LLM may classify which shape a call
    is, because that is judgement about ambiguity rather than a figure. It must
    never produce the number itself.
    """
    if max_tokens is not None and max_tokens > 0:
        fraction = OUTPUT_FRACTION_OF_CEILING.get(shape, OUTPUT_FRACTION_OF_CEILING[DEFAULT_SHAPE])
        return fraction.scale(float(max_tokens))
    return OUTPUT_ABSOLUTE.get(shape, OUTPUT_ABSOLUTE[DEFAULT_SHAPE])


def cost_range(
    input_per_mtok: float,
    output_per_mtok: float,
    input_tokens: Range,
    output_tokens: Range,
) -> Range:
    """Dollars for one call, carrying the token uncertainty through."""
    return (input_tokens.scale(input_per_mtok) + output_tokens.scale(output_per_mtok)).scale(
        1 / 1_000_000
    )


@dataclass(frozen=True)
class Factor:
    """One source of uncertainty, named so it can be blamed."""

    name: str
    layer: str  # "cost" | "multiplicity" | "frequency"
    range: Range
    source: str = "prior"  # "static" | "declared" | "observed" | "prior"

    @property
    def certain(self) -> bool:
        return self.range.is_point

    @property
    def conditional(self) -> bool:
        """True when this factor alone can drive the whole estimate to zero."""
        return self.range.low == 0 and self.range.high > 0


def product(ranges) -> Range:
    total = Range.exact(1.0)
    for item in ranges:
        total = total * item
    return total


def dominant(factors) -> Factor | None:
    """The factor whose uncertainty drives the range most.

    Defined by pinning: fix each factor at its expected value in turn, recompute,
    and see which pin collapses the spread furthest. Under purely multiplicative
    propagation that reduces to "the widest factor", and there is a test saying
    so — but the pinning definition is the meaningful one and survives
    propagation ever becoming something other than a product.

    Conditional factors are excluded from the contest. A guard makes the low end
    zero and therefore the spread infinite, so it would win every time while
    telling the user nothing they can act on. That it is conditional is reported
    separately, and remains true.
    """
    contenders = [f for f in factors if not f.certain and not f.conditional]
    if not contenders:
        return None

    base = product(f.range for f in factors if not f.conditional)
    if base.spread <= 1.0:
        return None

    best, best_collapse = None, 1.0
    for candidate in contenders:
        pinned = product(
            Range.exact(f.range.expected) if f is candidate else f.range
            for f in factors
            if not f.conditional
        )
        collapse = base.spread / pinned.spread if pinned.spread > 0 else float("inf")
        if collapse > best_collapse:
            best, best_collapse = candidate, collapse
    return best


@dataclass(frozen=True)
class SiteEstimate:
    """Everything known about one call site's cost, and how sure we are."""

    site_id: str
    shape: str  # the CallSite shape: "sdk" | "http" | "framework" | "wrapper"
    cost_per_call: Range | None
    multiplicity: Range | None
    frequency: Range | None
    factors: tuple[Factor, ...] = ()
    notes: tuple[str, ...] = ()
    confidence: float = 1.0

    @property
    def per_day(self) -> Range | None:
        """Dollars a day, or ``None`` when any layer is unknown.

        A missing layer is not a zero and not a guess. Three layers times None
        is None, and the comment says which one is missing.
        """
        if self.cost_per_call is None or self.multiplicity is None or self.frequency is None:
            return None
        return self.cost_per_call * self.multiplicity * self.frequency

    @property
    def per_month(self) -> Range | None:
        daily = self.per_day
        return daily.scale(30.0) if daily is not None else None

    @property
    def resolved(self) -> bool:
        return self.per_day is not None

    @property
    def conditional(self) -> bool:
        return any(f.conditional for f in self.factors)

    @property
    def missing(self) -> tuple[str, ...]:
        """Which layers blocked a figure, for the comment to report."""
        gaps = []
        if self.cost_per_call is None:
            gaps.append("cost per call")
        if self.multiplicity is None:
            gaps.append("multiplicity")
        if self.frequency is None:
            gaps.append("frequency")
        return tuple(gaps)

    def dominant(self) -> Factor | None:
        return dominant(self.factors)


@dataclass(frozen=True)
class Total:
    """The headline figure, plus the coverage that keeps it honest."""

    per_month: Range | None
    counted: tuple[SiteEstimate, ...] = ()
    unresolved: tuple[SiteEstimate, ...] = ()
    excluded: tuple[SiteEstimate, ...] = ()

    @property
    def analysed(self) -> int:
        return len(self.counted)

    @property
    def coverage(self) -> str:
        """Always reported. A miss the reader can see is not a miss that lies."""
        total = len(self.counted) + len(self.unresolved)
        return f"priced {len(self.counted)} of {total} call sites"

    def dominant(self) -> Factor | None:
        return dominant([f for estimate in self.counted for f in estimate.factors])


def aggregate(estimates) -> Total:
    """Sum what can be summed, and say what could not.

    Wrapper sites are excluded from the total on purpose. `service.handle`
    calling `self.llm.chat()` and `LLMClient.chat` calling the SDK are the same
    money seen from two places; frequency propagation already routes the traffic
    from every entry point down to the SDK site, so counting the wrapper as well
    would bill it twice. Wrapper sites still matter — they are the blast radius,
    the answer to which call sites a change affects — so they are kept and
    labelled rather than dropped.

    Framework sites are counted. `llm.invoke(...)` resolves to a class outside
    the repository, so there is no site underneath it to be double.
    """
    counted, unresolved, excluded = [], [], []
    for estimate in estimates:
        if estimate.shape == "wrapper":
            excluded.append(estimate)
        elif estimate.resolved:
            counted.append(estimate)
        else:
            unresolved.append(estimate)

    total: Range | None = None
    for estimate in counted:
        monthly = estimate.per_month
        total = monthly if total is None else total + monthly  # type: ignore[operator]

    return Total(
        per_month=total,
        counted=tuple(counted),
        unresolved=tuple(unresolved),
        excluded=tuple(excluded),
    )
