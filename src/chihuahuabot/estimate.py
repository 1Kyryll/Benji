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
