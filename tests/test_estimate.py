"""The range type every number in the system is carried in.

The bot never prints a point estimate. Fake precision is what kills a tool like
this: "+$127.40/mo", wrong by a factor of ten, is worse than saying nothing.
"""

from __future__ import annotations

import pytest

from chihuahuabot.estimate import Range


def test_ranges_multiply_low_by_low_and_high_by_high():
    """Deliberately naive. Correlations are unknowable here, and pretending
    otherwise would dress a guess as a calculation."""
    assert Range(1, 2, 3) * Range(10, 20, 30) == Range(10, 40, 90)


def test_an_exact_value_is_still_a_range():
    """So nothing downstream has to special-case certainty."""
    assert Range.exact(5).is_point


def test_a_range_out_of_order_is_rejected():
    with pytest.raises(ValueError):
        Range(10, 2, 3)


def test_expected_may_sit_at_either_end():
    assert Range(1, 1, 3).expected == 1
    assert Range(1, 3, 3).expected == 3


def test_spread_is_how_many_times_wider_the_high_is():
    assert Range(2, 5, 10).spread == 5


def test_a_zero_low_is_unbounded_spread():
    """A guard can make a call never happen, and no multiplier rescues that."""
    assert Range(0, 0.5, 1).spread == float("inf")


def test_multiplying_by_an_exact_range_preserves_the_spread():
    assert (Range(1, 2, 4) * Range.exact(3)).spread == Range(1, 2, 4).spread


def test_point_ranges_render_as_a_single_number():
    assert str(Range.exact(3)) == "3"


def test_wide_ranges_show_both_ends_and_the_expectation():
    assert str(Range(1, 2, 4)) == "1..4 (exp 2)"
