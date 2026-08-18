"""Carrying uncertainty through the three layers, and naming what drives it.

The dominant-uncertainty calculation is the highest-value output in the product.
"$40 to $900" is useless; "the range is driven by tickets-per-org" is a sentence
someone installs a bot to read. These tests protect that sentence being true.
"""

from __future__ import annotations

import pytest

from chihuahuabot.estimate import (
    DEFAULT_SLOT_PRIOR,
    Factor,
    Range,
    SiteEstimate,
    aggregate,
    cost_range,
    dominant,
    input_range,
    output_range,
)


def estimate(**kwargs) -> SiteEstimate:
    base = dict(
        site_id="app.py:handle:0",
        shape="sdk",
        cost_per_call=Range.exact(0.002),
        multiplicity=Range.exact(1),
        frequency=Range.exact(1000),
    )
    base.update(kwargs)
    return SiteEstimate(**base)  # type: ignore[arg-type]


def factor(name: str, low: float, high: float, layer: str = "cost") -> Factor:
    return Factor(name, layer, Range(low, (low + high) / 2, high))


# --- input tokens ---------------------------------------------------------


def test_a_fully_literal_prompt_contributes_no_uncertainty():
    assert input_range(120, 0).is_point


def test_each_hole_widens_the_input_range():
    assert input_range(120, 1).high > input_range(120, 0).high


def test_two_holes_are_wider_than_one():
    assert input_range(0, 2).high == 2 * DEFAULT_SLOT_PRIOR.high


def test_the_static_half_is_carried_exactly():
    assert input_range(120, 1).low == 120 + DEFAULT_SLOT_PRIOR.low


# --- output tokens --------------------------------------------------------


def test_max_tokens_is_a_ceiling_not_a_prediction():
    """A classifier told to answer in one word against max_tokens=50 overshoots
    by twenty times if you believe the ceiling."""
    assert output_range(50, "classifier").expected < 50


def test_output_scales_with_the_declared_ceiling():
    assert output_range(1000, "chat").expected > output_range(100, "chat").expected


def test_task_shape_changes_the_prior():
    assert output_range(1000, "classifier").expected < output_range(1000, "summariser").expected


def test_a_call_without_a_ceiling_still_gets_a_prior():
    assert output_range(None, "chat").high > 0


def test_an_unknown_shape_falls_back_rather_than_failing():
    assert output_range(1000, "nonsense") == output_range(1000, "chat")


# --- cost per call --------------------------------------------------------


def test_cost_carries_token_uncertainty_through():
    certain = cost_range(2.5, 10.0, Range.exact(1000), Range.exact(300))
    uncertain = cost_range(2.5, 10.0, Range(500, 1000, 5000), Range.exact(300))
    assert uncertain.spread > certain.spread


def test_cost_is_dollars():
    assert cost_range(2.5, 10.0, Range.exact(1_000_000), Range.exact(0)).expected == pytest.approx(
        2.5
    )


# --- propagation ----------------------------------------------------------


def test_the_three_layers_multiply():
    result = estimate(
        cost_per_call=Range.exact(0.002), multiplicity=Range.exact(10), frequency=Range.exact(100)
    )
    assert result.per_day == Range.exact(2.0)


def test_a_month_is_thirty_days():
    assert estimate().per_month.expected == pytest.approx(estimate().per_day.expected * 30)


def test_a_missing_layer_yields_no_figure_rather_than_a_zero():
    """Three layers times None is None, not nothing-to-see-here."""
    assert estimate(frequency=None).per_day is None


def test_a_missing_layer_is_named():
    assert estimate(frequency=None, cost_per_call=None).missing == ("cost per call", "frequency")


def test_a_resolved_estimate_reports_no_gaps():
    assert estimate().missing == ()


# --- dominant uncertainty -------------------------------------------------


def test_the_widest_factor_is_blamed():
    tokens = factor("ticket.body", 20, 200)
    volume = factor("tickets per org", 1, 1000, "frequency")
    assert dominant([tokens, volume]) is volume


def test_pinning_and_max_spread_agree_under_multiplication():
    """The pinning definition is the meaningful one; this records that it
    reduces to the widest factor while propagation stays a product."""
    factors = [factor("a", 1, 4), factor("b", 1, 50), factor("c", 2, 6)]
    widest = max(factors, key=lambda f: f.range.spread)
    assert dominant(factors) is widest


def test_certain_factors_are_never_blamed():
    assert (
        dominant([factor("wide", 1, 100), Factor("exact", "cost", Range.exact(5))]).name == "wide"
    )


def test_nothing_is_blamed_when_everything_is_certain():
    assert dominant([Factor("a", "cost", Range.exact(2))]) is None


def test_a_guard_does_not_win_by_being_infinite():
    """A conditional call has a zero low and therefore an infinite spread. It
    would win every time while telling the user nothing they can act on."""
    guard = Factor("if urgent", "multiplicity", Range(0, 0.5, 1))
    volume = factor("tickets per org", 1, 1000, "frequency")
    assert dominant([guard, volume]) is volume


def test_a_conditional_estimate_says_so_separately():
    guard = Factor("if urgent", "multiplicity", Range(0, 0.5, 1))
    assert estimate(factors=(guard,)).conditional


def test_dominance_survives_a_single_contender():
    assert dominant([factor("only", 1, 10)]).name == "only"


# --- aggregation ----------------------------------------------------------


def test_totals_sum_across_sites():
    two = [estimate(site_id="a"), estimate(site_id="b")]
    assert aggregate(two).per_month.expected == pytest.approx(2 * estimate().per_month.expected)


def test_wrapper_sites_are_not_billed_twice():
    """`service.handle` calling `self.llm.chat()` and `LLMClient.chat` calling
    the SDK are the same money seen from two places. Frequency propagation
    already routes traffic to the SDK site."""
    sites = [estimate(site_id="sdk"), estimate(site_id="wrapper", shape="wrapper")]
    total = aggregate(sites)
    assert total.per_month.expected == pytest.approx(estimate().per_month.expected)
    assert len(total.excluded) == 1


def test_excluded_wrapper_sites_are_kept_not_dropped():
    """They are the blast radius: which call sites a change affects."""
    sites = [estimate(site_id="w", shape="wrapper")]
    assert aggregate(sites).excluded[0].site_id == "w"


def test_framework_sites_are_counted():
    """`llm.invoke(...)` resolves outside the repository, so there is no site
    underneath it to be double."""
    sites = [estimate(site_id="f", shape="framework")]
    assert aggregate(sites).per_month is not None


def test_unresolved_sites_do_not_silently_vanish():
    sites = [estimate(site_id="ok"), estimate(site_id="no", frequency=None)]
    assert len(aggregate(sites).unresolved) == 1


def test_coverage_is_always_reported():
    """A miss the reader can see is not a miss that lies."""
    sites = [estimate(site_id="ok"), estimate(site_id="no", frequency=None)]
    assert aggregate(sites).coverage == "priced 1 of 2 call sites"


def test_a_total_blames_the_widest_factor_across_every_site():
    sites = [
        estimate(site_id="a", factors=(factor("narrow", 1, 2),)),
        estimate(site_id="b", factors=(factor("len(orgs)", 3, 500, "multiplicity"),)),
    ]
    assert aggregate(sites).dominant().name == "len(orgs)"


def test_a_total_with_nothing_priced_is_none_not_zero():
    assert aggregate([estimate(frequency=None)]).per_month is None


# --- deltas -----------------------------------------------------------------


def test_negating_a_range_flips_it():
    """A removed call site is a saving; the old high becomes the new low."""
    assert -Range(10, 20, 30) == Range(-30, -20, -10)


def test_a_signed_range_multiplies_over_all_corners():
    """A cost delta can be negative. Multiplying its low by a multiplier's low
    would put the best case where the worst case belongs."""
    assert Range(-2, 0, 2) * Range(0, 0.5, 1) == Range(-2, 0, 2)


def test_non_negative_multiplication_is_unchanged():
    assert Range(1, 2, 3) * Range(10, 20, 30) == Range(10, 40, 90)
