"""The PR comment.

The only part of the system anyone reads, so the invariants that matter are
about honesty rather than layout: never a bare point estimate, never a silent
gap, always the coverage line, always a way to find the comment again.
"""

from __future__ import annotations

from chihuahuabot.estimate import Factor, Range, SiteEstimate
from chihuahuabot.render import (
    MARKER,
    BlastRadius,
    Coverage,
    Report,
    SiteChange,
    money,
    money_range,
    render,
)


def site(**kwargs) -> SiteEstimate:
    base = dict(
        site_id="api/tickets.py:TicketService.handle:0",
        shape="sdk",
        cost_per_call=Range.exact(0.0031),
        multiplicity=Range.exact(1),
        frequency=Range.exact(5000),
    )
    base.update(kwargs)
    return SiteEstimate(**base)  # type: ignore[arg-type]


def report(**kwargs) -> Report:
    base = dict(
        delta=Range(180, 610, 2400),
        changes=(
            SiteChange(
                "edited",
                "api/tickets.py:TicketService.handle:0",
                "`gpt-4o-mini` → `gpt-4o`",
                site(),
            ),
        ),
        coverage=Coverage(34, 51),
        prices_as_of="2026-07-01",
    )
    base.update(kwargs)
    return Report(**base)  # type: ignore[arg-type]


# --- never a point estimate -----------------------------------------------


def test_the_headline_is_a_range():
    assert "+$180 to +$2,400" in render(report())


def test_a_range_never_collapses_to_one_number_in_the_headline():
    """The failure that kills a tool like this is a confident wrong figure."""
    body = render(report())
    assert " to " in body.split("\n")[2]


def test_a_saving_is_rendered_as_a_saving():
    body = render(report(delta=Range(-900, -400, -100)))
    assert "−" in body


def test_a_delta_that_might_go_either_way_shows_both_signs():
    body = render(report(delta=Range(-50, 100, 400)))
    assert "−$50" in body and "+$400" in body


def test_an_unresolvable_delta_says_so_rather_than_printing_zero():
    """A layer we could not resolve is not a change that costs nothing."""
    headline = render(report(delta=None)).split("\n")[2]
    assert "unknown" in headline.lower() and "$" not in headline


# --- the sentence the tool exists for -------------------------------------


def test_the_dominant_uncertainty_is_named():
    body = render(
        report(
            dominant=Factor("len(orgs)", "multiplicity", Range(3, 40, 500), "prior"),
            total_spread=13.0,
        )
    )
    assert "len(orgs)" in body and "driven almost entirely" in body


def test_an_undeclared_driver_says_how_to_declare_it():
    body = render(
        report(
            dominant=Factor("len(orgs)", "multiplicity", Range(3, 40, 500), "prior"),
            total_spread=13.0,
        )
    )
    assert "chihuahuabot.toml" in body


def test_the_narrowing_is_quantified():
    """'declare this and the range gets 10x tighter' is the actionable half."""
    body = render(
        report(dominant=Factor("orgs", "multiplicity", Range(1, 5, 10), "prior"), total_spread=50.0)
    )
    assert "50×" in body and "5.0×" in body


def test_a_tight_range_does_not_blame_anything():
    """Naming a driver when the range is already tight is noise."""
    body = render(
        report(dominant=Factor("orgs", "multiplicity", Range(1, 1.1, 1.2)), total_spread=1.2)
    )
    assert "driven almost entirely" not in body


def test_a_declared_driver_is_described_differently_from_a_prior():
    body = render(
        report(
            dominant=Factor("tickets/day", "frequency", Range(2000, 5000, 20000), "declared"),
            total_spread=10.0,
        )
    )
    assert "declared as" in body


# --- coverage is never omitted --------------------------------------------


def test_coverage_is_always_reported():
    assert "34 of 51 candidate call sites" in render(report())


def test_coverage_reasons_are_listed_when_known():
    body = render(report(coverage=Coverage(34, 51, (("dynamic dispatch", 12), ("depth limit", 5)))))
    assert "12 dynamic dispatch" in body and "5 depth limit" in body


def test_full_coverage_still_prints_the_line():
    assert "51 of 51" in render(report(coverage=Coverage(51, 51)))


# --- the table ------------------------------------------------------------


def test_a_changed_site_shows_what_changed():
    assert "`gpt-4o-mini` → `gpt-4o`" in render(report())


def test_a_new_site_is_marked_new():
    body = render(report(changes=(SiteChange("added", "w.py:digest:0", "", site()),)))
    assert "**new**" in body


def test_an_unpriced_site_names_the_missing_layer():
    """A gap the reader can see is not a gap that lies."""
    body = render(
        report(changes=(SiteChange("added", "w.py:digest:0", "", site(cost_per_call=None)),))
    )
    assert "unpriced" in body and "cost per call" in body


def test_an_undeclared_frequency_is_stated_in_the_row():
    body = render(report(changes=(SiteChange("edited", "a.py:f:0", "", site(frequency=None)),)))
    assert "frequency undeclared" in body


def test_an_unknown_loop_bound_is_shown_by_name():
    body = render(
        report(changes=(SiteChange("added", "a.py:f:0", "", site(multiplicity=None), "len(orgs)"),))
    )
    assert "×`len(orgs)`" in body


def test_low_confidence_is_surfaced_in_the_row():
    body = render(report(changes=(SiteChange("edited", "a.py:f:0", "", site(), confidence=0.7),)))
    assert "conf 70%" in body


def test_a_pipe_in_a_site_id_cannot_break_the_table():
    body = render(report(changes=(SiteChange("edited", "a|b.py:f:0", "", site()),)))
    assert "a\\|b.py" in body


# --- blast radius ---------------------------------------------------------


def test_blast_radius_is_reported_when_a_change_is_not_local():
    body = render(report(blast=BlastRadius("LLMClient.chat", 47, 12, 4)))
    assert "47 call sites across 12 files" in body


def test_a_single_site_change_gets_no_blast_radius_banner():
    body = render(report(blast=BlastRadius("f", 1, 1)))
    assert "call sites across" not in body


# --- staleness and configuration ------------------------------------------


def test_the_price_date_is_always_shown():
    assert "prices as of 2026-07-01" in render(report())


def test_a_stale_price_table_warns():
    """A stale table producing a confident number is the failure to avoid."""
    assert "may be out of date" in render(report(prices_stale=True))


def test_a_missing_config_is_explained_not_ignored():
    body = render(report(config_present=False))
    assert "chihuahuabot.toml" in body and "dollars per day are not available" in body


def test_the_frequency_source_is_named():
    """A reader must tell a declared number from an observed one."""
    assert "declared" in render(report(frequency_source="declared (chihuahuabot.toml)"))


# --- finding the comment again --------------------------------------------


def test_the_comment_carries_its_marker():
    """One comment per branch, updated in place, not twenty."""
    assert MARKER in render(report())


def test_an_empty_report_is_detectable_so_the_bot_can_stay_silent():
    assert Report().empty


# --- formatting -----------------------------------------------------------


def test_fractions_of_a_cent_keep_their_precision():
    assert money(0.0008) == "$0.0008"


def test_large_figures_are_readable():
    assert money(2400) == "$2,400"


def test_a_certain_range_renders_as_one_number():
    assert money_range(Range.exact(5)) == "$5.00"


def test_a_meaningless_spread_blames_nothing():
    """A delta that crosses zero has no high-over-low ratio worth quoting."""
    driver = Factor("orgs", "multiplicity", Range(1, 5, 10), "prior")
    assert "driven almost entirely" not in render(
        report(delta=Range(-462, 299, 8186), dominant=driver, total_spread=-17.7)
    )


def test_an_infinite_spread_blames_nothing():
    driver = Factor("orgs", "multiplicity", Range(1, 5, 10), "prior")
    assert "driven almost entirely" not in render(
        report(dominant=driver, total_spread=float("inf"))
    )


def test_a_delta_crossing_zero_still_reports_both_ends():
    """The honest statement is that it could go either way."""
    body = render(report(delta=Range(-462, 299, 8186)))
    assert "−$462 to +$8,186" in body
