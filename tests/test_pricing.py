"""The price table and the arithmetic on top of it.

Every test here guards one of two failures: printing a dollar figure we cannot
justify, or printing one that is wrong by an order of magnitude.
"""

from __future__ import annotations

from datetime import date

import pytest

from benji.pricing import PriceTable, cost


@pytest.fixture(scope="module")
def table() -> PriceTable:
    return PriceTable.load()


# --- lookup ---------------------------------------------------------------


def test_known_model_resolves_exactly(table: PriceTable):
    price = table.lookup("openai", "gpt-4o")
    assert price is not None and price.match == "exact"


def test_unknown_model_returns_none_rather_than_a_price(table: PriceTable):
    """`model=config.DEFAULT_MODEL` is unresolvable by design."""
    assert table.lookup("openai", "some-model-nobody-has-heard-of") is None


def test_absent_model_returns_none(table: PriceTable):
    assert table.lookup("openai", None) is None


def test_dated_snapshot_resolves_by_prefix(table: PriceTable):
    price = table.lookup("openai", "gpt-4o-2024-08-06")
    assert price is not None and price.model == "gpt-4o" and price.match == "prefix"


def test_prefix_match_is_not_reported_as_exact(table: PriceTable):
    """The layer above must be able to lower its confidence."""
    assert table.lookup("openai", "gpt-4o-2024-08-06").exact is False


def test_mini_snapshot_does_not_fall_through_to_the_larger_model(table: PriceTable):
    """`gpt-4o-mini-...` startswith `gpt-4o`. Longest prefix must win, or the
    estimate is out by a factor of sixteen."""
    price = table.lookup("openai", "gpt-4o-mini-2024-07-18")
    assert price.model == "gpt-4o-mini"


def test_similar_name_without_a_boundary_does_not_match(table: PriceTable):
    assert table.lookup("openai", "gpt-4omega") is None


def test_model_resolves_when_the_provider_is_unknown(table: PriceTable):
    """A LangChain or raw-HTTP call knows the model but not who bills for it."""
    price = table.lookup("framework", "claude-sonnet-4")
    assert price is not None and price.provider == "anthropic"
    assert price.match == "cross-provider"


# --- arithmetic -----------------------------------------------------------


def test_cost_is_dollars_per_call(table: PriceTable):
    price = table.lookup("openai", "gpt-4o")  # 2.50 in / 10.00 out per Mtok
    assert cost(price, input_tokens=1_000_000, output_tokens=0) == pytest.approx(2.50)


def test_input_and_output_are_priced_separately(table: PriceTable):
    price = table.lookup("openai", "gpt-4o")
    assert cost(price, 1_000_000, 1_000_000) == pytest.approx(12.50)


def test_cached_input_is_cheaper_than_fresh_input(table: PriceTable):
    price = table.lookup("openai", "gpt-4o")
    fresh = cost(price, 1_000_000, 0)
    cached = cost(price, 1_000_000, 0, cached_tokens=1_000_000)
    assert cached < fresh


def test_cached_tokens_are_a_subset_of_input_not_an_addition(table: PriceTable):
    """Prompt caching changes the input price by up to ten times. Treating the
    cached count as extra tokens would inflate every cached call."""
    price = table.lookup("openai", "gpt-4o")
    half = cost(price, 1_000_000, 0, cached_tokens=500_000)
    assert half == pytest.approx(0.5 * 2.50 + 0.5 * 1.25)


def test_embedding_model_has_no_output_cost(table: PriceTable):
    price = table.lookup("openai", "text-embedding-3-small")
    assert cost(price, 1_000_000, 1_000_000) == pytest.approx(0.02)


# --- staleness ------------------------------------------------------------


def test_table_reports_its_own_age(table: PriceTable):
    """A stale table producing a confident wrong number is the failure the
    whole range machinery exists to prevent, so the age must be visible."""
    assert table.age_days(date(2026, 7, 31)) == 30


def test_fresh_table_is_not_stale(table: PriceTable):
    assert table.is_stale(date(2026, 7, 31)) is False


def test_old_table_is_stale(table: PriceTable):
    assert table.is_stale(date(2027, 1, 1)) is True


def test_unparseable_date_counts_as_stale():
    """Not knowing the age is not reassuring."""
    assert PriceTable({"as_of": "whenever"}).is_stale(date(2026, 7, 31)) is True


def test_every_price_carries_the_tables_date(table: PriceTable):
    assert table.lookup("anthropic", "claude-sonnet-4").as_of == table.as_of
