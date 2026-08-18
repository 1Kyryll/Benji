"""Declared traffic, and carrying it down the call graph to individual sites.

The layer that makes the tool more than a linter, and the one most able to
produce a confidently wrong number. Every test here is about refusing to invent
traffic, or about carrying a real declaration to the right place.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benji.estimate import Range
from benji.frequency import (
    Config,
    ConfigFrequencySource,
    FrequencySource,
    parse_range,
    propagate,
)
from benji.index import RepoIndex

REPO = Path(__file__).parent / "fixtures" / "repo"

HANDLER = "app.service:AnnotatedService.handle"
SINGLETON = "app.singleton:go"
DEEP = "app.deep:Layer1.run"
SDK_CALL = "app.clients:LLMClient.chat"


@pytest.fixture(scope="module")
def index() -> RepoIndex:
    return RepoIndex.build(REPO, sorted(REPO.rglob("*.py")))


def source(**declared: object) -> ConfigFrequencySource:
    return ConfigFrequencySource(
        Config(frequency={k: parse_range(v, k) for k, v in declared.items()})
    )


# --- declaring ------------------------------------------------------------


def test_a_bare_number_is_a_certainty():
    assert parse_range(5000, "x") == Range.exact(5000)


def test_a_table_is_a_range():
    assert parse_range({"low": 2000, "expected": 5000, "high": 20000}, "x") == Range(
        2000, 5000, 20000
    )


def test_expected_defaults_to_the_midpoint():
    assert parse_range({"low": 0, "high": 10}, "x").expected == 5


def test_a_range_missing_its_bounds_is_rejected_by_name():
    """The error has to say which key is wrong, or the user cannot fix it."""
    with pytest.raises(ValueError, match="frequency.handler"):
        parse_range({"expected": 5}, "frequency.handler")


def test_an_out_of_order_range_is_rejected():
    with pytest.raises(ValueError):
        parse_range({"low": 10, "expected": 1, "high": 20}, "x")


def test_a_string_is_not_a_frequency():
    with pytest.raises(ValueError):
        parse_range("5000/day", "x")


def test_a_boolean_is_not_a_frequency():
    with pytest.raises(ValueError):
        parse_range(True, "x")


# --- the config file ------------------------------------------------------


def test_config_reads_frequency_and_iterables(tmp_path: Path):
    config_file = tmp_path / "benji.toml"
    config_file.write_text(
        "[frequency]\n"
        '"api:handle" = 5000\n'
        "[iterables]\n"
        '"handle:orgs" = { low = 3, expected = 40, high = 500 }\n'
    )
    config = Config.load(config_file)
    assert config.frequency["api:handle"] == Range.exact(5000)
    assert config.iterables["handle:orgs"].high == 500


def test_a_missing_config_is_not_an_error(tmp_path: Path):
    """The bot still reports cost per call; it just cannot reach dollars per day."""
    config = Config.discover(tmp_path)
    assert config.frequency == {} and not config.declared


def test_discover_finds_the_config_in_the_repository_root(tmp_path: Path):
    (tmp_path / "benji.toml").write_text('[frequency]\n"a:b" = 1\n')
    assert Config.discover(tmp_path).declared


# --- propagation ----------------------------------------------------------


def test_a_declared_function_gets_its_own_traffic(index: RepoIndex):
    result = propagate(index, source(**{HANDLER: 5000}), HANDLER)
    assert result.range == Range.exact(5000)


def test_traffic_reaches_the_call_site_through_the_wrapper(index: RepoIndex):
    """Nobody declares how often LLMClient.chat runs. They declare the handler."""
    result = propagate(index, source(**{HANDLER: 5000}), SDK_CALL)
    assert result.range == Range.exact(5000)
    assert result.entry_points == (HANDLER,)


def test_traffic_reaches_through_several_hops(index: RepoIndex):
    result = propagate(index, source(**{DEEP: 100}), SDK_CALL)
    assert result.resolved


def test_several_entry_points_sum(index: RepoIndex):
    result = propagate(index, source(**{HANDLER: 5000, SINGLETON: 1000}), SDK_CALL)
    assert result.range == Range.exact(6000)
    assert set(result.entry_points) == {HANDLER, SINGLETON}


def test_ranges_sum_end_to_end(index: RepoIndex):
    declared = source(
        **{HANDLER: {"low": 1, "expected": 2, "high": 3}, SINGLETON: {"low": 10, "high": 30}}
    )
    assert propagate(index, declared, SDK_CALL).range == Range(11, 22, 33)


def test_an_undeclared_call_site_has_no_frequency(index: RepoIndex):
    """No declaration reaches it, so nothing is assumed."""
    result = propagate(index, source(), SDK_CALL)
    assert result.range is None and not result.resolved


def test_an_undeclared_result_says_why(index: RepoIndex):
    assert "no declared entry point" in propagate(index, source(), SDK_CALL).note


def test_a_declaration_elsewhere_does_not_leak(index: RepoIndex):
    """`UnusedClient.chat` is reached by nobody and shares a method name with the
    real client. A name collision must not manufacture traffic."""
    result = propagate(index, source(**{HANDLER: 5000}), "app.clients:UnusedClient.chat")
    assert result.range is None


def test_propagated_traffic_admits_the_path_multiplicity_gap(index: RepoIndex):
    """If an entry point calls a helper in a loop, the helper runs more often
    than this reports. Stated rather than hidden."""
    result = propagate(index, source(**{HANDLER: 5000}), SDK_CALL)
    assert "multiplicity" in result.note


def test_a_directly_declared_function_carries_no_such_caveat(index: RepoIndex):
    assert propagate(index, source(**{HANDLER: 5000}), HANDLER).note == ""


# --- the seam -------------------------------------------------------------


def test_the_config_source_satisfies_the_protocol():
    """Telemetry plugs in here without touching a caller."""
    assert isinstance(ConfigFrequencySource(Config()), FrequencySource)


def test_a_stub_telemetry_source_drops_in(index: RepoIndex):
    """The point of the adapter: swapping the source changes nothing downstream."""

    class StubTelemetry:
        name = "stub"

        def invocations_per_day(self, function_key: str) -> Range | None:
            return Range(900, 1000, 1100) if function_key == HANDLER else None

    result = propagate(index, StubTelemetry(), SDK_CALL)
    assert result.range == Range(900, 1000, 1100)


def test_the_source_names_itself_for_the_comment():
    """A reader must be able to tell a declared number from an observed one."""
    assert "declared" in ConfigFrequencySource(Config()).name
