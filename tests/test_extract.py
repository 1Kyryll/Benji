"""Extraction and call-site identity.

The load-bearing test here is ``test_id_survives_inserted_lines``. Everything
else in the system is filed under these identifiers — telemetry, matching across
versions, cost history — so an identifier that moves when someone adds an import
would silently orphan all of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chihuahuabot.extract import CallSite, attr_path, extract, kwarg_literal

FIXTURE = Path(__file__).parent / "fixtures" / "simple.py"


@pytest.fixture(scope="module")
def source() -> str:
    return FIXTURE.read_text()


@pytest.fixture(scope="module")
def sites(source: str) -> list[CallSite]:
    return extract(source, "simple.py")


def by_qualname(sites: list[CallSite], qualname: str) -> list[CallSite]:
    return [s for s in sites if s.qualname == qualname]


# --- identity -------------------------------------------------------------


def test_id_survives_inserted_lines(source: str):
    """Adding an import must not renumber or rename a single call site."""
    before = [s.id for s in extract(source, "simple.py")]
    after = [s.id for s in extract("import os\n\n" + source, "simple.py")]
    assert before == after


def test_inserting_lines_does_move_the_reported_line_number(source: str):
    """The opposite guarantee: lineno tracks the file, so it cannot be identity."""
    before = [s.lineno for s in extract(source, "simple.py")]
    after = [s.lineno for s in extract("import os\n\n" + source, "simple.py")]
    assert after == [line + 2 for line in before]


def test_id_is_file_qualname_ordinal(sites: list[CallSite]):
    assert by_qualname(sites, "outer")[0].id == "simple.py:outer:0"


# --- ordinals -------------------------------------------------------------


def test_ordinal_counts_metered_calls_only(sites: list[CallSite]):
    """`normalise(...)` sits between two metered calls and must not consume one."""
    triage = by_qualname(sites, "TicketService.triage")
    assert [s.ordinal for s in triage] == [0, 1]


def test_nested_function_gets_its_own_counter(sites: list[CallSite]):
    """`outer.inner` restarts at 0 rather than continuing `outer`'s sequence."""
    assert by_qualname(sites, "outer")[0].ordinal == 0
    assert by_qualname(sites, "outer.inner")[0].ordinal == 0


def test_module_level_call_is_attributed_to_module_scope(sites: list[CallSite]):
    assert by_qualname(sites, "<module>")[0].ordinal == 0


def test_method_qualname_includes_the_class(sites: list[CallSite]):
    assert by_qualname(sites, "TicketService.triage")


def test_every_id_is_unique(sites: list[CallSite]):
    ids = [s.id for s in sites]
    assert len(ids) == len(set(ids))


# --- detection ------------------------------------------------------------


def test_openai_chat_call_is_detected(sites: list[CallSite]):
    site = by_qualname(sites, "outer")[0]
    assert (site.provider, site.api, site.model) == ("openai", "chat", "gpt-4o")


def test_anthropic_messages_call_is_detected(sites: list[CallSite]):
    site = by_qualname(sites, "TicketService.triage")[1]
    assert (site.provider, site.api) == ("anthropic", "messages")


def test_receiver_name_does_not_affect_detection():
    """Detectors match the tail, so any receiver works — that is the whole point."""
    source = 'self._oai.chat.completions.create(model="gpt-4o")\n'
    assert extract(source, "f.py")[0].provider == "openai"


def test_unresolved_model_is_none_not_a_guess(sites: list[CallSite]):
    """`model=config.DEFAULT_MODEL` is unknowable. None is correct output."""
    assert by_qualname(sites, "unresolved_model")[0].model is None


def test_unresolved_model_still_produces_a_call_site(sites: list[CallSite]):
    """The call is still there and still costs money; only the model is unknown."""
    assert len(by_qualname(sites, "unresolved_model")) == 1


# --- HTTP shape -----------------------------------------------------------


def test_raw_http_call_is_a_call_site(sites: list[CallSite]):
    site = by_qualname(sites, "raw_http")[0]
    assert (site.shape, site.provider) == ("http", "openai")


def test_http_model_comes_from_the_request_payload(sites: list[CallSite]):
    assert by_qualname(sites, "raw_http")[0].model == "gpt-4o-mini"


def test_http_call_to_an_unrelated_endpoint_is_not_a_call_site():
    source = 'session.post(f"{u}/api/health", json={"model": "x"})\n'
    assert extract(source, "f.py") == []


def test_http_model_is_none_when_the_payload_is_not_literal():
    source = 'session.post("https://api.openai.com/v1/chat/completions", json=payload)\n'
    assert extract(source, "f.py")[0].model is None


# --- helpers --------------------------------------------------------------


def test_call_site_is_hashable():
    """Sites go into sets and dict keys all over the matching stage."""
    site = extract('client.messages.create(model="claude-sonnet-4")\n', "f.py")[0]
    assert len({site, site}) == 1


def test_attr_path_flattens_nested_attributes():
    import ast

    node = ast.parse("a.b.c(x)").body[0].value.func
    assert attr_path(node) == ("a", "b", "c")


def test_kwarg_literal_returns_none_for_a_non_literal():
    import ast

    call = ast.parse("f(model=config.DEFAULT)").body[0].value
    assert kwarg_literal(call, "model") is None


def test_await_does_not_hide_a_call():
    source = 'async def f():\n    await client.messages.create(model="claude-sonnet-4")\n'
    assert len(extract(source, "f.py")) == 1


def test_unparseable_source_raises_rather_than_reporting_no_cost():
    with pytest.raises(SyntaxError):
        extract("def broken(:\n", "f.py")


# --- the callable-object framework shape ----------------------------------


def test_a_model_object_called_directly_is_a_call_site():
    """`self.llm(messages)` — LangChain models are callable, and older code
    calls them this way. A method-name detector cannot see it at all."""
    site = extract("def f():\n    return self.llm(messages)\n", "a.py")[0]
    assert site.shape == "framework"


def test_a_bare_model_object_called_directly_is_a_call_site():
    assert extract("def f():\n    return llm(messages)\n", "a.py")


def test_a_pytorch_forward_pass_is_not_a_call_site():
    """`self.model(x)` is a forward pass in every piece of PyTorch ever written.
    Pricing a tensor multiply would be worse than missing a call."""
    assert extract("def f():\n    return self.model(x)\n", "a.py") == []


def test_the_callable_shape_carries_reduced_confidence():
    """A variable name is weaker evidence than a type."""
    assert extract("def f():\n    return self.llm(m)\n", "a.py")[0].confidence < 1.0
