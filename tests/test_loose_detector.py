"""The loose detector fixes the denominator of the coverage number.

Its contract is recall, not precision: a metered call that never becomes a
candidate is invisible forever, while a false candidate merely shows up as an
admitted gap. Every test here protects one side of that asymmetry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corpus.detect import attr_path, classify, find_candidates

FIXTURE = Path(__file__).parent / "fixtures" / "wrapping_styles.py"


@pytest.fixture(scope="module")
def candidates():
    source = FIXTURE.read_text()
    return {c.dotted: c for c in find_candidates(source, "wrapping_styles.py")}


def test_inline_sdk_call_is_a_candidate(candidates):
    assert "client.chat.completions.create" in candidates


def test_anthropic_messages_create_is_a_candidate(candidates):
    # Tail is the weak verb `create`; `messages` is what rescues it.
    assert "anthropic.Anthropic().messages.create" in candidates


def test_wrapped_client_call_is_a_candidate(candidates):
    # The shape the whole project exists for: the SDK is behind a wrapper.
    assert "self.llm.chat" in candidates


def test_module_singleton_call_is_a_candidate(candidates):
    assert "llm.generate" in candidates


def test_bare_provider_function_is_a_candidate(candidates):
    assert "completion" in candidates


def test_orm_create_is_not_a_candidate(candidates):
    # Django's `Model.objects.create` would otherwise flood the denominator.
    assert "Ticket.objects.create" not in candidates


def test_db_session_call_is_not_a_candidate(candidates):
    assert "session.create" not in candidates


def test_logging_and_cache_calls_are_not_candidates(candidates):
    assert "logger.info" not in candidates
    assert "cache.get" not in candidates


def test_factory_receiver_does_not_collapse_to_a_bare_name():
    # `get_client().chat` must keep evidence that a call produced the receiver,
    # otherwise it is indistinguishable from a bare `chat(...)`. The factory's
    # own name is retained too — step 2 will need it to resolve the receiver.
    source = "get_client().chat(x)\n"
    found = find_candidates(source, "f.py")
    assert [c.path for c in found] == [("get_client", "()", "chat")]


def test_dotted_renders_the_way_source_reads():
    source = "get_client().chat(x)\n"
    assert find_candidates(source, "f.py")[0].dotted == "get_client().chat"


def test_capitalised_bare_name_is_not_a_candidate(candidates):
    # prompt_toolkit's `Completion(...)` class, found in aider by the corpus run.
    # A capitalised bare name is a constructor by convention.
    assert "Completion" not in candidates


def test_raw_http_call_to_a_provider_endpoint_is_a_candidate(candidates):
    # open-webui never imports a provider SDK. If the denominator ignored this
    # shape, coverage would look good precisely by omitting what we cannot do.
    http = [c for c in candidates.values() if "http+endpoint" in c.rules]
    assert len(http) == 1
    assert http[0].dotted == "session.post"


def test_raw_http_call_to_an_unrelated_endpoint_is_not_a_candidate(candidates):
    # `/api/health` costs nothing. Endpoint evidence is what separates them.
    http_lines = {c.lineno for c in candidates.values() if "http+endpoint" in c.rules}
    source = FIXTURE.read_text().splitlines()
    assert not any("api/health" in source[line - 1] for line in http_lines)


def test_http_rule_survives_an_anti_noun_receiver():
    # The canonical shape is `session.post`, and `session` is an anti-noun.
    import ast

    call = ast.parse('session.post(f"{u}/v1/messages")').body[0].value
    assert "http+endpoint" in classify(attr_path(call.func), call)


def test_weak_verb_alone_is_not_a_candidate():
    # No provider noun anywhere in the path, so `create` proves nothing.
    assert classify(("thing", "create")) == ()


def test_strong_verb_survives_an_anti_noun():
    # Anti-nouns suppress weak matches only. `db.chat(...)` is odd but the tail
    # is unambiguous, and recall wins ties.
    assert "strong" in classify(("db", "chat"))


def test_attr_path_flattens_nested_attributes():
    import ast

    node = ast.parse("a.b.c(x)").body[0].value.func
    assert attr_path(node) == ("a", "b", "c")


def test_candidates_are_ordered_by_position():
    source = "llm.chat(1)\nother.thing()\nllm.chat(2)\n"
    found = find_candidates(source, "f.py")
    assert [c.lineno for c in found] == [1, 3]
