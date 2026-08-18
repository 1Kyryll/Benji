"""Splitting input tokens into what we counted and what we cannot see.

The split is the product. Counting the literal part exactly and reporting the
interpolated part as a *count of holes* is what lets a later stage name the
dominant uncertainty instead of printing an invented total.
"""

from __future__ import annotations

from chihuahuabot.extract import extract
from chihuahuabot.tokens import count, estimate_input


def site(call: str):
    return extract(f"def f(x):\n    return {call}\n", "app.py")[0]


def chat(messages: str, model: str = "gpt-4o", extra: str = ""):
    """An OpenAI chat call whose `messages=` argument is `messages`."""
    return site(f'c.chat.completions.create(model="{model}", messages={messages}{extra})')


def estimate(call, model: str | None = "gpt-4o"):
    return estimate_input(call.content, model)


USER = '[{"role": "user", "content": "Summarise this ticket."}]'


# --- the split ------------------------------------------------------------


def test_fully_literal_prompt_is_counted_exactly():
    result = estimate(chat(USER))
    assert result.static > 0 and result.complete


def test_interpolated_value_is_reported_as_a_hole_not_a_guess():
    result = estimate(chat('[{"role": "user", "content": ticket.body}]'))
    assert result.unknown_slots == 1 and not result.complete


def test_f_string_splits_into_literal_and_hole():
    result = estimate(chat('[{"content": f"Summarise: {ticket.body}"}]'))
    assert result.static > 0 and result.unknown_slots == 1


def test_the_hole_is_named_so_it_can_be_blamed_later():
    """'the range is driven by ticket.body' needs the name, not just a count."""
    result = estimate(chat('[{"content": f"Hi {ticket.body}"}]'))
    assert "ticket.body" in result.note


def test_several_holes_are_counted_separately():
    assert estimate(chat('[{"content": f"{a} and {b}"}]')).unknown_slots == 2


def test_whole_messages_variable_is_one_hole():
    result = estimate(chat("history"))
    assert result.unknown_slots == 1 and result.static == 0


# --- what is and is not prompt text ---------------------------------------


def test_structural_keys_are_not_counted_as_prose():
    """Counting "user" and "assistant" would inflate every message."""
    with_role = estimate(chat('[{"role": "assistant", "content": "hi"}]'))
    without = estimate(chat('[{"content": "hi"}]'))
    assert with_role.static == without.static


def test_configuration_arguments_cost_nothing():
    bare = estimate(chat('[{"content": "hi"}]'))
    configured = estimate(chat('[{"content": "hi"}]', extra=", temperature=0.7, stream=True"))
    assert bare.static == configured.static


def test_message_count_is_tracked_for_per_message_framing():
    assert estimate(chat('[{"content": "a"}, {"content": "b"}]')).messages == 2


def test_http_payload_prompt_is_counted_too():
    """Raw HTTP calls carry the prompt in `json=`, not in `messages=`."""
    payload = '{"model": "gpt-4o", "messages": [{"content": "Summarise this."}]}'
    call = site(f'session.post("https://api.openai.com/v1/messages", json={payload})')
    assert estimate(call).static > 0


def test_system_prompt_is_counted():
    call = site('c.messages.create(model="claude-sonnet-4", system="You are terse.")')
    assert estimate(call, "claude-sonnet-4").static > 0


# --- honesty about the tokenizer ------------------------------------------


def test_unknown_model_still_produces_a_count_but_flags_it():
    """A count from the wrong tokenizer beats no count, as long as it says so."""
    tokens, approximate, _ = count("hello world", "claude-sonnet-4")
    assert tokens > 0 and approximate


def test_openai_model_uses_its_own_encoding():
    tokens, approximate, name = count("hello world", "gpt-4o")
    assert tokens > 0 and not approximate and name


def test_empty_text_costs_nothing():
    assert count("", "gpt-4o")[0] == 0


def test_unparseable_content_does_not_crash():
    result = estimate_input("def broken(:", None)
    assert result.static == 0 and result.approximate
