# Parsed, never executed. Fake keys and nonsense code are fine here.
#
# This fixture exists to break naive implementations. In order: a module-level
# call that has no enclosing function, a method, a nested helper whose ordinals
# must not continue the outer function's sequence, an unresolvable model, and an
# HTTP call that never imports an SDK.

import anthropic
import openai

import config  # noqa: F401 — never imported for real

client = openai.OpenAI(api_key="sk-fake-000")

# Module scope. Nothing encloses this call.
warmup = client.chat.completions.create(model="gpt-4o-mini", messages=[])


class TicketService:
    def __init__(self, http):
        self.http = http

    def triage(self, ticket):
        # ordinal 0 in TicketService.triage
        summary = client.chat.completions.create(model="gpt-4o-mini", messages=[])
        # An ordinary call in between. It must not consume an ordinal.
        normalise(summary)
        # ordinal 1 in TicketService.triage
        return anthropic.Anthropic().messages.create(
            model="claude-sonnet-4", max_tokens=512, messages=[]
        )


def outer(items):
    # ordinal 0 in outer
    first = client.chat.completions.create(model="gpt-4o", messages=[])

    def inner(item):
        # ordinal 0 in outer.inner — its own counter, not outer's
        return client.chat.completions.create(model="gpt-4o-mini", messages=[])

    return first, [inner(i) for i in items]


def unresolved_model(prompt):
    # The model is not knowable statically. None is the correct answer here;
    # downstream stages have to be able to see that it is unresolved.
    return client.chat.completions.create(model=config.DEFAULT_MODEL, messages=[])


async def raw_http(session, base_url, prompt):
    # No SDK anywhere. The model lives in the request payload.
    return await session.post(
        f"{base_url}/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": []},
    )


def normalise(value):
    return value
