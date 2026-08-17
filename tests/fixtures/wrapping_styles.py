# Parsed, never executed. Fake keys and nonsense code are fine here.
# Each block is a call shape the loose detector has to get right.

import anthropic
import openai
from litellm import completion

API_KEY = "sk-not-a-real-key-000000"

client = openai.OpenAI(api_key=API_KEY)
llm = LLMClient()  # noqa: F821 — module-level singleton, never constructed


def inline_openai(prompt):
    # Inline SDK call, the easy case.
    return client.chat.completions.create(model="gpt-4o", messages=[{"role": "user"}])


def inline_anthropic(prompt):
    # Anthropic's namespace is `messages`, and the tail is the weak verb `create`.
    return anthropic.Anthropic().messages.create(model="claude-sonnet-4", max_tokens=64)


def bare_litellm(prompt):
    # No receiver at all. A bare name carries zero evidence beyond the name.
    return completion(model="gpt-4o-mini", messages=[])


class TicketService:
    def __init__(self, llm_client):
        self.llm = llm_client

    def handle(self, ticket):
        # The common real shape: the SDK is wrapped and this is what gets called.
        return self.llm.chat(ticket.body)


def module_singleton(text):
    return llm.generate(text)


def via_factory(text):
    # The receiver is the result of a call, so it must not collapse to a bare name.
    return get_client().chat(text)  # noqa: F821


def noisy_orm(subject):
    # Django. `create` is a weak verb and `objects` is an anti-noun.
    return Ticket.objects.create(subject=subject)  # noqa: F821


def noisy_db(payload):
    return session.create(payload)  # noqa: F821


def noisy_logging(message):
    logger.info(message)  # noqa: F821
    cache.get(message)  # noqa: F821


def noisy_completion_class(word):
    # prompt_toolkit's UI class. Capitalised bare name, so a constructor.
    return Completion(word, start_position=-1)  # noqa: F821


async def raw_http_provider(base_url, payload):
    # No SDK anywhere. open-webui talks to providers exactly like this, and the
    # endpoint in the f-string is the only evidence that money changes hands.
    async with session.post(f"{base_url}/api/chat", json=payload) as r:  # noqa: F821
        return await r.json()


async def raw_http_not_a_provider(base_url):
    async with session.post(f"{base_url}/api/health") as r:  # noqa: F821
        return await r.json()
