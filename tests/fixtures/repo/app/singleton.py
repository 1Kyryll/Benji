from app.clients import LLMClient

llm = LLMClient()


def go(prompt):
    return llm.chat(prompt)


def not_metered(x):
    return x.strip().lower()
