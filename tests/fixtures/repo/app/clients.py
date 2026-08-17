# The wrapper every real codebase has. Nothing calls the SDK anywhere else.
import openai

_oai = openai.OpenAI(api_key="sk-fake-000")


class LLMClient:
    def chat(self, prompt):
        return _oai.chat.completions.create(model="gpt-4o-mini", messages=[])

    def embed(self, text):
        return _oai.embeddings.create(model="text-embedding-3-small", input=text)


class UnusedClient:
    def chat(self, prompt):
        return "not metered, and shares a method name with LLMClient"
