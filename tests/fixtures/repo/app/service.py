from app.clients import LLMClient


class AnnotatedService:
    # Parameter annotation passed straight through: the strongest evidence.
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def handle(self, ticket):
        return self.llm.chat(ticket)


class ConstructedService:
    # Constructed in place.
    def __init__(self):
        self.llm = LLMClient()

    def handle(self, ticket):
        return self.llm.chat(ticket)

    def indirect(self, ticket):
        # self.method(...) hop, then the wrapper hop underneath it.
        return self.handle(ticket)


def free_function(ticket):
    local = LLMClient()
    return local.chat(ticket)
