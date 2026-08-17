from app.clients import LLMClient


class Layer3:
    def __init__(self):
        self.llm = LLMClient()

    def run(self, x):
        return self.llm.chat(x)


class Layer2:
    def __init__(self):
        self.inner = Layer3()

    def run(self, x):
        return self.inner.run(x)


class Layer1:
    def __init__(self):
        self.inner = Layer2()

    def run(self, x):
        return self.inner.run(x)
