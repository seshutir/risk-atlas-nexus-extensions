from fm_factual.context_retriever import ContextRetriever


class CustomRetriever(ContextRetriever):

    def __init__(self, contexts):
        self.contexts = contexts

    def query(self, text):
        return [
            {"title": "", "text": context, "snippet": "", "link": "", "probability": p}
            for (context, p) in self.contexts
        ]
