import json

import requests

from risk_policy_distillation.llms.llm_component import LLMComponent


class OllamaComponent(LLMComponent):

    def __init__(self, model_name, ollama_server="http://0.0.0.0:11435/api/chat"):
        self.connection_type = "ollama"
        self.ollama_server = ollama_server
        self.model = model_name

    def send_request(self, context, prompt, response=None, temperature=0.7):
        messages = self.generate_message(context, prompt, response)
        label = self.ollama_gen(messages, temperature)
        return label["content"]

    def ollama_gen(self, messages, temperature=0.7, seed=42):
        r = requests.post(
            self.ollama_server,
            json={
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "num_ctx": 1024 * 8,
                    "temperature": temperature,
                    "seed": seed,
                },
            },
            stream=False,
        )
        r.raise_for_status()
        output = ""

        for line in r.iter_lines():
            body = json.loads(line)

            if "error" in body:
                raise Exception(body["error"])

            if body.get("done") is False:
                message = body.get("message", "")
                content = message.get("content", "")
                output += content

            if body.get("done", False):
                message = body.get("message", "")
                message["content"] = output
                return message
