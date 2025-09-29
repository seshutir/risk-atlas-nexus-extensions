import os

import requests


class RITSTokenizer():

    def __init__(self, model_name):
        self.model_name = model_name
        self.endpoint = ''

    def tokenize(self, prompt):
        if isinstance(prompt, list):
            result = []
            for p in prompt:
                response = requests.post(
                    'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-guard-3-8b/tokenize',
                    headers={
                        "RITS_API_KEY": os.getenv('RITS_API_KEY'),
                        "Content-Type": "application/json"},
                    json={
                        "model": self.model_name,
                        "prompt": p
                    }
                )

                result.append(response.json()['tokens'])
            return result
        else:
            response = requests.post(
                'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-guard-3-8b/tokenize',
                headers={
                    "RITS_API_KEY": os.getenv('RITS_API_KEY'),
                    "Content-Type": "application/json"},
                json={
                    "model": self.model_name,
                    "prompt": p
                }
            )

            return response.json()['tokens']
