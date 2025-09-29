import numpy as np
import requests
import json
import os


OLLAMA_SERVER = "http://0.0.0.0:11435/api/chat"


# Method for posting prompts and parsing results to the LLM model
def ollama_gen(messages, model, seed=42):
    r = requests.post(
        OLLAMA_SERVER,
        json={
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_ctx": 1024 * 8,
                "temperature": 0.8,
                "seed": seed}
        },
        stream=False)
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


def generate_message(context, user, assistant=None):
    messages = [
        {
            'role': 'system',
            'content': context
        },
        {
            'role': 'user',
            'content': user
        }
    ]
    if assistant is not None:
        messages.append({
            'role': 'assistant',
            'content': assistant
        })

    return messages


def generate_and_post_message(model, context, user_text, assistant_text=None, seed=0):
    messages = generate_message(context, user_text, assistant_text)
    label = ollama_gen(messages, model, seed)
    return label['content']


def post_rits_req(model_name, model_served_name, context, prompt, response=None, temperature=0.7):
    messages = generate_message(context, prompt, response)
    response = rits_gen(messages, model_name, model_served_name, temperature)

    return response


def rits_post_guardian_message(prompt, model_name, model_served_name):
    rits_url = f'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{model_name}/v1/completions'

    if os.getenv('RITS_API_KEY') is None:
        print('rits key not found')

    response = requests.post(
        rits_url,
        headers={
            "RITS_API_KEY": os.getenv('RITS_API_KEY'),
            "Content-Type": "application/json"},
        json={
            "model": model_served_name,
            "prompt": prompt
        }
    )

    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']


def rits_gen(messages, model_name, model_served_name, temperature=0.7):
    rits_url = f'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{model_name}/v1/chat/completions'

    if os.getenv('RITS_API_KEY') is None:
        print('rits key not found')

    i = 0
    while i < 10:
        response = requests.post(
            rits_url,
            headers={
                "RITS_API_KEY": os.getenv('RITS_API_KEY'),
                "Content-Type": "application/json"},
            json={
                "model": model_served_name,
                "messages": messages,
                "temperature": temperature
            }
        )
        if response.status_code == 200:
            break
        else:
            i += 1

    return response.json()['choices'][0]['message']['content']

def post_rits_req_emb(model_name, model_served_name, input_text):
    rits_url = f'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{model_name}/v1/embeddings'

    if os.getenv('RITS_API_KEY') is None:
        print('rits key not found')

    response = requests.post(
            rits_url,
            headers={
                "RITS_API_KEY": os.getenv('RITS_API_KEY'),
                "Content-Type": "application/json"},
            json={
                "model": model_served_name,
                "input": input_text
            }
        )
    result = response.json()['data']

    embeddings = [element['embedding'] for element in result]

    return np.array(embeddings)

