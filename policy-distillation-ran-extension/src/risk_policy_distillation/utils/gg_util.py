import math
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "ibm-granite/granite-guardian-3.1-2b"
guardian_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
guarian_tokenizer = AutoTokenizer.from_pretrained(model_path)


def parse_output(output, input_len):
    safe_token = "No"
    unsafe_token = "Yes"
    nlogprobs = 20
    label, prob_of_risk = None, None

    if nlogprobs > 0:

        list_index_logprobs_i = [torch.topk(token_i, k=nlogprobs, largest=True, sorted=True)
                                 for token_i in list(output.scores)[:-1]]
        if list_index_logprobs_i is not None:
            prob = get_probabilities(list_index_logprobs_i)

    return prob


def get_probabilities(logprobs):
    safe_token = "No"
    unsafe_token = "Yes"
    nlogprobs = 20

    safe_token_prob = 1e-50
    unsafe_token_prob = 1e-50
    for gen_token_i in logprobs:
        for logprob, index in zip(gen_token_i.values.tolist()[0], gen_token_i.indices.tolist()[0]):
            decoded_token = guarian_tokenizer.convert_ids_to_tokens(index)
            if decoded_token.strip().lower() == safe_token.lower():
                safe_token_prob += math.exp(logprob)
            if decoded_token.strip().lower() == unsafe_token.lower():
                unsafe_token_prob += math.exp(logprob)

    probabilities = torch.softmax(
        torch.tensor([math.log(safe_token_prob), math.log(unsafe_token_prob)]), dim=0
    )

    return probabilities


def predict_proba(inputs):
    probs = []

    guardian_config = {"risk_name": "harm"}

    for i in inputs:
        with torch.no_grad():
            messages = [{"role": "user", "content": i}]

            input_ids = guarian_tokenizer.apply_chat_template(
                messages, guardian_config=guardian_config, return_tensors="pt"
            ).to(guardian_model.device)

            input_len = input_ids.shape[1]

            output = guardian_model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=20,
                return_dict_in_generate=True,
                output_scores=True,
            )

        p = parse_output(output, input_len)
        probs.append(p)

    return np.array(probs)
