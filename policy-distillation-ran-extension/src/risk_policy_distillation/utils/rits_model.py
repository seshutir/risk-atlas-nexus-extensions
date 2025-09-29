import os

from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser
from openai import OpenAI
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from risk_policy_distillation.utils.rits_util import post_rits_req


class RITSModel:
    def __init__(self, model_name, model_path, config):
        self.model_name = model_name
        self.model_path = model_path
        self.config = config

        # params
        self.num_parallel_requests = self.config["num-parallel-requests"]

        # API key
        load_dotenv()
        rits_api_key = os.getenv("RITS_API_KEY")
        if rits_api_key is None:
            raise ValueError("RITS API key not found")

        base_url = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-guardian-3-2-3b-a800m"

        # client
        self.model_name = model_name

        self.client = OpenAI(
            api_key=rits_api_key,
            base_url=base_url,
            default_headers={"RITS_API_KEY": rits_api_key},
        )

    def generate(
        self,
        prompts,
    ):

        responses = [None] * len(prompts)
        retry_indices = list(range(len(prompts)))
        retry_count = 0

        prompts = self._format_prompts(prompts)

        generation_params = self._get_generation_params()
        total_attempts = 0
        with tqdm(total=len(prompts), desc="Generating") as pbar:
            while retry_indices and retry_count < self.config["max-retries"]:
                retry_mapping = {
                    i: original_idx for i, original_idx in enumerate(retry_indices)
                }

                prompts = self._filter_prompts(prompts, retry_indices)

                args_list = [(prompt, generation_params) for prompt in prompts]

                generation_outputs = thread_map(
                    self._get_rits_completion,
                    args_list,
                    max_workers=self.num_parallel_requests,
                )
                generation_outputs = list(generation_outputs)

                prev_retry_count = len(retry_indices)
                retry_indices = self._process_outputs(
                    generation_outputs, responses, retry_mapping
                )
                successful_generations = prev_retry_count - len(retry_indices)
                pbar.update(successful_generations)

                retry_count += 1
                total_attempts += len(prompts)

                success_rate = (
                    (len(prompts) - len(retry_indices)) / total_attempts
                ) * 100
                pbar.set_description(
                    f"progress: {len(prompts) - len(retry_indices)}/{len(prompts)} "
                    f"(parsing success rate: {success_rate:.1f}%)"
                )

        return [resp if resp is not None else {} for resp in responses]

    def _get_rits_completion(self, args):
        prompt, generation_params = args
        response = post_rits_req(
            self.model_path, self.model_name, context="", prompt=prompt
        )

        return response

    def _format_prompts(self, prompts):
        return prompts

    def _get_generation_params(self):
        params = {
            "temperature": self.config.get("temperature", 0.7),
            "max_new_tokens": self.config.get("max_new_tokens", 100),
            "min_new_tokens": self.config.get("min_new_tokens", 10),
            "top_p": self.config.get("min_new_tokens", 0.95),
            "do_sample": self.config.get("do-sample", True),
        }
        return params

    def _process_outputs(self, generation_outputs, responses, retry_mapping):
        new_retry_indices = []

        for temp_idx, temp_response in enumerate(generation_outputs):
            original_idx = retry_mapping[temp_idx]

            if temp_response.status == "200":
                responses[original_idx] = temp_response
            else:
                print(f"index {original_idx}: parsing failed")
                print(f"error: {e}")
                new_retry_indices.append(original_idx)
                responses[original_idx] = None

        return new_retry_indices

    @staticmethod
    def _filter_prompts(prompts, indices):
        return [prompts[i] for i in indices]
