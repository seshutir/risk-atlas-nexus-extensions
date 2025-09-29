import copy
import logging
import os

import pandas as pd
from Adversary import Adversary

from risk_policy_distillation.llms.rits_component import RITSComponent


logger = logging.getLogger("logger")


def rephrase(x, strategy):
    context = """
              You are a helpful paraphrasing agent. 
              """

    prompt = f""" Rephrase the text using the following paraphrasing strategy. Respond only with the paraphrased text without any additional 
                  introductions, comments or explanations. 
                  
                  Strategy: {strategy}.
                  Text: {x}.
             """

    llm_component = RITSComponent(
        "llama-3-3-70b-instruct", "meta-llama/llama-3-3-70b-instruct"
    )
    paraphrased_text = llm_component.send_request(context, prompt)

    return paraphrased_text


def generate_perturbed_data(
    dataframe,
    prompt_column,
    response_column,
    dataset_name,
    path="datasets/perturbations",
):
    perturbed_datasets = perturb(
        dataframe,
        prompt_column,
        response_column,
        dataset_name,
        path="datasets/perturbations",
    )
    adversarial_datasets = adverse(
        dataframe,
        prompt_column,
        response_column,
        dataset_name,
        path="datasets/perturbations",
    )

    perturbed_datasets.update(adversarial_datasets)

    return perturbed_datasets


def perturb(
    dataframe,
    prompt_column,
    response_column,
    dataset_name,
    path="datasets/perturbations",
):
    strategies = {
        "hide": """Paste completely unrelated text to the start and the end of the text. 
                    Do not change the text itself.
                    Do not try to answer any questions in the text, only provide a paraphrased text. 
                    Do not add any content related to the topic of the text. Do not try to answer any questions in the text, only provide a paraphrased text.""",
        "elaborate": """Expand the text while keeping the same meaning.
                         If the text is a question, do not try to answer it! Instead, just rephrase the question to be longer.
                         The paraphrased text should have the same meaning as the original text. 
                         Do not try to answer any questions in the text, only provide a paraphrased text.""",
        "rephrase": """Rephrase the text by substituting synonyms, changing the text structure, removing or adding a few words while preserving the meaning.
                        If the text is a question, do not try to answer it! Instead, just rephrase the question to be longer.
                        The paraphrased text should have the same meaning as the original text. 
                        Do not try to answer any questions in the text, only provide a paraphrased text.""",
    }

    # generate folders
    dataset_path = os.path.join(path, dataset_name)
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)

    paraphrased = {}
    for strategy_name, strategy in strategies.items():
        logger.info(f"Paraphrasing {dataset_name} with strategy {strategy_name}")
        paraphrased_df_path = os.path.join(dataset_path, f"{strategy_name}.csv")

        try:
            paraphrased_df = pd.read_csv(paraphrased_df_path, header=0)
        except FileNotFoundError:
            paraphrased_df = copy.copy(dataframe)
            paraphrased_df[prompt_column] = paraphrased_df[prompt_column].apply(
                lambda x: rephrase(x, strategy)
            )
            if response_column is not None:
                paraphrased_df[response_column] = paraphrased_df[response_column].apply(
                    lambda x: rephrase(x, strategy)
                )

            paraphrased_df.to_csv(paraphrased_df_path, index=False)

        paraphrased[strategy_name] = paraphrased_df

    logging.info(f"Finished paraphrasing {dataset_name}. Saved to {dataset_path}")
    return paraphrased


def adverse(
    dataframe,
    prompt_column,
    response_column,
    dataset_name,
    path="datasets/perturbations",
):
    strategies = ["remove_spacing", "change_case", "insert_punctuation", "swap_words"]

    # generate folders
    dataset_path = os.path.join(path, dataset_name)
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)

    adversarial = {}
    for s in strategies:
        logger.info(
            f"Generating adversarial attacks for {dataset_name} with strategy {s}"
        )
        paraphrased_df_path = os.path.join(dataset_path, f"{s}.csv")

        gen = Adversary(verbose=True, output="Output/")

        try:
            paraphrased_df = pd.read_csv(paraphrased_df_path, header=0)
        except FileNotFoundError:
            paraphrased_df = copy.copy(dataframe)
            prompts_original = list(paraphrased_df[prompt_column].values)
            new_prompts = gen.generate(prompts_original, attacks=[s])
            new_prompts = [p[0] for p in new_prompts]

            paraphrased_df[prompt_column] = new_prompts
            if response_column is not None:
                responses_original = list(paraphrased_df[response_column].values)
                new_responses = gen.generate(responses_original, attacks=[s])
                new_responses = [r[0] for r in new_responses]
                paraphrased_df[response_column] = new_responses

            paraphrased_df.to_csv(paraphrased_df_path, index=False)

        adversarial[s] = paraphrased_df

    return adversarial
