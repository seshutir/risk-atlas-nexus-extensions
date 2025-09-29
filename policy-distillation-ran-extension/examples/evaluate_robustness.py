import datetime
import itertools
import json
import logging
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from datasets import load_dataset
from risk_policy_distillation.evaluation.generate_perturbed_data import (
    generate_perturbed_data,
)
from risk_policy_distillation.evaluation.quan_eval import (
    evaluate_dataset,
    evaluate_performance,
    fidelity,
    perf_degradation,
)
from risk_policy_distillation.llms.ollama_component import OllamaComponent
from risk_policy_distillation.llms.rits_component import RITSComponent
from risk_policy_distillation.models.explainers.global_explainers.global_expl import (
    GlobalExplainer,
)
from risk_policy_distillation.models.explainers.local_explainers.lime import LIME
from risk_policy_distillation.models.explainers.local_explainers.shap_vals import SHAP
from risk_policy_distillation.models.guardians.rits_guardian import RITSGuardian
from risk_policy_distillation.utils.data_util import seed_everything


def run_experiments():
    seed_everything(42)
    load_dotenv()

    # loading datasets
    with open("assets/dataset_map.json") as f:
        data_mappings = json.load(f)

    with open("assets/guardian_map.json") as f:
        guardian_mappings = json.load(f)

    local_explainers = {"lime": LIME}

    llm_components = {
        "llama3.3:70b": RITSComponent(
            "llama-3-3-70b-instruct", "meta-llama/llama-3-3-70b-instruct"
        ),
        "llama3.1:8b": OllamaComponent("llama3.1:8b"),
        "gpt-oss-20b": RITSComponent("gpt-oss-20b", "openai/gpt-oss-20b"),
    }

    all_combinations = list(
        itertools.product(
            list(llm_components.keys()),
            list(guardian_mappings.keys()),
            list(local_explainers.keys()),
            list(data_mappings.keys()),
        )
    )

    logger.info(f"Running {len(all_combinations)} experiments...")

    for experiment in all_combinations:
        llm_component_name, guardian_name, local_expl_name, dataset_name = experiment
        logger.info(
            f"Running an experiment:\n\tGuardian = {guardian_name}\n\tDataset = {dataset_name}\n\tLocal explainer = {local_expl_name}\n\tLLM Component = {llm_component_name}"
        )

        # load a guardian model
        guardian_config = guardian_mappings[guardian_name]
        guardian = RITSGuardian(
            guardian_config["rits"]["model_name"],
            guardian_config["rits"]["model_served_name"],
            guardian_config,
            guardian_name,
        )

        # load the global explanation
        path = f"results/{guardian_name}/{llm_component_name}/{local_expl_name}/{dataset_name}/global/global_expl.pkl"
        try:
            expl = GlobalExplainer(expl_path=path)

            # defining dataset
            try:
                dataframe = pd.read_csv(
                    f"datasets/perturbations/{dataset_name}/original.csv"
                )
            except FileNotFoundError:
                if "subset" in data_mappings[dataset_name]["general"].keys():
                    dataframe = load_dataset(
                        data_mappings[dataset_name]["general"]["location"],
                        data_mappings[dataset_name]["general"]["subset"],
                    )
                else:
                    dataframe = load_dataset(
                        data_mappings[dataset_name]["general"]["location"]
                    )

                dataframe = dataframe.data[
                    data_mappings[dataset_name]["split"]["subset"]
                ].table.to_pandas()

                n_samples = 100
                try:
                    dataframe = dataframe.sample(n_samples)
                except ValueError:
                    dataframe = dataframe.sample(len(dataframe))

                dataframe.to_csv(
                    f"datasets/perturbations/{dataset_name}/original.csv", index=False
                )

            # generate a perturbed dataset
            prompt_col = data_mappings[dataset_name]["data"]["prompt_col"]
            try:
                response_col = data_mappings[dataset_name]["data"]["response_col"]
            except KeyError:
                response_col = None

            perturbed_dataframes = generate_perturbed_data(
                dataframe,
                prompt_col,
                response_col,
                dataset_name,
                path="datasets/perturbation",
            )
            dataframe["index"] = np.arange(len(dataframe))

            for strategy, perturbed_df in perturbed_dataframes.items():
                logger.info(
                    f"Evaluating robustness on dataset {dataset_name} with strategy {strategy}..."
                )
                evaluation_dir = f"results/robustness/{guardian_name}/{llm_component_name}/{local_expl_name}/{dataset_name}"
                if not os.path.isdir(evaluation_dir):
                    os.makedirs(evaluation_dir)

                evaluated_path_original = os.path.join(evaluation_dir, "original.csv")
                evaluated_path_perturbed = os.path.join(
                    evaluation_dir, f"{strategy}.csv"
                )

                expl_input = response_col if not response_col is None else prompt_col
                train_evaluated_original = evaluate_dataset(
                    expl,
                    guardian,
                    dataframe,
                    expl_input,
                    "expl_answer",
                    "guard_answer",
                    evaluated_path_original,
                )
                train_evaluated_perturbed = evaluate_dataset(
                    expl,
                    guardian,
                    perturbed_df,
                    expl_input,
                    "expl_answer",
                    "guard_answer",
                    evaluated_path_perturbed,
                )

                label_col = data_mappings[dataset_name]["data"]["label_col"]
                if label_col == "":
                    label_col = "label"

                guardian_performance_original = evaluate_performance(
                    train_evaluated_original,
                    "guard_answer",
                    label_col,
                    guardian.label_names,
                    data_mappings[dataset_name]["data"]["flip_labels"],
                )
                expl_performance_original = evaluate_performance(
                    train_evaluated_original,
                    "expl_answer",
                    label_col,
                    guardian.label_names,
                    data_mappings[dataset_name]["data"]["flip_labels"],
                )

                guardian_performance_perturbed = evaluate_performance(
                    train_evaluated_perturbed,
                    "guard_answer",
                    label_col,
                    guardian.label_names,
                    data_mappings[dataset_name]["data"]["flip_labels"],
                )
                expl_performance_perturbed = evaluate_performance(
                    train_evaluated_perturbed,
                    "expl_answer",
                    label_col,
                    guardian.label_names,
                    data_mappings[dataset_name]["data"]["flip_labels"],
                )

                logger.info(
                    "Guardian performance = {} -> {}\nExpl performance = {} -> {}".format(
                        guardian_performance_original,
                        guardian_performance_perturbed,
                        expl_performance_original,
                        expl_performance_perturbed,
                    )
                )

        except FileNotFoundError:
            continue


if __name__ == "__main__":
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(
        f'logs/{datetime.datetime.now().strftime("%m_%d__%H_%M")}.log'
    )
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    run_experiments()
