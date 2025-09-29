import datetime
import itertools
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from risk_policy_distillation.datasets.dataset_factory import DatasetFactory
from risk_policy_distillation.evaluation.quan_eval import (
    evaluate_dataset,
    fidelity,
    perf_degradation,
)
from risk_policy_distillation.llms.ollama_component import OllamaComponent
from risk_policy_distillation.llms.rits_component import RITSComponent
from risk_policy_distillation.models.explainers.local_explainers.lime import LIME
from risk_policy_distillation.models.explainers.local_explainers.shap_vals import SHAP
from risk_policy_distillation.models.guardians.rits_guardian import RITSGuardian
from risk_policy_distillation.pipeline.clusterer import Clusterer
from risk_policy_distillation.pipeline.concept_extractor import Extractor
from risk_policy_distillation.pipeline.pipeline import Pipeline
from risk_policy_distillation.utils.data_util import seed_everything


def run_experiments():
    seed_everything(42)
    load_dotenv()

    # loading datasets
    with open(
        os.path.join(Path(__file__).parent.absolute(), "assets", "dataset_map.json")
    ) as f:
        data_mappings = json.load(f)

    with open(
        os.path.join(Path(__file__).parent.absolute(), "assets", "guardian_map.json")
    ) as f:
        guardian_mappings = json.load(f)

    local_explainers = {"lime": LIME, "shap": SHAP}

    llm_components = {
        "llama3.3:70b": RITSComponent(
            "llama-3-3-70b-instruct", "meta-llama/llama-3-3-70b-instruct"
        ),
        # "llama3.1:8b": OllamaComponent("llama3.1:8b"),
        # "gpt-oss-20b": RITSComponent("gpt-oss-20b", "openai/gpt-oss-20b"),
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

        # defining guardian model
        guardian_config = guardian_mappings[guardian_name]
        guardian = RITSGuardian(
            guardian_config["rits"]["model_name"],
            guardian_config["rits"]["model_served_name"],
            guardian_config,
            guardian_name,
        )

        # defining dataset
        dataset = DatasetFactory.get_dataset(data_mappings[dataset_name])

        # defining components
        llm_component = llm_components[llm_component_name]
        local_explainer = local_explainers[local_expl_name](
            dataset_name, guardian_config["label_names"]
        )

        # same concepts can be used for 'full' and 'no_fr' settings, but different are needed for 'no_lime'
        pipeline = Pipeline(
            extractor=Extractor(
                guardian,
                llm_component,
                guardian_config["criterion"],
                guardian_config["criterion_definition"],
                local_explainer,
            ),
            clusterer=Clusterer(
                llm_component,
                guardian_config["criterion_definition"],
                guardian_config["label_names"],
                n_iter=200,
            ),
            lime=True,
            fr=True,
            verbose=1,
        )

        # evaluate the global explanation
        path = f"results/{guardian_name}/{llm_component_name}/{local_expl_name}/"
        expl = pipeline.run(dataset, path=path)

        evaluated_path = path + f"{dataset_name}/results.csv"
        train_evaluated = evaluate_dataset(
            expl,
            guardian,
            dataset.train,
            dataset.expl_input,
            "expl_answer",
            "guard_answer",
            evaluated_path,
        )

        perf_degr = perf_degradation(
            train_evaluated,
            "expl_answer",
            "guard_answer",
            dataset.label_col,
            guardian_config["label_names"],
        )
        fidelity_acc, fidelity_f1 = fidelity(
            train_evaluated,
            "expl_answer",
            "guard_answer",
            guardian_config["label_names"],
        )

        logger.info(
            "Performance degradation = {}\nFidelity (acc) = {}\nFidelity (f1) = {}\n".format(
                perf_degr, fidelity_acc, fidelity_f1
            )
        )


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
