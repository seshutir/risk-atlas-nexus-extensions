import datetime
import itertools
import json
import logging
import os

from dotenv import load_dotenv

from risk_policy_distillation.datasets.dataset_factory import DatasetFactory
from risk_policy_distillation.evaluation.quan_eval import (
    evaluate_dataset,
    fidelity,
    perf_degradation,
)
from risk_policy_distillation.models.explainers.global_explainers.gelpe import Gelpe
from risk_policy_distillation.models.guardians.rits_guardian import RITSGuardian


def run_baselines():
    load_dotenv()

    # load risk definitions
    with open("assets/dataset_map.json") as f:
        data_mappings = json.load(f)

    with open("assets/guardian_map.json") as f:
        guardian_mappings = json.load(f)

    all_combinations = list(
        itertools.product(list(guardian_mappings.keys()), list(data_mappings.keys()))
    )
    logger.info(f"Running {len(all_combinations)} baseline experiments...")

    for experiment in all_combinations:
        guardian_name, dataset_name = experiment
        logger.info(
            f"Running an experiment:\n\tGuardian = {guardian_name}\n\tDataset = {dataset_name}"
        )

        guardian_config = guardian_mappings[guardian_name]
        data_config = data_mappings[dataset_name]

        guardian = RITSGuardian(
            guardian_config["rits"]["model_name"],
            guardian_config["rits"]["model_served_name"],
            guardian_config,
            guardian_name,
        )

        dataset = DatasetFactory.get_dataset(data_config)

        gelpe = Gelpe(guardian, dataset, dataset_name)
        gelpe.run()

        f = f"results/GELPE/{guardian_name}/{dataset_name}/"
        if not os.path.isdir(f):
            os.makedirs(f)
            logger.info("Created results folder at: {}".format(f))

        evaluated_path = f + "results.csv"
        train_evaluated = evaluate_dataset(
            gelpe,
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
    run_baselines()
