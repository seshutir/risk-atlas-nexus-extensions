from datasets import load_dataset
from risk_policy_distillation.datasets.prompt_dataset import PromptDataset
from risk_policy_distillation.datasets.prompt_response_dataset import (
    PromptResponseDataset,
)


class DatasetFactory:

    def __init__(self):
        pass

    @staticmethod
    def get_dataset(config):
        try:
            dataframe = load_dataset(config["general"]["location"])
        except:
            dataframe = load_dataset(
                config["general"]["location"], config["general"]["subset"]
            )

        if config["data"]["type"] == "prompt":
            return PromptDataset(dataframe=dataframe, config=config)
        elif config["data"]["type"] == "prompt_response":
            return PromptResponseDataset(dataframe=dataframe, config=config)
