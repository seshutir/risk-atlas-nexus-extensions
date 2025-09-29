import pickle

from risk_policy_distillation.models.explainers.global_explainers.global_expl import (
    GlobalExplainer,
)


class LLooM(GlobalExplainer):

    def __init__(self, expl=None, expl_path=None):
        self.name = "lloom"
        if expl is None and expl_path is not None:
            with open(expl_path, "rb") as file:
                self.expl = pickle.load(file)
                print("Loaded LLooM explanation from {}.".format(expl_path))

        self.verifier_model = (
            "llama-3-3-70b-instruct",
            "meta-llama/llama-3-3-70b-instruct",
        )

    def predict(self, x):
        covered = False
        for e in self.expl:
            if self.covers(e, x):
                covered = True
                break

        return covered
