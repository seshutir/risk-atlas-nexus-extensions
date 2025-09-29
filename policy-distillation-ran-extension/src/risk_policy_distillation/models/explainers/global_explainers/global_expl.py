import json
import logging
import pickle
from datetime import datetime, timedelta
from json import JSONDecodeError

import numpy as np

from risk_policy_distillation.llms.llm_component import LLMComponent
from risk_policy_distillation.llms.ollama_component import OllamaComponent
from risk_policy_distillation.llms.rits_component import RITSComponent
from risk_policy_distillation.utils.rits_util import post_rits_req


logger = logging.getLogger("logger")


class GlobalExplainer:

    def __init__(
        self, expl_graph=None, expl_path=None, guardian=None, name="GloVE", perc=1.0
    ):
        self.expl_graph = expl_graph
        self.expl_path = expl_path

        self.guardian = guardian

        if expl_graph is None and expl_path is not None:
            self.expl_graph = self.load(self.expl_path)

        expl = self.expl_graph.get_expl()
        self.rules, self.despites, self.predictions, self.importances = (
            self.unpack_expl(expl)
        )

        keep_rules = int(perc * len(self.rules))
        self.rules = self.rules[0:keep_rules]
        self.despites = self.despites[0:keep_rules]
        self.predictions = self.predictions[0:keep_rules]

        logger.info("Loaded {} rules".format(len(self.rules)))

        self.name = name

        self.verifier = RITSComponent(
            "llama-3-3-70b-instruct", "meta-llama/llama-3-3-70b-instruct"
        )

    def unpack_expl(self, rules):
        arguments = []
        predictions = []
        despites = []
        importances = []

        for r in rules:
            arguments.append(r.argument_because)
            if not len(r.argument_despite):
                despites.append("none")
            else:
                despites.append(r.argument_despite)

            predictions.append(r.prediction)
            importances.append(r.importance)

        arguments = list(reversed(arguments))
        despites = list(reversed(despites))
        predictions = list(reversed(predictions))
        importances = list(reversed(importances))
        return arguments, despites, predictions, importances

    def predict(self, x):
        start_time = datetime.now()

        for i, r in enumerate(self.rules):
            current_time = datetime.now()
            time_difference = current_time - start_time
            three_minutes = timedelta(minutes=3)

            time_expired = time_difference >= three_minutes
            if time_expired:
                logger.info("Time expired")
                return None

            # check if main clause is covered
            if not self.covers(r, x):
                continue

            # check if one of the despite clauses is covered
            if self.despites[i] != "none":
                for d in self.despites[i]:
                    if self.covers(d, x):
                        return self.predictions[i]
            else:
                return self.predictions[i]

        return None

    def covers(self, expl_clause, x):
        context = """You are a verifier model that can evaluate whether a specific concept is contained in a text.
                         """

        prompt = """
                        Here is a text: {sentence}.

                        Can this text be described by the following concept: {clause}?

                        Answer with Yes only if you are certain the concept is present in the text.

                        Answer with Yes or No and your step-by-step reasoning <REASONING> in the following JSON format:
                        {{
                           "reasoning": <REASONING>,
                           "answer": "Yes"/"No"
                        }}
                        """.format(
            sentence=x, clause=expl_clause
        )

        try_num = 1
        i = 0
        response = ""

        while i < try_num:
            try:
                response = self.verifier.send_request(context, prompt, temperature=0.0)
                response = json.loads(response)["answer"]
                break
            except JSONDecodeError:
                i += 1

        return response == "Yes"

    def load(self, path):
        with open(path, "rb") as file:
            graph = pickle.load(file)

            logger.info("Loaded explanation at {}".format(path))

            return graph

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self.expl_graph, file)

            logger.info("Stored explanation at {}".format(path))
