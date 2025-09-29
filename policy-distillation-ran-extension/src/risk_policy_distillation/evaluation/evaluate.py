import json
import logging
from json import JSONDecodeError

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sympy.physics.quantum.density import fidelity
from tqdm import tqdm

from risk_policy_distillation.utils.rits_util import post_rits_req


logger = logging.getLogger("logger")


class Evaluator:

    def __init__(self, expl_model, guardian, risk_type="harm", prompt_only=False):
        self.expl_model = expl_model
        self.guardian = guardian
        self.risk_type = risk_type
        self.prompt_only = prompt_only

    def run_evaluation(self, expl, dataset, label):
        logger.info("\tRunning evaluation...")
        num_clauses = self.readability(expl)
        d = self.dissimilarity(expl)
        f = self.fidelity(expl, dataset, label, prompt_only=self.prompt_only)

        logger.info(
            "\t\t Number of clauses = {}\n\t\t"
            "Dissimilarity = {}\n\t\tFidelity = {}".format(num_clauses, d, f)
        )

        return num_clauses, d, f

    def get_num_arguments(self, clause):
        explanation_model_served_name = "meta-llama/llama-3-3-70b-instruct"
        explanation_model_name = "llama-3-3-70b-instruct"

        context = (
            "You are a helpful AI agent that can measure the complexity of sentence."
        )
        prompt = """ I have this sentence describing why a text was classified as harmful:
    
                        {text}
    
                    Can you tell me how many arguments are presented in this sentence? 
                    Different arguments will usually be separated by AND or OR.
    
                    Answer ONLY with the list of arguments <ARGS> and number of arguments <N_ARGS> in the sentence in the following JSON format:
                    {{
                        "number_of_arguments": <N_ARGS>, 
                        "arguments": <ARGS>
                    }}
                  """

        res = post_rits_req(
            explanation_model_name,
            explanation_model_served_name,
            context,
            prompt.format(text=clause),
        )
        n_args = json.loads(res)["number_of_arguments"]

        return n_args

    def readability(self, expl):
        logger.info("Evaluating readability....")
        num_clauses = len(expl)

        return num_clauses

    def dissimilarity(self, expl):
        logger.info("Evaluating dissimilarity...")
        clustering_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = clustering_model.encode(expl)

        cosine_scores = util.cos_sim(embeddings, embeddings).numpy()

        cosine_scores = np.mean(np.triu(cosine_scores))

        return cosine_scores / len(expl)

    def explained_by(self, text, clause):
        context = "You are a helpful AI assistant that can identify whether a certain topic is present in a text"
        prompt = """ I have the following text <TEXT> and a description <DESCRIPTION>:
                        <TEXT> = {text}
                        <DESCRIPTION> = {description}
    
                        Does this description accurately describe the text?
    
                        Answer ONLY with Yes or No string in the following JSON format:
                        {{
                            "answer": [Yes/No]
                        }}
                    """

        res = post_rits_req(
            *self.expl_model, context, prompt.format(text=text, description=clause)
        )
        try:
            contained = json.loads(res)["answer"]
        except JSONDecodeError:
            contained = "No"

        return contained == "Yes"

    def fidelity(self, expl, dataset, label, prompt_only):
        logger.info("Evaluating fidelity...")
        f = 0.0
        n = 0.0

        for i, row in tqdm(dataset.iterrows()):
            # get user prompt and AI assistant's response
            prompt = row["prompt"]
            response = None
            if not prompt_only:
                response = row["response"]

            # judge the message using GG
            guardian_response = self.guardian.ask_guardian((prompt, response))
            guardian_label = self.guardian.label_names.index(guardian_response)

            if (
                guardian_label == label
            ):  # only measuring if the expl captures harmful or harmless instances
                surrogate_label = False
                for clause in expl:
                    if self.explained_by(response, clause):
                        surrogate_label = True
                        break
                n += 1
                f += surrogate_label == guardian_label

        return f / n

    def coverage(self, expl, dataset):
        logger.info("Evaluating coverage...")
        coverage = 0.0
        n = 0.0
        for i in tqdm(range(len(dataset))):
            response = dataset["Response"][i]

            for clause in expl:
                if self.explained_by(response, clause):
                    coverage += 1
                    break

        return coverage / n
