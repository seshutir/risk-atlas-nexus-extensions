import json

from risk_policy_distillation.llms.llm_component import LLMComponent
from risk_policy_distillation.models.components.context_generator import (
    ContextGenerator,
)


class Labeller:

    def __init__(self, llm_component: LLMComponent):
        """
        LLM-based component producing labels for clusters of concepts
        :param llm_component: LLM wrapper component
        """
        self.llm_component = llm_component

        cg = ContextGenerator()
        self.labeling_context = cg.generate_labeling_context()

    def label(self, additional_context, cluster, temperature):
        """
        Labels a cluster using LLM
        :param additional_context: a list of previously tried cluster labels
        :param cluster: A list of concepts to be labelled
        :param temperature: Labelling LLM temperature parameter
        :return: A common label for the cluster
        """

        # appending previously used labels to the context to encourage creativity
        context = self.labeling_context + additional_context

        prompt = """
                 Bulletpoints: {bulletpoints}
                 """.format(
            bulletpoints=cluster
        )

        # prompting LLM to label the cluster
        cluster_name = self.llm_component.send_request(
            context, prompt, temperature=temperature
        )
        cluster_name = json.loads(cluster_name)["common_reason"]

        return cluster_name
