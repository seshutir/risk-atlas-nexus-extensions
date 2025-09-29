import json
import logging
from json import JSONDecodeError

from risk_policy_distillation.llms.llm_component import LLMComponent
from risk_policy_distillation.models.components.context_generator import (
    ContextGenerator,
)


logger = logging.getLogger("logger")


class Summarizer:

    def __init__(self, llm_component: LLMComponent):
        """
        LLM-based component for summarizing open-text reasoning
        :param llm_component: LLM wrapper component
        """
        self.llm_component = llm_component

        cg = ContextGenerator()
        self.summarizing_context = cg.generate_summarization_context()

    def summarize(self, message):
        """
        Summarizes open text reasoning into short bulletpoins
        :param message: Open-text reasoning
        :return:
        """
        output = self.llm_component.send_request(self.summarizing_context, message)
        bulletpoints = []
        try:
            json_output = json.loads(output)
            bulletpoints = json_output["causes"]

            bulletpoints = [
                b[0].lower() if isinstance(b, list) else b.lower() for b in bulletpoints
            ]
            return bulletpoints
        except JSONDecodeError:
            pass

        return bulletpoints
