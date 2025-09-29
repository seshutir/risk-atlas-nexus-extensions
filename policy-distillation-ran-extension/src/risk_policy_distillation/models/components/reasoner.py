from risk_policy_distillation.llms.llm_component import LLMComponent
from risk_policy_distillation.models.components.context_generator import (
    ContextGenerator,
)
from risk_policy_distillation.models.guardians.judge import Judge


class Reasoner:

    def __init__(self, llm_component: LLMComponent, guardian: Judge):
        """
        LLM-based component for generating open-text reasoning about a decision
        :param llm_component: LLM wrapper component
        :param guardian: LLM-as-a-Judge being explained
        """
        self.llm_component = llm_component

        cg = ContextGenerator()
        self.reasoning_context = cg.generate_reasoning_context(
            guardian.task, guardian.definition, guardian.label_names
        )

    def reason(self, message) -> str:
        """
        Produces LLM-generated open-text reasoning on a message.
        :param message: A message containing LLM-as-a-Judge input and its decision
        :return: Open-text reasoning about the decision
        """
        reasoning = self.llm_component.send_request(self.reasoning_context, message)

        return reasoning
