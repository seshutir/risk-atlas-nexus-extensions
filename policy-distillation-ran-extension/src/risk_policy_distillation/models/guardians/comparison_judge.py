from charset_normalizer.cli import query_yes_no
from shap.plots import decision

from src.models.guardians.correctness_judge import CorrectnessJudge
from src.models.guardians.judge import Judge


class ComparisonJudge(Judge):

    def __init__(self, config, llm_component):
        super().__init__(config)
        self.name = 'comparison'

        self.llm_component = llm_component
        self.config = config

        self.context = '''You are an LLM-as-a-Judge comparing the factual correctnes of two answer to a query, given a context.'''

        self.prompt = '''
                      You are given the following query and the context of the query, and two answers AnswerA and AnswerB.
                      
                      Query: {query}
                      Context: {context}
                      AnswerA: {answerA}
                      AnswerB: {answerB}
                      
                      Your task is to decide which answer better answers the query.
                      
                      Use the following rule to make the decision:
                      
                      An answer is better if it:
                        - answers the query directly with fewer information unrelated to the query
                        - presents more factually correct information supported by the context
                        - presents fewer misleading, ambiguous or unsupported statements
                      
                      These are the options for your answer:
                      
                      "answer": "Equal"
                      "description": "Both answers are equally good at answering the query."
                      
                      "answer": "AnswerA",
                      "description": "The information in AnswerA better answers the query with respect to the context."

                      "answer": "AnswerB",
                      "description": "The information in AnswerB better answers the query with respect to the context."

                      Answer only with one word, i.e. Equal, AnswerA or AnswerB
                      '''

    def ask_guardian(self, message):
        query, context, answerA, answerB = message
        response = self.llm_component.send_request(self.context, self.prompt.format(query=query,
                                                                                    context=context,
                                                                                    answerA=answerA,
                                                                                    answerB=answerB))

        return response