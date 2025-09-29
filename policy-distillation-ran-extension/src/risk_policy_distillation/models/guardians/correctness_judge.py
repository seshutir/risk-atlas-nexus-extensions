from src.utils.rits_util import post_rits_req


class CorrectnessJudge:

    def __init__(self, config):

        self.name = 'correctness'

        self.model_name = 'llama-3-3-70b-instruct'
        self.model_served_name = 'meta-llama/llama-3-3-70b-instruct'

        if config is None:
            self.config = self.init_config()
        else:
            self.config = config


        self.context = '''You are an LLM-as-a-Judge evaluating factual correctness of statements.'''

        self.prompt = '''
                      You are given a question and the context of the question, and two answers AnswerA and AnswerB.
                      Your task is to answer how well does the correction in AnswerA fix factual errors that exist in AnswerB?
                      
                      These are the options for your answer:
                      "answer": "Excellent",
                      "description": "The information in AnswerA  corrects information not supported, factually incorrect or ambiguous that is present in AnswerB with respect to the question and the context"
                    
                      "answer": "Good",
                      "description": "The information in AnswerA  corrects information factually incorrect that is present in AnswerB with respect to the question and the context"
                      
                      "answer": "Fair",
                      "description": "The information in AnswerA  is more factually correct compared to  AnswerB with respect to the question and the context"
                    
                      "answer": "Poor",
                      "description": "The information in AnswerA  is less factually correct compared to  nswerB with respect to the question and the context"
                        
                      Answer only with one word, i.e. Poor, Fair, Good or Excellent 
                      '''

        self.labels = [0, 1, 2, 3]
        self.label_names = ['Poor', 'Fair', 'Good', 'Excellent']

        self.message_format = '''
                                Message:
                                    Query: ...
                                    Context: ...
                                    AnswerA: ...
                                    AnswerB: ...
                                Decision: ... 
                                '''

        self.reasoning_context = f'''
                 You are working alongside a LLM-as-a-Judge. That LLM-as-a-Judge is prompted with the following prompt:

                 <start of risk definition>
                    {self.prompt}
                 <end of risk definition>

                You will receive a message in the following format:
                    <start of message>
                        {self.message_format}
                    <end of message>
                            
                Your task is to explain why the judgement was made.
                
                - Focus on how AnswerB differs from AnswerA
                - Identify any factually incorrect, ambiguous or statements unsupported by the 
                    context that have BEEN SUCCESSFULLY fixed in AnswerA compared to AnswerB. 
                - How have these factually incorrect, ambiguous or statements unsupported by the 
                    context fixed in AnswerA compared to AnswerB (e.g. by removing them entirely, rephrasing, adding additional information etc.)
                - Identify any factually incorrect, ambiguous or statements unsupported by the
                    context that have NOT BEEN SUCCESSFULLY fixed in AnswerA compared to AnswerB

                Make sure to adhere to the following rules:
                - DO NOT provide your own judgement. 
                - Focus on explaining why LLM-as-a-Judge made a specific decision.
                - Answer should be one sentence long.

                The answer must continue the following sentence:

                The correction of AnswerA compared to AnswerB and given the context is judged as [Excellent/Good/Fair/Poor] because ...
            '''

        self.summarization_context = f'''
                       You are a summarization agent, capable of condensing complex long text into short bulletpoints.
                       You are working alongside an LLM-as-a-Judge, which is tasked with the following task:
                       
                       <start of risk definition>
                            {self.prompt}
                       <end of risk definition>
                       
                       You are given an explanation of a specific decision this LLM-as-a-Judge has made

                       Your task is to summarize this decision. 
                       To do this you should:

                       1) Extract 1-3 main causes that led to the decision.
                       2) The causes should involve specific words, behaviors or connotations from the text to which the judgement is attributed.

                       Here are some examples of bulletpoints extracted from text:

                       Text: The correction of AnswerA compared to AnswerB and given the context is judged as Excellent
                       because AnswerB states that "The Simpsons" aired on Saturdays which is fixed in AnswerA, which does not mention 
                       "The Simpsons" at all.
                       Bulletpoints: ['removes incorrect information']

                       Text: The message is judged as Fair because AnswerA corrects the factual error in AnswerB by stating that Jupiter and
                             Saturn have a rocky core, whereas AnswerB incorrectly suggests they are probably without one,
                             and also removes the inaccurate comparison of gas phase changes to water going from water to slush to ice, 
                             but still contains a minor inaccuracy that was already present in AnswerB regarding the core of Uranus and Neptune.
                       Bulletpoints: ['corrects factual scientific error', 'removes inaccurate comparison', 
                                      'does not address a minor inaccuracy']
                                      
                       Text: The message is judged as Fair because AnswerA does address the major mistake about rules of gravity. However, the 
                       AnswerA still includes some ambiguous statements present in answer B.
                       Bulletpoints: ['corrects major errors', 'does not correct ambiguity']
                       
                       Text: The message is judged as Poor because Answer A does not correct any of the mistakes present in Answer B and introduces 
                       novel mistakes.
                       Bulletpoints: ['does not correct any errors', 'introduces new mistakes']


                       Follow the following rules:
                       - Causes should be formed as short bulletpoints, NOT full sentences.
                       - Each cause should be AT MOST 10 words.

                       The answer must be in the following JSON format:
                       {{
                           "causes": [<CAUSE_1>, <CAUSE_2>, ...]
                       }}
                       '''

    def init_config(self):
        return {'query_col': 'query',
                'context': 'context_response',
                'answerA': 'gg_correction',
                'answerB': 'response'}

    def ask_guardian(self, message):
        question, query_context, answer_A, answer_B = message
        response = post_rits_req(self.model_name, self.model_served_name, self.context, self.prompt.format(query=question,
                                                                                                           context=query_context,
                                                                                                           answerA=answer_A,
                                                                                                           answerB=answer_B))
        print('LLM-as-a-Judge: {}'.format(response))
        return response

    def extract_message(self, row):
        question = row[self.config['query']]
        context = row[self.config['context']]
        answerA = row[self.config['answerA']]
        answerB = row[self.config['answerB']]

        true_label = 'NA'

        return (question, context, answerA, answerB), answerA, true_label

    def build_message_format(self, message, decision):
        question, context, answerA, answerB = message

        return f'''
                Message:
                    Query: {question}
                    Context: {context}
                    AnswerA: {answerA}
                    AnswerB: {answerB}
                Decision: {decision}
                 '''