class LLMJudge:

    def __init__(self, model_name, model_served_name, context, remove_ability = None):
        self.model_name = model_name
        self.model_served_name = model_served_name

        self.context = context
        if remove_ability is not None:
            self.context += '''
                                You do not have the following knowledge {a}
                            '''.format(a=remove_ability)

        self.name = 'llm_judge'



    def ask_guardian(self, x):
        return 0
