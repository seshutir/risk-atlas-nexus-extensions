class LLMComponent:

    def __init__(self, connection_type='rits'):
        """
        A wrapper objected for calling an LLM
        :param connection_type: supported types so far are 'rits' and 'ollama'
        """
        self.connection_type = connection_type

        # TODO: assert connection type is supported

    def send_request(self, context, prompt, response=None, temperature=0.7):
        pass

    def generate_message(self, context, user, assistant=None):
        messages = [
            {
                'role': 'system',
                'content': context
            },
            {
                'role': 'user',
                'content': user
            }
        ]
        if assistant is not None:
            messages.append({
                'role': 'assistant',
                'content': assistant
            })

        return messages