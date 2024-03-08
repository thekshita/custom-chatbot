import json

from agent import Agent
import openai

class QueryService:

    def __init__(self, option='General'):
        self.option = option
        if option!='General':
            self.agent = Agent(option.lower())


    def ask_agent(self, question: str):
        if self.option == 'General':
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": question}]
            )
            return response['choices'][0]['message']['content']
        else:
            try:
                answer = self.agent.query_agent(user_input=question)
                response_formatted = json.loads(answer, strict=False)
            except Exception as e:
                print(f"An error occurred when querying the agent. Error: {e}")
                raise e
    
            return response_formatted['result']
