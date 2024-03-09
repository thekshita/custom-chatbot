from langchain.agents import Tool, ConversationalChatAgent, AgentExecutor

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from tools.QnA import create_vector_db_tool

from utils import is_answer_formatted_in_json, output_response, _parse_source_docs
import streamlit as st


CHAT_MODEL = 'gpt-3.5-turbo'

class Agent:

    def __init__(self, option):
        self.llm = ChatOpenAI(temperature=0, model_name=CHAT_MODEL)#, api_key=st.secrets['OPENAI_API_KEY'])
        self.agent_executor = self.create_agent_executor(option)
        self.counter = 0

    def create_agent_executor(self, option):
        q_and_a_tool = create_vector_db_tool(option, llm=self.llm)
        tools = [
            Tool(
                name="QnA Tool",
                return_direct=True,
                func=lambda query: _parse_source_docs(q_and_a_tool, query),
                description="Provides accurate answers to questions related to the Disability Services Office at UW. Include relevant information and source URLs in the response."
            )
        ] #customise
        self.memory = ConversationBufferWindowMemory(llm=self.llm, k=10, memory_key="chat_history", return_messages=True,
                                                human_prefix="user", ai_prefix="assistant", input_key="input")
        
        custom_agent = ConversationalChatAgent.from_llm_and_tools(llm=self.llm,
                                                                  tools=tools,
                                                                  verbose=True,
                                                                  max_iterations=3,
                                                                  handle_parsing_errors=True,
                                                                  memory=self.memory,
                                                                  input_variables=["input", "chat_history",
                                                                                   "agent_scratchpad"],
                                                                  system_message=
                                                                  f"""
                                                                    Welcome! Let's have a conversation. Your goal is to assist the user to the best of your abilities. 
                                                                    Utilize the available tools effectively to provide accurate and helpful responses. 
                                                                    You have access to the following tools: 
                                                                    - QnA Tool: Useful for answering questions related to the Disability Services Office at UW. 
                                                             
                                                                    """,
                                                                    format_instructions=
                                                                    f"""
                                                                    Also include one source url along with the response.
                                                                    """
                                                                    #format instructions and human message if needed
                                                                  )
        return AgentExecutor.from_agent_and_tools(agent=custom_agent, tools=tools, memory=self.memory,
                                                  verbose=True)

    def query_agent(self, user_input):
        
        try:
            self.counter+=1
            print('memory: '+str(self.memory)+str(self.counter))
            response = self.agent_executor.run(input=user_input)
            if is_answer_formatted_in_json(response):
                return response
            return f"""
            {{
                "result": "{response}"
            }}"""

        except ValueError as e:
            response = str(e)
            response_prefix = "Could not parse LLM output: `\nAI: "
            if not response.startswith(response_prefix):
                raise e
            response_suffix = "`"
            if response.startswith(response_prefix):
                response = response[len(response_prefix):]
            if response.endswith(response_suffix):
                response = response[:-len(response_suffix)]
            output_response(response)
            return response



